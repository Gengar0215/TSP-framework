import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import glob

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


parser = argparse.ArgumentParser(description="Launch the multi-index faiss retriever.")
parser.add_argument("--multi_index_dir", type=str, required=True, help="Directory containing multiple index folders.")
parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Name of the retriever model.")
parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for FAISS index.")
parser.add_argument("--num_gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
parser.add_argument("--gpu_allocation_strategy", type=str, default="round_robin", 
                    choices=["round_robin", "memory_balanced", "single_gpu"], 
                    help="Strategy for allocating indices to GPUs.")

args = parser.parse_args()

def load_json_corpus(json_path: str):
    """Load corpus from a JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_json_docs(corpus, doc_idxs, docids=None):
    """Load documents from JSON-based corpus using indices"""
    results = []
    for idx in doc_idxs:
        if isinstance(idx, (int, np.integer)) and 0 <= idx < len(docids):
            # Get document ID from docids list
            doc_id = docids[idx]
            # Get document content from corpus dictionary
            if doc_id in corpus:
                doc_content = corpus[doc_id]
                # Parse the document content if it's a string
                if isinstance(doc_content, str):
                    results.append({
                        "title": doc_id,
                        "text": doc_content,
                        "contents": doc_content,
                        "doc_id": doc_id
                    })
                else:
                    results.append(doc_content)
            else:
                results.append({"title": "", "text": "", "contents": "", "doc_id": ""})
        else:
            # Handle invalid indices
            results.append({"title": "", "text": "", "contents": "", "doc_id": ""})
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class MultiIndexRetriever:
    """Retriever that handles multiple indices simultaneously"""
    
    def __init__(self, multi_index_dir, retriever_model, topk=10, faiss_gpu=True, num_gpus=-1, gpu_allocation_strategy="round_robin"):
        self.multi_index_dir = multi_index_dir
        self.topk = topk
        self.faiss_gpu = faiss_gpu
        self.num_gpus = num_gpus
        self.gpu_allocation_strategy = gpu_allocation_strategy
        
        # Multi-GPU setup
        self.available_gpus = []
        self.gpu_index_mapping = {}  # index_name -> gpu_id
        self.gpu_memory_usage = {}   # gpu_id -> memory_usage_estimate
        
        if self.faiss_gpu:
            self._setup_gpus()
        
        self.indices = {}
        self.corpora = {}
        self.docids = {}
        self.index_names = []
        
        # Load multiple indices from directory
        self._load_multiple_indices()
            
        self.encoder = Encoder(
            model_name = "e5",
            model_path = retriever_model,
            pooling_method = "mean",
            max_length = 256,
            use_fp16 = True
        )
        self.batch_size = 512

    def _setup_gpus(self):
        """Setup available GPUs and initialize memory tracking"""
        import torch
        
        total_gpus = torch.cuda.device_count()
        if total_gpus == 0:
            print("Warning: No available GPUs detected, using CPU mode")
            self.faiss_gpu = False
            return
            
        if self.num_gpus == -1:
            self.num_gpus = total_gpus
        else:
            self.num_gpus = min(self.num_gpus, total_gpus)
            
        self.available_gpus = list(range(self.num_gpus))
        self.gpu_memory_usage = {gpu_id: 0 for gpu_id in self.available_gpus}
        
        print(f"Using {self.num_gpus} GPUs: {self.available_gpus}")
        
        # Print GPU information
        for gpu_id in self.available_gpus:
            gpu_props = torch.cuda.get_device_properties(gpu_id)
            total_memory = gpu_props.total_memory / 1024**3  # GB
            print(f"GPU {gpu_id}: {gpu_props.name}, Total memory: {total_memory:.1f}GB")

    def _allocate_gpu_for_index(self, index_name: str, index_size: int = 0) -> int:
        """为索引分配GPU"""
        if not self.faiss_gpu or not self.available_gpus:
            return -1  # CPU mode
            
        if self.gpu_allocation_strategy == "single_gpu":
            return self.available_gpus[0]
        elif self.gpu_allocation_strategy == "round_robin":
            # 简单轮询分配
            allocated_count = len(self.gpu_index_mapping)
            return self.available_gpus[allocated_count % len(self.available_gpus)]
        elif self.gpu_allocation_strategy == "memory_balanced":
            # 选择当前内存使用最少的GPU
            return min(self.gpu_memory_usage.keys(), key=lambda x: self.gpu_memory_usage[x])
        else:
            return self.available_gpus[0]

    def _load_index_to_gpu(self, cpu_index, gpu_id: int):
        """Load index to the specified GPU"""
        if gpu_id == -1:
            return cpu_index  # Return CPU index
            
        try:
            # Create GPU resource configuration
            gpu_resource = faiss.StandardGpuResources()
            
            # Move index to specified GPU
            gpu_index = faiss.index_cpu_to_gpu(gpu_resource, gpu_id, cpu_index)
            
            print(f"Successfully loaded index to GPU {gpu_id}")
            return gpu_index
            
        except Exception as e:
            print(f"Failed to load index to GPU {gpu_id}: {str(e)}")
            print("Using CPU mode")
            return cpu_index

    def _load_multiple_indices(self):
        """Load multiple indices from a directory structure"""
        # Find all index directories
        index_dirs = glob.glob(os.path.join(self.multi_index_dir, "index__*"))
        
        if not index_dirs:
            raise ValueError(f"No index directories found in {self.multi_index_dir}")
        
        print(f"Found {len(index_dirs)} index directories")
        
        for index_dir in tqdm(index_dirs, desc="Loading indices"):
            index_name = os.path.basename(index_dir)
            
            # Expected files in each index directory
            faiss_index_path = os.path.join(index_dir, "faiss_index.bin")
            documents_path = os.path.join(index_dir, "documents.json")
            docids_path = os.path.join(index_dir, "docids.json")
            
            if not all(os.path.exists(p) for p in [faiss_index_path, documents_path, docids_path]):
                print(f"Warning: Skipping {index_name}, missing required files")
                continue
            
            try:
                # Load FAISS index to CPU first
                cpu_index = faiss.read_index(faiss_index_path)
                index_size_estimate = cpu_index.ntotal * cpu_index.d * 4  # Rough estimate in bytes
                
                # Allocate GPU for this index
                assigned_gpu = self._allocate_gpu_for_index(index_name, index_size_estimate)
                
                # Load index to assigned GPU or keep on CPU
                if assigned_gpu != -1 and self.faiss_gpu:
                    final_index = self._load_index_to_gpu(cpu_index, assigned_gpu)
                    self.gpu_index_mapping[index_name] = assigned_gpu
                    self.gpu_memory_usage[assigned_gpu] += index_size_estimate
                    print(f"Index {index_name} allocated to GPU {assigned_gpu}")
                else:
                    final_index = cpu_index
                    print(f"Index {index_name} using CPU mode")
                
                # Load documents and docids
                documents = load_json_corpus(documents_path)
                with open(docids_path, 'r', encoding='utf-8') as f:
                    docids = json.load(f)
                
                self.indices[index_name] = final_index
                self.corpora[index_name] = documents
                self.docids[index_name] = docids
                self.index_names.append(index_name)
                
                print(f"Loaded {index_name}: {len(documents)} documents, {len(docids)} docids, index size: {cpu_index.ntotal}")
                
            except Exception as e:
                print(f"Error loading {index_name}: {str(e)}")
                continue
        
        # Print final GPU allocation summary
        if self.faiss_gpu and self.available_gpus:
            print("\n=== GPU Allocation Summary ===")
            for gpu_id in self.available_gpus:
                indices_on_gpu = [name for name, gpu in self.gpu_index_mapping.items() if gpu == gpu_id]
                memory_mb = self.gpu_memory_usage[gpu_id] / (1024**2)
                print(f"GPU {gpu_id}: {len(indices_on_gpu)} indices, estimated memory usage: {memory_mb:.1f}MB")
                for idx_name in indices_on_gpu:
                    print(f"  - {idx_name}")

    def _search_single_index(self, query: str, index_name: str, num: int = None, return_score: bool = False):
        """Search within a single index"""
        if num is None:
            num = self.topk
            
        if index_name not in self.indices:
            if return_score:
                return [], []
            return []
        
        query_emb = self.encoder.encode(query)
        scores, idxs = self.indices[index_name].search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        
        # Load documents from the specific corpus
        results = load_json_docs(self.corpora[index_name], idxs, self.docids[index_name])
        
        # Add index information to results
        for result in results:
            if isinstance(result, dict):
                result['source_index'] = index_name
        
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def search(self, query: str, num: int = None, return_score: bool = False, target_indices: List[str] = None):
        """Search across all indices or specified indices"""
        if num is None:
            num = self.topk
        
        # If target_indices is specified, only search those
        search_indices = target_indices if target_indices else self.index_names
        
        all_results = []
        all_scores = []
        
        for index_name in search_indices:
            results, scores = self._search_single_index(query, index_name, num, return_score=True)
            
            # Add scores and results with index information
            for i, (result, score) in enumerate(zip(results, scores)):
                all_results.append(result)
                all_scores.append(score)
        
        # Sort by score (descending) and keep top num results
        if all_scores:
            sorted_pairs = sorted(zip(all_results, all_scores), key=lambda x: x[1], reverse=True)
            all_results, all_scores = zip(*sorted_pairs[:num])
            all_results = list(all_results)
            all_scores = list(all_scores)
        
        if return_score:
            return all_results, all_scores
        else:
            return all_results

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False, target_indices: List[str] = None):
        """Batch search across multiple indices"""
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        
        for query in tqdm(query_list, desc='Multi-index batch retrieval'):
            query_results, query_scores = self.search(query, num, return_score=True, target_indices=target_indices)
            results.append(query_results)
            scores.append(query_scores)
        
        if return_score:
            return results, scores
        else:
            return results
    
    def get_available_indices(self):
        """Return list of available index names"""
        return self.index_names
    
    def get_index_info(self):
        """Return information about loaded indices"""
        info = {}
        for index_name in self.index_names:
            gpu_id = self.gpu_index_mapping.get(index_name, -1)
            info[index_name] = {
                'num_documents': len(self.corpora[index_name]),
                'index_size': self.indices[index_name].ntotal if hasattr(self.indices[index_name], 'ntotal') else 0,
                'gpu_id': gpu_id,
                'gpu_name': f"GPU {gpu_id}" if gpu_id != -1 else "CPU"
            }
        return info

    def get_gpu_allocation_info(self):
        """Return detailed GPU allocation information"""
        if not self.faiss_gpu:
            return {"mode": "CPU", "message": "GPU mode not enabled"}
        
        gpu_info = {}
        for gpu_id in self.available_gpus:
            indices_on_gpu = [name for name, gpu in self.gpu_index_mapping.items() if gpu == gpu_id]
            memory_mb = self.gpu_memory_usage[gpu_id] / (1024**2)
            
            # Get GPU properties
            import torch
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(gpu_id)
                total_memory_gb = gpu_props.total_memory / (1024**3)
                gpu_name = gpu_props.name
            else:
                total_memory_gb = 0
                gpu_name = "Unknown"
            
            gpu_info[f"gpu_{gpu_id}"] = {
                "gpu_id": gpu_id,
                "gpu_name": gpu_name,
                "total_memory_gb": total_memory_gb,
                "estimated_usage_mb": memory_mb,
                "num_indices": len(indices_on_gpu),
                "indices": indices_on_gpu
            }
        
        return {
            "mode": "Multi-GPU",
            "strategy": self.gpu_allocation_strategy,
            "total_gpus": len(self.available_gpus),
            "gpus": gpu_info
        }


#####################################
# FastAPI server below
#####################################

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    target_indices: Optional[List[str]] = None  # New field for specifying which indices to search

app = FastAPI()

# Initialize the global retriever
retriever = MultiIndexRetriever(
    multi_index_dir=args.multi_index_dir,
    retriever_model=args.retriever_model,
    topk=args.topk,
    faiss_gpu=args.faiss_gpu,
    num_gpus=args.num_gpus,
    gpu_allocation_strategy=args.gpu_allocation_strategy
)

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true,
      "target_indices": ["index__astropy__astropy-7166"]  # Optional: specify which indices to search
    }
    """
    if not request.topk:
        request.topk = retriever.topk  # fallback to default

    # Multi-index retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores,
        target_indices=request.target_indices
    )
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}

@app.get("/indices")
def get_indices_endpoint():
    """
    Get information about available indices
    """
    return {
        "available_indices": retriever.get_available_indices(),
        "index_info": retriever.get_index_info(),
        "gpu_allocation_info": retriever.get_gpu_allocation_info()
    }

@app.get("/gpu_info")
def get_gpu_info_endpoint():
    """
    Get detailed GPU allocation information
    """
    return retriever.get_gpu_allocation_info()

if __name__ == "__main__":
    # Launch the server
    uvicorn.run(app, host="0.0.0.0", port=8000) 