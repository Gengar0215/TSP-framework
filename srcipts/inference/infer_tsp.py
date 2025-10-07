import pandas as pd
import argparse
import json
import numpy as np
import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

def load_model_and_tokenizer(model_name, tensor_parallel_size=1, gpu_devices="0,1"):
    """Load vllm model and tokenizer with multi-GPU support"""
    import os
    
    # Set CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, tokenizer

def read_problem_from_parquet(parquet_file, index=0):
    """Read problem_statement and row data from parquet file"""
    try:
        df = pd.read_parquet(parquet_file)
        if "problem_statement" not in df.columns:
            raise ValueError("Column 'problem_statement' not found in parquet file")
        
        if index >= len(df):
            raise ValueError(f"Index {index} out of range, data has {len(df)} rows")
        
        row_data = df.iloc[index].to_dict()
        problem_statement = row_data["problem_statement"]
        return problem_statement, row_data
    except Exception as e:
        print(f"Failed to read parquet file: {e}")
        return None, None

def create_first_round_prompt(problem_statement):
    """Create first round prompt for think and search"""
    base_prompt = """In this task, you will be given a software development issue from a real GitHub repository along with a codebase. You must complete the task in the following sequence:
Analysis & Planning: Within <think></think> tags:
    - Analyze the issue
    - Break down the task
    - Design search queries to retrieve contextual code relevant to the issue
Single Search Round:
    - Call the search engine via <search></search>
    - Search queries should only contain three class names or method names related to the issue
Resolution:
After receiving the search results, solve the issue by generating a single patch file that can be directly applied to the repository using git apply. Format your response as:
<patch>
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
def euclidean(a, b):
- while b:
- a, b = b, a % b
- return a
+ if b == 0:
+ return a
+ return euclidean(b, a % b)
def bresenham(x0, y0, x1, y1):
points = []
dx = abs(x1 - x0)
dy = abs(y1 - y0)
- sx = 1 if x0 < x1 else -1
- sy = 1 if y0 < y1 else -1
- err = dx - dy
+ x, y = x0, y0
+ sx = -1 if x0 > x1 else 1
+ sy = -1 if y0 > y1 else 1

- while True:
- points.append((x0, y0))
- if x0 == x1 and y0 == y1:
- break
- e2 = 2 * err
- if e2 > -dy:
+ if dx > dy:
+ err = dx / 2.0
+ while x != x1:
+ points.append((x, y))
err -= dy
- x0 += sx
- if e2 < dx:
- err += dx
- y0 += sy
+ if err < 0:
+ y += sy
+ err += dx
+ x += sx
+ else:
+ err = dy / 2.0
+ while y != y1:
+ points.append((x, y))
+ err -= dx
+ if err < 0:
+ x += sx
+ err += dy
+ y += sy
+ points.append((x, y))
return points
</patch>

Here is the software development issue:
<issue>
{problem_statement}
</issue>"""
    
    return base_prompt.format(problem_statement=problem_statement)

def create_second_round_prompt(first_round_prompt, first_round_response, information):
    """Create second round prompt with first round response and retrieval information"""
    second_round_prompt = f"{first_round_prompt}\n\n{first_round_response}\n\n{information}\n\nNow you need to patch the code."
    return second_round_prompt

def extract_search_queries(response):
    """Extract content between <search></search> tags from response"""
    search_start = "<search>"
    search_end = "</search>"
    
    start_idx = response.find(search_start)
    end_idx = response.find(search_end)
    
    if start_idx == -1 or end_idx == -1:
        return []
    
    search_content = response[start_idx + len(search_start):end_idx].strip()
    
    if not search_content:
        return []
    
    # Split by spaces into multiple queries
    queries = [query.strip() for query in search_content.split() if query.strip()]
    return queries

def extract_target_index(index_path):
    """Extract target_indices from index_path"""
    if not index_path:
        return None
    
    # Get content after the last /
    target_index = index_path.split("/")[-1]
    return target_index

def format_retrieval_results(retrieval_result):
    """
    Format retrieval results into <information></information> tags
    Extract doc_id and contents, concatenate as: doc1: xxx, doc2: xxx
    """
    if not retrieval_result or "result" not in retrieval_result:
        return "<information></information>"
    
    results = retrieval_result["result"]
    if not results:
        return "<information></information>"
    
    formatted_docs = []
    doc_count = 1
    
    # Iterate through results for each query (result is nested array)
    for query_results in results:
        if isinstance(query_results, list):
            # Iterate through each result for this query
            for result_item in query_results:
                if "document" in result_item:
                    doc = result_item["document"]
                    contents = doc.get("contents", "")
                    
                    if contents:
                        formatted_docs.append(f"doc{doc_count}: {contents}")
                        doc_count += 1
    
    if not formatted_docs:
        return "<information></information>"
    
    # Join all documents with commas
    content = ", ".join(formatted_docs)
    return f"<information>{content}</information>"

def call_retrieval_api(queries, target_indices, topk=2, return_scores=True, api_url="http://localhost:8000/retrieve"):
    """Call retrieval API"""
    if not queries or not target_indices:
        return None
    
    request_data = {
        "queries": queries,
        "topk": topk,
        "return_scores": return_scores,
        "target_indices": target_indices
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(api_url, json=request_data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        return None

def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_result_to_jsonl(result_data, output_file="result.jsonl"):
    """Save result to jsonl file"""
    try:
        # Convert non-serializable objects
        serializable_data = convert_to_serializable(result_data)
        
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        pass  # Silently handle errors

def truncate_response_after_search(response):
    """Truncate response after </search> tag"""
    search_end_tag = "</search>"
    if search_end_tag in response:
        truncate_index = response.find(search_end_tag) + len(search_end_tag)
        return response[:truncate_index]
    return response

def generate_response_single(llm, tokenizer, prompt, max_new_tokens=16384):
    """Generate single model response"""
    messages = [
        {"role": "system", "content": "You are an excellent bug fixer with access to a codebase search tool."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        stop=None
    )
    
    outputs = llm.generate([text], sampling_params)
    response = outputs[0].outputs[0].text
    
    return response

def process_single_item(llm, tokenizer, problem_statement, row_data, index, args):
    """Process single data item with complete two-stage pipeline"""
    try:
        # Stage 1: Generate first round response for think and search
        first_round_prompt = create_first_round_prompt(problem_statement)
        first_round_response = generate_response_single(llm, tokenizer, first_round_prompt, args.max_new_tokens)
        
        # Truncate response after search
        truncated_response = truncate_response_after_search(first_round_response)
        
        # Extract search queries
        search_queries = extract_search_queries(truncated_response)
        
        # Extract target index
        target_index = extract_target_index(row_data.get("index_path"))
        
        # Call retrieval API
        retrieval_result = None
        information = "<information></information>"
        if search_queries and target_index:
            retrieval_result = call_retrieval_api(search_queries, [target_index], topk=args.topk, api_url=args.api_url)
            if retrieval_result is not None:
                information = format_retrieval_results(retrieval_result)
        
        # Stage 2: Generate final patch based on retrieval results
        second_round_prompt = create_second_round_prompt(first_round_prompt, truncated_response, information)
        second_round_response = generate_response_single(llm, tokenizer, second_round_prompt, args.max_new_tokens)
        
        # Prepare result data
        result_data = {
            "index": index,
            "problem_statement": problem_statement,
            "first_round_response": truncated_response,
            "second_round_response": second_round_response,
            "search_queries": search_queries,
            "target_index": target_index,
            "retrieval_result": retrieval_result,
            "information": information,
            "row_data": row_data
        }
        
        return result_data
        
    except Exception as e:
        print(f"Error processing item {index}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Combined two-stage inference: think/search + patch generation")
    parser.add_argument("--parquet_file", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--index", type=int, default=None, help="Specific data index to process, if not specified, process all data")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    parser.add_argument("--output_file", type=str, default="result_combined.jsonl", help="Output result file path")
    parser.add_argument("--start_index", type=int, default=0, help="Start processing index")
    parser.add_argument("--end_index", type=int, default=None, help="End processing index (exclusive)")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/retrieve", help="Retrieval API URL")
    parser.add_argument("--topk", type=int, default=2, help="Top-k results to retrieve")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--gpu_devices", type=str, default="0,1", help="GPU device numbers to use")
    
    args = parser.parse_args()
    
    # First read parquet file to get total number of rows
    try:
        df = pd.read_parquet(args.parquet_file)
        total_rows = len(df)
        print(f"Data loaded: {total_rows} items")
    except Exception as e:
        print(f"Failed to read parquet file: {e}")
        return
    
    # Load model (only once)
    print(f"Loading model: {args.model_name}")
    print(f"Using GPU devices: {args.gpu_devices}")
    llm, tokenizer = load_model_and_tokenizer(args.model_name, args.tensor_parallel_size, args.gpu_devices)
    
    # Determine processing range
    if args.index is not None:
        # Process single index
        indices_to_process = [args.index]
        print(f"Processing single item, index: {args.index}")
    else:
        # Process all data or specified range
        start_idx = args.start_index
        end_idx = args.end_index if args.end_index is not None else total_rows
        indices_to_process = list(range(start_idx, min(end_idx, total_rows)))
        print(f"Processing range: {start_idx} to {end_idx-1} ({len(indices_to_process)} items)")
    
    successful_count = 0
    failed_count = 0
    
    # Process data sequentially (since each item needs two-stage processing)
    for index in tqdm(indices_to_process, desc="Processing items", unit="item"):
        try:
            problem_statement, row_data = read_problem_from_parquet(args.parquet_file, index)
            
            if problem_statement is None:
                failed_count += 1
                continue
            
            # Process with complete two-stage pipeline
            result_data = process_single_item(llm, tokenizer, problem_statement, row_data, index, args)
            
            if result_data is not None:
                # Save result to jsonl file
                save_result_to_jsonl(result_data, args.output_file)
                successful_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            failed_count += 1
            continue
    
    print(f"\nProcessing completed!")
    print(f"Success: {successful_count} | Failed: {failed_count}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 