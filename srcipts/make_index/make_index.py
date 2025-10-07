import json
import os
import ast
import jedi
import shutil
import traceback
import subprocess
import numpy as np
import torch
import faiss
from filelock import FileLock
from typing import Any, List, Dict
from datasets import load_from_disk, load_dataset
from git import Repo
from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading

from swebench.inference.make_datasets.utils import list_files, string_to_bool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Thread-safe progress bar
class ThreadSafeProgressBar:
    def __init__(self, total, desc):
        self.pbar = tqdm(total=total, desc=desc)
        self.lock = threading.Lock()
    
    def update(self, n=1):
        with self.lock:
            self.pbar.update(n)
    
    def close(self):
        self.pbar.close()

# Global lock for thread safety
git_lock = threading.Lock()

# Context manager for Git repository version switching to ensure file processing at specific commit
class ContextManager:
    def __init__(self, repo_path, base_commit, verbose=False):
        self.repo_path = Path(repo_path).resolve().as_posix()
        self.base_commit = base_commit
        self.verbose = verbose
        self.repo = None

    def __enter__(self):
        if self.verbose:
            print(f"Switching to {self.base_commit}")
        try:
            with git_lock:
                self.repo = Repo(self.repo_path)
                self.repo.git.reset("--hard", self.base_commit)
                self.repo.git.clean("-fdxq")
        except Exception as e:
            logger.error(f"Failed to switch to {self.base_commit}")
            logger.error(e)
            raise e
        return self

    def get_readme_files(self):
        files = os.listdir(self.repo_path)
        files = list(filter(lambda x: os.path.isfile(x), files))
        files = list(filter(lambda x: x.lower().startswith("readme"), files))
        return files

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Document encoding functions
def file_name_and_contents(filename, relative_path):
    text = relative_path + "\n"
    with open(filename) as f:
        text += f.read()
    return text

def file_name_and_documentation(filename, relative_path):
    text = relative_path + "\n"
    try:
        with open(filename) as f:
            node = ast.parse(f.read())
        data = ast.get_docstring(node)
        if data:
            text += f"{data}"
        for child_node in ast.walk(node):
            if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                data = ast.get_docstring(child_node)
                if data:
                    text += f"\n\n{child_node.name}\n{data}"
    except Exception as e:
        logger.error(e)
        logger.error(f"Failed to parse file {str(filename)}. Using simple filecontent.")
        with open(filename) as f:
            text += f.read()
    return text

def file_name_and_docs_jedi(filename, relative_path):
    text = relative_path + "\n"
    with open(filename) as f:
        source_code = f.read()
    try:
        script = jedi.Script(source_code, path=filename)
        module = script.get_context()
        docstring = module.docstring()
        text += f"{module.full_name}\n"
        if docstring:
            text += f"{docstring}\n\n"
        abspath = Path(filename).absolute()
        names = [
            name
            for name in script.get_names(all_scopes=True, definitions=True, references=False)
            if not name.in_builtin_module()
        ]
        for name in names:
            try:
                origin = name.goto(follow_imports=True)[0]
                if origin.module_name != module.full_name:
                    continue
                if name.parent().full_name != module.full_name:
                    if name.type in {"statement", "param"}:
                        continue
                full_name = name.full_name
                text += f"{full_name}\n"
                docstring = name.docstring()
                if docstring:
                    text += f"{docstring}\n\n"
            except:
                continue
    except Exception as e:
        logger.error(e)
        logger.error(f"Failed to parse file {str(filename)}. Using simple filecontent.")
        text = f"{relative_path}\n{source_code}"
        return text
    return text

def file_name_and_ast_parsed(filename, relative_path):
    """
    Parse Python file using AST, return each class and function as separate documents with line numbers
    """
    documents = {}
    try:
        with open(filename) as f:
            source_code = f.read()
            source_lines = source_code.splitlines()
        tree = ast.parse(source_code)
        
        # Process module-level documentation
        module_doc = ast.get_docstring(tree)
        if module_doc:
            module_lineno = tree.body[0].lineno if tree.body and hasattr(tree.body[0], 'lineno') else 1
            numbered_doc = '\n'.join(f"{module_lineno + i} {line}" for i, line in enumerate(module_doc.splitlines()))
            documents[f"{relative_path}:module"] = (
                f"{relative_path}\n"
                f"Module Documentation:\n"
                f"{numbered_doc}"
            )
        
        # Walk through AST tree
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                node_type = "class" if isinstance(node, ast.ClassDef) else "function"
                docstring = ast.get_docstring(node)
                node_source = ast.unparse(node)
                start_line = node.lineno
                numbered_source = '\n'.join(f"{start_line + i} {line}" for i, line in enumerate(node_source.splitlines()))
                
                content = (
                    f"{relative_path}\n"
                    f"{node_type.title()}: {node.name}\n"
                )
                if docstring:
                    numbered_docstring = '\n'.join(f"{start_line + 1 + i} {line}" for i, line in enumerate(docstring.splitlines()))
                    content += f"Documentation:\n{numbered_docstring}\n"
                content += f"\nSource Code:\n{numbered_source}"
                
                key = f"{relative_path}:{node.name}"
                documents[key] = content
                
    except Exception as e:
        logger.error(f"Failed to parse file {filename} with AST: {e}")
        numbered_source = '\n'.join(f"{i+1} {line}" for i, line in enumerate(source_code.splitlines()))
        documents[relative_path] = f"{relative_path}\n{numbered_source}"
    
    return documents

def file_name_and_ast_parsed_functions_only(filename, relative_path):
    """
    Parse Python file using AST, return only standalone functions and class methods with line numbers
    """
    documents = {}
    try:
        with open(filename) as f:
            source_code = f.read()
            source_lines = source_code.splitlines()
        tree = ast.parse(source_code)
        
        # Walk through AST tree
        for node in ast.walk(tree):
            # Process standalone functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if this is a class method (by checking parent node)
                if not isinstance(node.parent, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    function_source = ast.unparse(node)
                    start_line = node.lineno
                    numbered_source = '\n'.join(f"{start_line + i} {line}" for i, line in enumerate(function_source.splitlines()))
                    
                    content = (
                        f"{relative_path}\n"
                        f"Function: {node.name}\n"
                    )
                    if docstring:
                        numbered_docstring = '\n'.join(f"{start_line + 1 + i} {line}" for i, line in enumerate(docstring.splitlines()))
                        content += f"Documentation:\n{numbered_docstring}\n"
                    content += f"\nSource Code:\n{numbered_source}"
                    
                    key = f"{relative_path}:{node.name}"
                    documents[key] = content
            
            # Process class methods
            elif isinstance(node, ast.ClassDef):
                for class_node in node.body:
                    if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_docstring = ast.get_docstring(class_node)
                        method_source = ast.unparse(class_node)
                        method_start_line = class_node.lineno
                        numbered_method_source = '\n'.join(f"{method_start_line + i} {line}" for i, line in enumerate(method_source.splitlines()))
                        
                        method_content = (
                            f"{relative_path}\n"
                            f"Method: {node.name}.{class_node.name}\n"
                        )
                        if method_docstring:
                            numbered_method_docstring = '\n'.join(f"{method_start_line + 1 + i} {line}" for i, line in enumerate(method_docstring.splitlines()))
                            method_content += f"Documentation:\n{numbered_method_docstring}\n"
                        method_content += f"\nSource Code:\n{numbered_method_source}"
                        
                        method_key = f"{relative_path}:{node.name}.{class_node.name}"
                        documents[method_key] = method_content
                
    except Exception as e:
        logger.error(f"Failed to parse file {filename} with AST: {e}")
        numbered_source = '\n'.join(f"{i+1} {line}" for i, line in enumerate(source_code.splitlines()))
        documents[relative_path] = f"{relative_path}\n{numbered_source}"
    
    return documents

DOCUMENT_ENCODING_FUNCTIONS = {
    "file_name_and_contents": file_name_and_contents,
    "file_name_and_documentation": file_name_and_documentation,
    "file_name_and_docs_jedi": file_name_and_docs_jedi,
    "file_name_and_ast_parsed": file_name_and_ast_parsed, # Parse Python file using AST, return each class and function as separate documents with line numbers
    "file_name_and_ast_parsed_functions_only": file_name_and_ast_parsed_functions_only, # Parse Python file using AST, return only standalone functions and class methods with line numbers
}

def clone_repo(repo, root_dir, token):
    """
    Clone GitHub repository to specified directory
    """
    repo_dir = Path(root_dir, f"repo__{repo.replace('/', '__')}")
    
    # If root_dir is already a repository path, use it directly
    if Path(root_dir).exists() and Path(root_dir).is_dir():
        try:
            test_repo = Repo(root_dir)
            logger.info(f"Using existing repository at {root_dir}")
            return root_dir
        except Exception as e:
            logger.warning(f"Directory {root_dir} exists but is not a valid Git repo: {e}")
    
    # If repo_dir exists and is a valid git repository, use it directly
    if repo_dir.exists() and repo_dir.is_dir():
        try:
            test_repo = Repo(repo_dir)
            logger.info(f"Using existing repository at {repo_dir}")
            return repo_dir
        except Exception as e:
            logger.warning(f"Directory {repo_dir} exists but is not a valid Git repo: {e}, will recreate it")
            shutil.rmtree(repo_dir)
    
    repo_name = repo.split('/')[-1]
    possible_paths = [
        Path(os.getcwd(), repo_name),
        Path(os.getcwd(), repo.replace('/', '__')),
        Path(os.getcwd(), repo)
    ]
    
    for current_path_repo in possible_paths:
        if current_path_repo.exists() and current_path_repo.is_dir():
            try:
                test_repo = Repo(current_path_repo)
                logger.info(f"Using existing repository at {current_path_repo}")
                
                if not repo_dir.exists():
                    if os.name != 'nt':
                        os.symlink(current_path_repo, repo_dir)
                        logger.info(f"Created symlink from {current_path_repo} to {repo_dir}")
                        return repo_dir
                    else:
                        logger.info(f"Copying repository from {current_path_repo} to {repo_dir}")
                        shutil.copytree(current_path_repo, repo_dir)
                        return repo_dir
                else:
                    return repo_dir
            except Exception as e:
                logger.warning(f"Directory {current_path_repo} exists but is not a valid Git repo or cannot be linked: {e}")
    
    if not repo_dir.exists():
        repo_url = f"https://{token}@github.com/{repo}.git"
        logger.info(f"Cloning {repo} {os.getpid()}")
        Repo.clone_from(repo_url, repo_dir)
    
    return repo_dir

def build_documents(repo_dir, commit, document_encoding_func):
    """
    Build documents at specified commit
    """
    documents = dict()
    with ContextManager(repo_dir, commit):
        filenames = list_files(repo_dir, include_tests=False)
        for relative_path in filenames:
            filename = os.path.join(repo_dir, relative_path)
            result = document_encoding_func(filename, relative_path)
            
            if isinstance(result, dict):
                documents.update(result)
            else:
                documents[relative_path] = result
                
    return documents

@contextmanager
def get_embedding_model():
    """
    Context manager for loading and using intfloat-e5-base-v2 embedding model
    """
    try:
        # Use environment variable or default model path
        model_path = os.environ.get('EMBEDDING_MODEL_PATH', 'intfloat/e5-base-v2')
        model = SentenceTransformer(model_path)
        if torch.cuda.is_available():
            model = model.to(torch.device('cuda:0'))
        yield model
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def make_index(
    repo_dir,
    root_dir,
    query,
    commit,
    document_encoding_func,
    python,
    instance_id,
):
    """
    Build vector index for documents using E5 model
    """
    index_dir = Path(root_dir, f"index__{str(instance_id)}")
    index_path = index_dir / "faiss_index.bin"
    docids_path = index_dir / "docids.json"
    documents_path = index_dir / "documents.json"
    
    if index_path.exists() and docids_path.exists() and documents_path.exists():
        with open(docids_path, 'r') as f:
            docids = json.load(f)
        with open(documents_path, 'r') as f:
            documents = json.load(f)
        return index_path, documents, docids
    
    if not index_dir.exists():
        index_dir.mkdir(parents=True, exist_ok=True)
    
    thread_prefix = f"(pid {os.getpid()}) "
    documents = build_documents(repo_dir, commit, document_encoding_func)
    
    with open(documents_path, 'w') as f:
        json.dump(documents, f)
    
    docids = list(documents.keys())
    texts = list(documents.values())
    
    with open(docids_path, 'w') as f:
        json.dump(docids, f)
    
    try:
        with get_embedding_model() as model:
            embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            faiss.write_index(index, index_path.as_posix())
            
    except Exception as e:
        logger.error(thread_prefix + f"Failed to build index for {instance_id}")
        logger.error(traceback.format_exc())
        raise e
        
    return index_path, documents, docids

def get_index_paths_worker(
    instance,
    root_dir_name,
    document_encoding_func,
    python,
    token,
):
    """
    Worker thread function to process single instance
    """
    index_info = None
    repo = instance["repo"]
    commit = instance["base_commit"]
    instance_id = instance["instance_id"]
    try:
        with git_lock:
            repo_dir = clone_repo(repo, root_dir_name, token)
        query = instance["problem_statement"]
        index_path, documents, docids = make_index(
            repo_dir=repo_dir,
            root_dir=root_dir_name,
            query=query,
            commit=commit,
            document_encoding_func=document_encoding_func,
            python=python,
            instance_id=instance_id,
        )
        index_info = (index_path, documents, docids)
    except Exception as e:
        logger.error(f"Failed to process {repo}/{commit} (instance {instance_id})")
        logger.error(traceback.format_exc())
    return instance_id, index_info

def get_index_paths(
    remaining_instances: list[dict[str, Any]],
    root_dir_name: str,
    document_encoding_func: Any,
    python: str,
    token: str,
    output_file: str,
    num_workers: int = 4,
) -> dict[str, tuple]:
    """
    Process instances in parallel using multithreading
    """
    all_index_infos = dict()
    
    # Create thread pool, reduce worker count to avoid resource contention
    num_workers = min(num_workers, 2)  # Limit maximum workers to 2
    
    # Create thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks to thread pool
        future_to_instance = {
            executor.submit(
                get_index_paths_worker,
                instance=instance,
                root_dir_name=root_dir_name,
                document_encoding_func=document_encoding_func,
                python=python,
                token=token,
            ): instance
            for instance in remaining_instances
        }
        
        # Use tqdm to show progress
        for future in tqdm(as_completed(future_to_instance), total=len(remaining_instances), desc="Indexing"):
            instance = future_to_instance[future]
            try:
                instance_id, index_info = future.result()
                if index_info is not None:
                    all_index_infos[instance_id] = index_info
            except Exception as e:
                logger.error(f"Failed to process instance {instance['instance_id']}: {str(e)}")
                logger.error(traceback.format_exc())
    
    return all_index_infos

def get_remaining_instances(instances, output_file):
    """
    Filter processed instances
    """
    instance_ids = set()
    remaining_instances = list()
    if output_file.exists():
        with FileLock(output_file.as_posix() + ".lock"):
            with open(output_file) as f:
                for line in f:
                    instance = json.loads(line)
                    instance_id = instance["instance_id"]
                    instance_ids.add(instance_id)
            logger.warning(
                f"Found {len(instance_ids)} existing instances in {output_file}. Will skip them."
            )
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return instances
    for instance in instances:
        instance_id = instance["instance_id"]
        if instance_id not in instance_ids:
            remaining_instances.append(instance)
    return remaining_instances

def search(instance, index_info):
    """
    Search relevant documents in vector index using E5 model
    """
    try:
        instance_id = instance["instance_id"]
        index_path, documents, docids = index_info
        
        index = faiss.read_index(index_path.as_posix())
        query_text = instance["problem_statement"]
        
        with get_embedding_model() as model:
            query_embedding = model.encode([query_text], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            k = 20
            scores, indices = index.search(query_embedding, k)
            
            results = {"instance_id": instance_id, "hits": []}
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx != -1:
                    docid = docids[idx]
                    score = float(scores[0][i])
                    results["hits"].append({"docid": docid, "score": score})
            
            return results
            
    except Exception as e:
        logger.error(f"Failed to process {instance_id}")
        logger.error(traceback.format_exc())
        return None

def search_indexes(remaining_instance, output_file, all_index_infos):
    """
    Search in indexes for given instances and write results to output file
    """
    for instance in tqdm(remaining_instance, desc="Retrieving"):
        instance_id = instance["instance_id"]
        if instance_id not in all_index_infos:
            continue
        index_info = all_index_infos[instance_id]
        results = search(instance, index_info)
        if results is None:
            continue
        with FileLock(output_file.as_posix() + ".lock"):
            with open(output_file, "a") as out_file:
                print(json.dumps(results), file=out_file, flush=True)

def get_missing_ids(instances, output_file):
    """
    Get missing instance IDs
    """
    with open(output_file) as f:
        written_ids = set()
        for line in f:
            instance = json.loads(line)
            instance_id = instance["instance_id"]
            written_ids.add(instance_id)
    missing_ids = set()
    for instance in instances:
        instance_id = instance["instance_id"]
        if instance_id not in written_ids:
            missing_ids.add(instance_id)
    return missing_ids

def get_root_dir(dataset_name, output_dir, document_encoding_style):
    """
    Get root directory
    """
    root_dir = Path(output_dir)
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
    root_dir_name = root_dir
    return root_dir, root_dir_name

def main(
    dataset_name_or_path,
    document_encoding_style,
    output_dir,
    shard_id,
    num_shards,
    splits,
    leave_indexes,
    num_workers: int = 4,
):
    """
    Main function
    """
    document_encoding_func = DOCUMENT_ENCODING_FUNCTIONS[document_encoding_style]
    token = os.environ.get("GITHUB_TOKEN", "git")
    dataset = load_dataset(dataset_name_or_path)
    dataset_name = dataset_name_or_path.replace("/", "__")
    
    if shard_id is not None:
        for split in splits:
            dataset[split] = dataset[split].shard(num_shards, shard_id)

    instances = list()
    if set(splits) - set(dataset.keys()) != set():
        raise ValueError(f"Unknown splits {set(splits) - set(dataset.keys())}")
    for split in splits:
        instances += list(dataset[split])
    python = subprocess.run("which python", shell=True, capture_output=True)
    python = python.stdout.decode("utf-8").strip()

    output_file = Path(
        output_dir, f"{dataset_name}_{document_encoding_style}.e5_retrieval.jsonl"
    )

    remaining_instances = get_remaining_instances(instances, output_file)

    root_dir, root_dir_name = get_root_dir(
        dataset_name, output_dir, document_encoding_style
    )
    
    all_index_infos = get_index_paths(
        remaining_instances,
        root_dir_name,
        document_encoding_func,
        python,
        token,
        output_file,
        num_workers=num_workers,
    )

    logger.info(f"Finished indexing {len(all_index_infos)} instances")

    search_indexes(remaining_instances, output_file, all_index_infos)

    missing_ids = get_missing_ids(instances, output_file)
    logger.warning(f"Missing indexes for {len(missing_ids)} instances.")
    logger.info(f"Saved retrieval results to {output_file}")
    del_dirs = list(root_dir.glob("repo__*"))
    logger.info(f"Cleaning up {root_dir}")
    if leave_indexes:
        index_dirs = list(root_dir.glob("index__*"))
        del_dirs += index_dirs
    for dirname in del_dirs:
        shutil.rmtree(dirname, ignore_errors=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="princeton-nlp/SWE-bench",
        help="Dataset to use for test set from HuggingFace Datasets or path to a save_to_disk directory.",
    )
    parser.add_argument(
        "--document_encoding_style",
        choices=DOCUMENT_ENCODING_FUNCTIONS.keys(),
        default="file_name_and_contents",
    )
    parser.add_argument("--output_dir", default="./retreival_results")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--shard_id", type=int)
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--leave_indexes", type=string_to_bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for parallel processing")
    args = parser.parse_args()
    main(**vars(args)) 