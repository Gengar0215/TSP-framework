import os
import logging
import json
from pathlib import Path
from git import Repo
from argparse import ArgumentParser
from tqdm.auto import tqdm
import pandas as pd

"""
Script to download GitHub repositories from SWE-bench dataset
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def clone_repo(repo, root_dir, token):
    """
    Clone a GitHub repository to the specified directory.
    
    Args:
        repo (str): GitHub repository to clone, format: "username/repo_name"
        root_dir (str): Root directory to clone repositories into
        token (str): GitHub personal access token for authentication
    
    Returns:
        Path: Path to the cloned repository directory
    """
    repo_dir = Path(root_dir, f"repo__{repo.replace('/', '__')}")

    if not repo_dir.exists():
        repo_url = f"https://{token}@github.com/{repo}.git"
        logger.info(f"Cloning {repo}, process ID: {os.getpid()}")
        Repo.clone_from(repo_url, repo_dir)
        logger.info(f"Successfully cloned {repo} to {repo_dir}")
    else:
        logger.info(f"Repository {repo} already exists at {repo_dir}, skipping clone")
    
    return repo_dir


def main():
    parser = ArgumentParser(description="Download SWE-bench repositories")
    parser.add_argument(
        "--parquet_file",
        type=str,
        required=True,
        help="Path to the parquet file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./",
        help="Directory to save downloaded repositories"
    )
    parser.add_argument(
        "--github_token", 
        type=str, 
        default=None,
        help="GitHub personal access token, if not provided will use GITHUB_TOKEN environment variable"
    )
    parser.add_argument(
        "--shard_id", 
        type=int, 
        default=None,
        help="ID of the data shard to process"
    )
    parser.add_argument(
        "--num_shards", 
        type=int, 
        default=1,
        help="Total number of data shards"
    )
    
    args = parser.parse_args()
    
    # Get GitHub token
    token = args.github_token or os.environ.get("GITHUB_TOKEN", "git")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read parquet file
    try:
        df = pd.read_parquet(args.parquet_file)
        logger.info(f"Successfully read parquet file: {args.parquet_file}")
        logger.info(f"Dataset size: {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to read parquet file: {e}")
        return
    
    # Shard the data
    if args.shard_id is not None:
        shard_size = len(df) // args.num_shards
        start_idx = args.shard_id * shard_size
        if args.shard_id == args.num_shards - 1:
            # Last shard includes all remaining rows
            end_idx = len(df)
        else:
            end_idx = start_idx + shard_size
        
        df = df.iloc[start_idx:end_idx]
        logger.info(f"Data sharded: {args.shard_id}/{args.num_shards}, processing {len(df)} rows")
    
    # Convert DataFrame to list of dictionaries
    instances = df.to_dict('records')
    
    logger.info(f"Total instances to process: {len(instances)}")
    
    # Track successful and failed repositories
    successful_repos = []
    failed_repos = []
    
    # Track cloned repositories to avoid duplicates
    cloned_repos = set()
    
    # Download repositories
    for instance in tqdm(instances, desc="Downloading repositories"):
        try:
            repo = instance["repo"]
            
            # Avoid cloning the same repository multiple times
            if repo in cloned_repos:
                continue
                
            repo_dir = clone_repo(repo, output_dir, token)
            successful_repos.append({"repo": repo, "dir": str(repo_dir)})
            cloned_repos.add(repo)
            
        except Exception as e:
            logger.error(f"Failed to clone repository {repo}: {e}")
            failed_repos.append({"repo": repo, "error": str(e)})
    
    # Save cloning results
    results = {
        "successful": successful_repos,
        "failed": failed_repos
    }
    
    results_file = output_dir / "clone_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Cloning completed. Successful: {len(successful_repos)}, Failed: {len(failed_repos)}")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()