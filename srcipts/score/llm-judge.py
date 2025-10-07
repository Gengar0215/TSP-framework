#!/usr/bin/env python3
import json
import time
import sys
import re
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from typing import Optional, Tuple
from multiprocessing import Pool, Manager
from dataclasses import dataclass, field
from transformers import HfArgumentParser
"""
Concurrent API calling with config.yaml configuration file
"""


# API configuration information
model_info = {
    "gpt-4o": {
        "api_name": "",
        "api_key": "",
        "base_url": ""
    }
}
@dataclass
class ConcurrentConfig:
    # API configuration
    api_model: str = field(
        default="qwen-max",
        metadata={"help": "API model to use (gpt-4o, qwen-max)"}
    )
    n_processes: int = field(
        default=5,
        metadata={"help": "Number of parallel processes"}
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of API retries"}
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum number of tokens in API response"}
    )
    
    # File paths
    input_file: str = field(
        default="result.jsonl",
        metadata={"help": "Path to input jsonl file"}
    )
    output_suffix: str = field(
        default="_processed",
        metadata={"help": "Suffix for output file name"}
    )
    error_file_suffix: str = field(
        default="_errors",
        metadata={"help": "Suffix for error file name"}
    )
    
    # Limit processing count (for testing)
    limit_records: Optional[int] = field(
        default=None,
        metadata={"help": "Limit number of records to process (None for all)"}
    )

def get_api_client(api_model: str) -> OpenAI:
    """Create API client"""
    try:
        if api_model not in model_info:
            raise ValueError(f"Unsupported model: {api_model}")
            
        if api_model == "gpt-4o":
            client = OpenAI(
                api_key=model_info[api_model]["api_key"],
                base_url=model_info[api_model]["base_url"],
                default_query={"api-version": "2024-06-01"},
                timeout=30.0
            )
        else:
            client = OpenAI(
                api_key=model_info[api_model]["api_key"],
                base_url=model_info[api_model]["base_url"],
                timeout=30.0
            )
        return client
    except Exception as e:
        print(f"Error creating API client: {str(e)}")
        raise

def run_api(client: OpenAI, api_model: str, prompt: str, max_tokens: int = 4096, max_retries: int = 3) -> Tuple[Optional[str], int, int, int]:
    """Run API call with retry mechanism"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_info[api_model]["api_name"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=30
            )
            
            clean_response = response.choices[0].message.content
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
            total_tokens = response.usage.total_tokens
            return clean_response, completion_tokens, prompt_tokens, total_tokens
            
        except Exception as e:
            retry_count += 1
            
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
            else:
                return None, 0, 0, 0

def extract_reward_score(api_result: str) -> Optional[float]:
    """Extract reward score from API response"""
    try:
        # Use regex to extract content from <reward></reward> tags
        match = re.search(r'<reward>(.*?)</reward>', api_result, re.DOTALL | re.IGNORECASE)
        if match:
            reward_text = match.group(1).strip()
            # Try to extract number (supports integers and decimals)
            score_match = re.search(r'(\d+(?:\.\d+)?)', reward_text)
            if score_match:
                return float(score_match.group(1))
        return None
    except Exception:
        return None

def extract_patch_content(response_text: str) -> str:
    """Extract content between <patch></patch> tags from response text"""
    try:
        # Use regex to extract content from <patch></patch> tags
        match = re.search(r'<patch>(.*?)</patch>', response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            # If no patch tags found, return original text
            return response_text
    except Exception:
        # If extraction fails, return original text
        return response_text

def process_single_record_with_api(idx, data, config, output_file, error_file, lock):
    """Process single record function (for multiprocessing)"""
    try:
        # Create client in subprocess
        client = get_api_client(config.api_model)
        
        # Extract required fields
        issue = data["problem_statement"]
        
        # Handle golden_patch - may be in different locations
        if "patch" in data:
            golden_patch = data["patch"]
        elif "row_data" in data and "patch" in data["row_data"]:
            golden_patch = data["row_data"]["patch"]
        else:
            # Output debug information
            available_keys = list(data.keys())
            error_msg = f"Patch field not found. Available fields: {available_keys}"
            if "row_data" in data:
                row_data_keys = list(data["row_data"].keys())
                error_msg += f", row_data fields: {row_data_keys}"
            raise KeyError(error_msg)
        
        # Handle model_output - check different possible fields
        if "second_round_response" in data:
            model_output = extract_patch_content(data["second_round_response"])
        elif "first_round_response" in data:
            model_output = extract_patch_content(data["first_round_response"])
        else:
            # Output debug information
            available_keys = list(data.keys())
            error_msg = f"Response field not found. Available fields: {available_keys}"
            raise KeyError(error_msg)
        
        original_reward = data.get("reward", None)  # Extract original reward, may not exist
        
        # Check if extracted content is valid
        if not model_output or model_output.strip() == "":
            print(f"Warning: model_output is empty for record {idx}")
            model_output = "Unable to extract valid patch content"
        
        # Build prompt
        request_prompt = """
As an SWE-bench task evaluation expert, you need to evaluate the model output based on the following materials:  
1. `issue`: Bug description reported by the user  
2. `golden patch`: Correct code repair solution  
3. `model output`: Response generated by the model (including thinking process, search query, recall information, and model patch)  
 
Scoring Dimensions and Weights:  
1. Context Relevance (20%)  
   - Check whether the search query accurately recalls the code context related to the issue  
2. Position Correctness (40%)  
   - Determine whether the model patch includes the modification content in the golden patch  
3. Core Fix Correctness (40%)  
   - Check whether the model patch solves the core problem of the issue  
Finally, put the score in <reward></reward>.

Issue: {issue}
Golden Patch: {golden_patch}
Model Output: {model_output}
"""
        
        prompt = request_prompt.format(issue=issue, golden_patch=golden_patch, model_output=model_output)
        
        # Call API
        result, completion_tokens, prompt_tokens, total_tokens = run_api(
            client, config.api_model, prompt, config.max_tokens, config.max_retries
        )
        
        if result is None:
            # API call failed, save to error file
            with lock:
                with open(error_file, 'a', encoding='utf-8') as f_err:
                    error_data = data.copy()
                    error_data['error_reason'] = "API call failed"
                    error_data['index'] = idx
                    f_err.write(json.dumps(error_data, ensure_ascii=False) + '\n')
            return "error", None
        
        # Extract reward score from API response
        api_reward_score = extract_reward_score(result)
        
        # Create output record
        output_record = data.copy()
        output_record['api_result'] = result
        output_record['api_reward_score'] = api_reward_score
        if original_reward is not None:
            output_record['original_reward'] = original_reward
        output_record['index'] = idx
        output_record['usage'] = {
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens
        }
        
        # Write successful result
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        
        return "success", api_reward_score
        
    except KeyError as e:
        # Data format error
        with lock:
            with open(error_file, 'a', encoding='utf-8') as f_err:
                error_data = data.copy()
                error_data['error_reason'] = f"Data format error: {str(e)}"
                error_data['index'] = idx
                f_err.write(json.dumps(error_data, ensure_ascii=False) + '\n')
        print(f"Record {idx} data format error: {str(e)}")
        return "error", None
        
    except Exception as e:
        # Other errors
        with lock:
            with open(error_file, 'a', encoding='utf-8') as f_err:
                error_data = data.copy()
                error_data['error_reason'] = f"Processing exception: {str(e)}"
                error_data['index'] = idx
                f_err.write(json.dumps(error_data, ensure_ascii=False) + '\n')
        print(f"Record {idx} processing exception: {str(e)}")
        return "error", None

def process_with_multiprocessing(config, data_list, output_file, error_file):
    """Process using multiprocessing concurrency"""
    
    # Create empty output and error files
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(error_file, "w", encoding="utf-8") as f:
        pass
    
    # Create multiprocessing manager lock
    manager = Manager()
    lock = manager.Lock()
    
    # Prepare arguments
    args_list = [(idx, data, config, output_file, error_file, lock) 
                 for idx, data in enumerate(data_list)]
    
    # Use process pool for parallel processing
    with Pool(processes=config.n_processes) as pool:
        # Submit all tasks
        pool_results = []
        for args in tqdm(args_list, desc="Submitting tasks"):
            res = pool.apply_async(process_single_record_with_api, args)
            pool_results.append(res)
        
        # Wait for all tasks to complete and collect results
        results = []
        reward_scores = []
        for res in tqdm(pool_results, desc=f"Processing ({config.api_model})"):
            try:
                status, reward_score = res.get()
                results.append(status)
                if reward_score is not None:
                    reward_scores.append(reward_score)
            except Exception as e:
                print(f"Process execution error: {str(e)}")
                results.append("error")
    
    # Collect statistics
    success_count = sum(1 for r in results if r == "success")
    error_count = sum(1 for r in results if r == "error")
    
    return success_count, error_count, reward_scores

def sort_output_file(output_file):
    """Sort output file by original index"""
    try:
        with open(output_file, 'r', encoding='utf-8') as fin:
            results = []
            for line in fin:
                try:
                    data = json.loads(line.strip())
                    results.append(data)
                except:
                    continue
        
        # Sort by index
        results.sort(key=lambda x: x.get('index', 0))
        
        # Rewrite file
        with open(output_file, 'w', encoding='utf-8') as fout:
            for data in results:
                # Remove temporary index field
                if 'index' in data:
                    del data['index']
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"✅ Output file sorted by original order")
        
    except Exception as e:
        print(f"⚠️ Error sorting output file: {str(e)}")

def main():
    # Parse arguments
    parser = HfArgumentParser((ConcurrentConfig,))
    
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
        print(f"Loading from config file: {sys.argv[1]}")
        config, = parser.parse_yaml_file(sys.argv[1])
    else:
        config, = parser.parse_args_into_dataclasses()
    
    print(f"🚀 Starting concurrent processing (model: {config.api_model}, processes: {config.n_processes})")
    
    # Set file paths
    input_path = Path(config.input_file)
    output_file = input_path.parent / f"{input_path.stem}{config.output_suffix}.jsonl"
    error_file = input_path.parent / f"{input_path.stem}{config.error_file_suffix}.jsonl"
    
    print(f"📁 Input file: {config.input_file}")
    print(f"📁 Output file: {output_file}")
    print(f"📁 Error file: {error_file}")
    
    # Read input file
    print("📖 Reading input file...")
    data_list = []
    original_rewards = []
    with open(config.input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                data_list.append(data)
                # Collect original reward scores
                if "reward" in data:
                    try:
                        original_rewards.append(float(data["reward"]))
                    except (ValueError, TypeError):
                        pass
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON on line {line_num}")
                continue
    
    print(f"✅ Successfully read {len(data_list)} records")
    if original_rewards:
        print(f"📊 Original data contains {len(original_rewards)} reward scores")
        print(f"📊 Original reward average: {sum(original_rewards)/len(original_rewards):.3f}")
    
    # Limit processing count (for testing)
    if config.limit_records and config.limit_records < len(data_list):
        data_list = data_list[:config.limit_records]
        print(f"🔧 Limited processing count: {len(data_list)} records")
    
    # Execute concurrent processing
    try:
        success_count, error_count, api_reward_scores = process_with_multiprocessing(
            config, data_list, output_file, error_file
        )
        
        # Sort output file
        if success_count > 0:
            sort_output_file(output_file)
        
        # Calculate API reward score statistics
        valid_api_scores = [score for score in api_reward_scores if score is not None]
        
        # Print result statistics
        print(f"\n🎉 Processing completed!")
        print(f"📊 Processing statistics:")
        print(f"  - Total records: {len(data_list)}")
        print(f"  - Successfully processed: {success_count}")
        print(f"  - Failed records: {error_count}")
        print(f"  - Success rate: {success_count/len(data_list)*100:.2f}%")
        
        # Print score statistics
        if valid_api_scores:
            avg_score = sum(valid_api_scores) / len(valid_api_scores)
            min_score = min(valid_api_scores)
            max_score = max(valid_api_scores)
            print(f"🏆 API scoring statistics:")
            print(f"  - Valid score count: {len(valid_api_scores)}")
            print(f"  - Average score: {avg_score:.3f}")
            print(f"  - Minimum score: {min_score:.3f}")
            print(f"  - Maximum score: {max_score:.3f}")
        else:
            print(f"⚠️ No valid API scores extracted")
            
        print(f"📁 Files saved:")
        print(f"  - Successful results: {output_file}")
        if error_count > 0:
            print(f"  - Error data: {error_file}")
        else:
            # Delete empty error file
            try:
                error_file.unlink()
                print(f"✨ No error data, deleted empty error file")
            except:
                pass
        
    except Exception as e:
        print(f"❌ Program execution error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 