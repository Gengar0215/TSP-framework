import json
import re
from typing import List, Dict, Any


def calculate_search_reward(data_item: Dict[str, Any]) -> float:
    """
    Calculate the search_reward for a single data item
    
    Args:
        data_item: Data item containing row_data
        
    Returns:
        float: search_reward score (0.0 to 1.0)
    """
    # Read relevant fields from row_data
    row_data = data_item.get('row_data', {})
    search_reward_classorfunc = row_data.get('search_reward_classorfunc', [])
    information = row_data.get('information', '')
    
    if not search_reward_classorfunc:
        return 0.0
    
    total_count = len(search_reward_classorfunc)
    matched_count = 0
    
    for func_or_class in search_reward_classorfunc:
        # Clean function/class definition, remove extra spaces and formatting
        cleaned_target = func_or_class.strip()
        
        # Check if it exists in information
        if cleaned_target in information:
            matched_count += 1
        else:
            # Try more flexible matching to handle possible format differences
            # Extract function name or class name for matching
            found = False
            
            if cleaned_target.startswith('def '):
                # Extract function name
                func_name_match = re.search(r'def\s+(\w+)', cleaned_target)
                if func_name_match:
                    func_name = func_name_match.group(1)
                    # Check if function name appears in information as def form
                    if re.search(rf'def\s+{re.escape(func_name)}\s*\(', information):
                        matched_count += 1
                        found = True
                        
            elif cleaned_target.startswith('class '):
                # Extract class name
                class_name_match = re.search(r'class\s+(\w+)', cleaned_target)
                if class_name_match:
                    class_name = class_name_match.group(1)
                    # Check if class name appears in information as class form
                    if re.search(rf'class\s+{re.escape(class_name)}\s*[:(]', information):
                        matched_count += 1
                        found = True
    
    return matched_count / total_count if total_count > 0 else 0.0


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        data_list.append(data)
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error at line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error occurred while reading file: {e}")
        return []
    
    return data_list


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python3 cal_search_reward.py <jsonl_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"Starting to process file: {file_path}")
    
    data_list = load_jsonl_file(file_path)
    if not data_list:
        print("No valid data found")
        sys.exit(1)
    
    print(f"Total loaded {len(data_list)} data items")
    
    # Calculate reward for each data item
    total_score = 0.0
    perfect_count = 0
    zero_count = 0
    individual_scores = []
    
    for i, data_item in enumerate(data_list):
        score = calculate_search_reward(data_item)
        total_score += score
        individual_scores.append(score)
        
        if score == 1.0:
            perfect_count += 1
        elif score == 0.0:
            zero_count += 1
        
        # Show progress every 50 data items
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(data_list)} data items...")
    
    # Calculate average score
    average_score = total_score / len(data_list)
    partial_count = len(data_list) - perfect_count - zero_count
    
    print(f"\n=== Final Statistics ===")
    print(f"Total data items: {len(data_list)}")
    print(f"Average score: {average_score:.4f}")
    print(f"Perfect matches (1.0): {perfect_count} items ({perfect_count/len(data_list)*100:.1f}%)")
    print(f"Partial matches (0-1): {partial_count} items ({partial_count/len(data_list)*100:.1f}%)")
    print(f"No matches (0.0): {zero_count} items ({zero_count/len(data_list)*100:.1f}%)")
    
    # Save results to file
    result = {
        'total_items': len(data_list),
        'average_score': average_score,
        'individual_scores': individual_scores,
        'perfect_matches': perfect_count,
        'partial_matches': partial_count,
        'zero_matches': zero_count
    }
    
    with open('search_reward_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: search_reward_results.json")
    
    # Show some examples
    print(f"\n=== Scores for first 5 data items ===")
    for i in range(min(5, len(individual_scores))):
        print(f"Data item {i}: {individual_scores[i]:.4f}")


if __name__ == "__main__":
    main() 