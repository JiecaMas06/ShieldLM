import json
import os

input_file = r"c:\Users\86135\Desktop\ShieldLM\train_code\sft_data\combine_rules_qwen_format_label_first\train_final.json"
output_file = r"c:\Users\86135\Desktop\ShieldLM\train_code\sft_data\combine_rules_qwen_format_label_first\train_final.jsonl"

def convert_to_jsonl(in_path, out_path):
    print(f"Reading {in_path}...")
    if not os.path.exists(in_path):
        print(f"Error: File {in_path} not found.")
        return

    try:
        with open(in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} items. Writing to {out_path}...")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print("Conversion complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    convert_to_jsonl(input_file, output_file)
