import json
import os

def check_and_fix_json(file_path):
    print(f"Checking file: {file_path}")
    
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    # 1. 尝试直接加载，看看是否真的有错
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print("File is valid JSON. No repair needed.")
        return
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error detected: {e}")
        print("Attempting to locate and fix error...")

    # 2. 如果是 JSON 数组格式，尝试逐个解析来定位错误
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 简单的尝试：有时候是因为生成文件时结尾截断了
    if not content.strip().endswith(']'):
        print("Warning: File does not end with ']'. Trying to append it.")
        # 尝试简单修复
        fixed_content = content.strip()
        if fixed_content.endswith(','):
            fixed_content = fixed_content[:-1]
        fixed_content += ']'
        
        try:
            json.loads(fixed_content)
            print("Fixed by appending ']'. Saving...")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return
        except json.JSONDecodeError:
            print("Appending ']' did not fix it. Proceeding to deeper check.")

    # 3. 逐行检查 (针对 pretty-printed JSON 或 jsonl)
    # 如果是标准 JSON array，这可能比较难，我们尝试使用 eval (非常不安全但对调试有效) 或者第三方库
    # 这里我们尝试一个更稳妥的方法：将文件内容视为字符串处理，查找未转义的引号
    
    # 通常 Unmatched '"' 是因为字符串内部包含了未转义的引号，例如 "msg": "Hello "World""
    # 这是一个非常难自动完美修复的问题。
    
    print("Detailed analysis: The error 'Unmatched \"' usually means a string contains unescaped quotes.")
    print("Please manually check the file content near the error position reported above.")
    
    # 为了帮助定位，我们尝试找到出错的行
    lines = content.splitlines()
    for i, line in enumerate(lines):
        # 简单的启发式检查：一行中引号数量应该是偶数（除非有转义）
        # 注意：这只是一个非常粗略的检查
        quote_count = 0
        escape = False
        for char in line:
            if char == '\\':
                escape = not escape
            elif char == '"' and not escape:
                quote_count += 1
                escape = False
            else:
                escape = False
        
        if quote_count % 2 != 0:
             print(f"Suspicious line {i+1}: Odd number of quotes. Content: {line.strip()[:100]}...")

if __name__ == "__main__":
    # 请修改为实际文件路径
    file_path = r"c:\Users\86135\Desktop\ShieldLM\train_code\sft_data\combine_rules_qwen_format_label_first\train_final.json"
    check_and_fix_json(file_path)
