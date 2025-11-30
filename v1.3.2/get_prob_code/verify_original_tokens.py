# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Verify original token IDs from get_probability.py in MindFormers tokenizers.

This script decodes the token IDs used in the PyTorch/Transformers implementation
to check if they correspond to the same tokens in MindFormers tokenizers.
"""

import os
import sys
import argparse

# Add research directory to path for model imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESEARCH_DIR = os.path.join(PROJECT_ROOT, 'research')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RESEARCH_DIR)

# Original token IDs from get_probability.py (PyTorch/Transformers version)
ORIGINAL_TOKENS = {
    "qwen": {
        "zh": {
            "token_place": 3,
            "safe": 41479,
            "unsafe": 86009,
            "controversial": 220
        },
        "en": {
            "token_place": 3,
            "safe": 6092,
            "unsafe": 19860,
            "controversial": 20129
        }
    },
    "baichuan": {
        "zh": {
            "token_place": 3,
            "safe": 92311,
            "unsafe": 100093,
            "controversial": 100047
        },
        "en": {
            "token_place": 3,
            "safe": 6336,
            "unsafe": 53297,
            "controversial": 20290
        }
    },
    "internlm": {
        "zh": {
            "token_place": 4,
            "safe": 68419,
            "unsafe": 60358,
            "controversial": 60360
        },
        "en": {
            "token_place": 3,
            "safe": 6245,
            "unsafe": 20036,
            "controversial": 20304
        }
    },
    "chatglm": {
        "zh": {
            "token_place": 3,
            "safe": 30910,
            "unsafe": 34121,
            "controversial": 35284
        },
        "en": {
            "token_place": 5,
            "safe": 3544,
            "unsafe": 27233,
            "controversial": 13204
        }
    }
}


def load_baichuan2_tokenizer(tokenizer_path: str):
    """Load Baichuan2 tokenizer."""
    baichuan_path = os.path.join(RESEARCH_DIR, 'baichuan2')
    sys.path.insert(0, baichuan_path)
    from baichuan2_tokenizer import Baichuan2Tokenizer
    return Baichuan2Tokenizer(vocab_file=tokenizer_path)


def load_qwen_tokenizer(tokenizer_path: str):
    """Load Qwen tokenizer."""
    qwen_path = os.path.join(RESEARCH_DIR, 'qwen')
    sys.path.insert(0, qwen_path)
    from qwen_tokenizer import QwenTokenizer
    return QwenTokenizer(vocab_file=tokenizer_path)


def decode_token_id(tokenizer, token_id: int) -> str:
    """Safely decode a token ID."""
    try:
        decoded = tokenizer.decode([token_id])
        return decoded
    except Exception as e:
        return f"<ERROR: {e}>"


def main():
    parser = argparse.ArgumentParser(
        description="Verify original token IDs from get_probability.py"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["baichuan2-13b", "qwen-14b"],
        help="MindFormers model type"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer file"
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading {args.model_type} tokenizer from {args.tokenizer}...")
    if args.model_type == "baichuan2-13b":
        tokenizer = load_baichuan2_tokenizer(args.tokenizer)
        original_key = "baichuan"
    else:
        tokenizer = load_qwen_tokenizer(args.tokenizer)
        original_key = "qwen"
    
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    print("=" * 70)
    
    # Get original tokens for this model
    original_config = ORIGINAL_TOKENS.get(original_key, {})
    
    print(f"\n📋 Original Token IDs from get_probability.py ({original_key}):")
    print("=" * 70)
    
    for lang in ["zh", "en"]:
        lang_config = original_config.get(lang, {})
        lang_name = "Chinese" if lang == "zh" else "English"
        
        print(f"\n🌐 {lang_name} ({lang}):")
        print("-" * 50)
        print(f"  token_place: {lang_config.get('token_place', 'N/A')}")
        print()
        
        for token_type in ["safe", "unsafe", "controversial"]:
            token_id = lang_config.get(token_type, 0)
            decoded = decode_token_id(tokenizer, token_id)
            
            # Check if token ID is within vocab range
            in_range = "✅" if token_id < tokenizer.vocab_size else "❌ OUT OF RANGE"
            
            print(f"  {token_type:15s}: ID={token_id:6d} -> '{decoded}' {in_range}")
    
    # Also show what the expected keywords encode to
    print("\n" + "=" * 70)
    print("📝 Expected keywords encoded with MindFormers tokenizer:")
    print("=" * 70)
    
    keywords = {
        "zh": ["安全", "不安全", "有争议"],
        "en": ["safe", "unsafe", "controversial"]
    }
    
    for lang, kw_list in keywords.items():
        lang_name = "Chinese" if lang == "zh" else "English"
        print(f"\n🌐 {lang_name} ({lang}):")
        print("-" * 50)
        
        for kw in kw_list:
            token_ids = tokenizer.encode(kw)
            decoded = tokenizer.decode(token_ids)
            is_single = "✅ Single" if len(token_ids) == 1 else f"⚠️  {len(token_ids)} tokens"
            print(f"  '{kw}' -> {token_ids} -> '{decoded}' ({is_single})")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("📊 Comparison Summary:")
    print("=" * 70)
    print("""
If the original token IDs decode to the expected keywords (安全/safe, etc.),
then the original IDs from get_probability.py can be used directly.

If they decode to different text or show errors, you need to use the
token IDs found by encoding the keywords with your MindFormers tokenizer.

Use find_token_ids.py --safety_keywords to get the correct IDs for your tokenizer.
""")


if __name__ == "__main__":
    main()
