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
Token ID Finder for Baichuan2 and Qwen tokenizers.

This script helps find the correct token IDs for specific keywords
used in safety evaluation (safe/unsafe/controversial).
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


def find_token_ids(tokenizer, keywords: list):
    """
    Find token IDs for given keywords.
    
    Args:
        tokenizer: Loaded tokenizer instance
        keywords: List of keywords to find
        
    Returns:
        Dictionary mapping keywords to their token information
    """
    results = {}
    
    for keyword in keywords:
        # Encode the keyword
        token_ids = tokenizer.encode(keyword)
        
        # Decode back to verify
        decoded = tokenizer.decode(token_ids)
        
        # Check if it's a single token
        is_single_token = len(token_ids) == 1
        
        results[keyword] = {
            "token_ids": token_ids,
            "decoded": decoded,
            "is_single_token": is_single_token,
            "first_token_id": token_ids[0] if token_ids else None
        }
    
    return results


def search_vocab(tokenizer, search_term: str, max_results: int = 20):
    """
    Search vocabulary for tokens containing the search term.
    
    Args:
        tokenizer: Loaded tokenizer instance
        search_term: Term to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of (token, token_id) tuples
    """
    try:
        vocab = tokenizer.get_vocab()
        matches = []
        
        for token, idx in vocab.items():
            token_str = str(token)
            if search_term in token_str:
                matches.append((token_str, idx))
        
        # Sort by token ID
        matches.sort(key=lambda x: x[1])
        return matches[:max_results]
    except Exception as e:
        print(f"Warning: Could not search vocab: {e}")
        return []


def verify_token_id(tokenizer, token_id: int):
    """
    Verify what a specific token ID decodes to.
    
    Args:
        tokenizer: Loaded tokenizer instance
        token_id: Token ID to verify
        
    Returns:
        Decoded string
    """
    try:
        decoded = tokenizer.decode([token_id])
        return decoded
    except Exception as e:
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Find token IDs for specific keywords in Baichuan2/Qwen tokenizers"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["baichuan2-13b", "qwen-14b"],
        help="Model type"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer file"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=None,
        help="Keywords to find token IDs for"
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search term to find in vocabulary"
    )
    parser.add_argument(
        "--verify",
        type=int,
        nargs="+",
        default=None,
        help="Token IDs to verify (decode)"
    )
    parser.add_argument(
        "--safety_keywords",
        action="store_true",
        help="Find token IDs for safety evaluation keywords"
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading {args.model_type} tokenizer from {args.tokenizer}...")
    if args.model_type == "baichuan2-13b":
        tokenizer = load_baichuan2_tokenizer(args.tokenizer)
    else:
        tokenizer = load_qwen_tokenizer(args.tokenizer)
    
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    print("=" * 60)
    
    # Default safety keywords
    if args.safety_keywords:
        keywords = [
            # Chinese
            "安全", "不安全", "有争议",
            # English
            "safe", "unsafe", "controversial",
            # Additional variants
            "Safe", "Unsafe", "Controversial",
            "安", "全", "不", "争", "议"
        ]
        args.keywords = keywords
    
    # Find token IDs for keywords
    if args.keywords:
        print("\n📝 Token ID Lookup Results:")
        print("-" * 60)
        
        results = find_token_ids(tokenizer, args.keywords)
        
        for keyword, info in results.items():
            status = "✅ Single token" if info["is_single_token"] else "⚠️  Multiple tokens"
            print(f"\nKeyword: '{keyword}'")
            print(f"  Status: {status}")
            print(f"  Token IDs: {info['token_ids']}")
            print(f"  Decoded: '{info['decoded']}'")
            if info["is_single_token"]:
                print(f"  >>> Use token ID: {info['first_token_id']}")
            else:
                print(f"  >>> First token ID: {info['first_token_id']} (may need verification)")
    
    # Search vocabulary
    if args.search:
        print(f"\n🔍 Vocabulary Search for '{args.search}':")
        print("-" * 60)
        
        matches = search_vocab(tokenizer, args.search)
        
        if matches:
            for token, idx in matches:
                print(f"  {idx:8d}: {repr(token)}")
        else:
            print("  No matches found.")
    
    # Verify token IDs
    if args.verify:
        print("\n✓ Token ID Verification:")
        print("-" * 60)
        
        for token_id in args.verify:
            decoded = verify_token_id(tokenizer, token_id)
            print(f"  Token ID {token_id:8d} -> '{decoded}'")
    
    # Print summary for safety tokens
    if args.safety_keywords:
        print("\n" + "=" * 60)
        print("📋 SAFETY_TOKENS Configuration Template:")
        print("=" * 60)
        
        results = find_token_ids(tokenizer, ["安全", "不安全", "有争议", "safe", "unsafe", "controversial"])
        
        model_key = args.model_type
        print(f'''
"{model_key}": {{
    "zh": {{
        "token_place": 3,  # Adjust based on your prompt template
        "safe": {results["安全"]["first_token_id"]},  # '{results["安全"]["decoded"]}' - {"Single" if results["安全"]["is_single_token"] else "Multiple"} token
        "unsafe": {results["不安全"]["first_token_id"]},  # '{results["不安全"]["decoded"]}' - {"Single" if results["不安全"]["is_single_token"] else "Multiple"} token
        "controversial": {results["有争议"]["first_token_id"]}  # '{results["有争议"]["decoded"]}' - {"Single" if results["有争议"]["is_single_token"] else "Multiple"} token
    }},
    "en": {{
        "token_place": 3,
        "safe": {results["safe"]["first_token_id"]},  # '{results["safe"]["decoded"]}' - {"Single" if results["safe"]["is_single_token"] else "Multiple"} token
        "unsafe": {results["unsafe"]["first_token_id"]},  # '{results["unsafe"]["decoded"]}' - {"Single" if results["unsafe"]["is_single_token"] else "Multiple"} token
        "controversial": {results["controversial"]["first_token_id"]}  # '{results["controversial"]["decoded"]}' - {"Single" if results["controversial"]["is_single_token"] else "Multiple"} token
    }}
}}
''')
        print("⚠️  Note: If a keyword is split into multiple tokens, you may need to")
        print("   adjust the token_place or use a different token representation.")


if __name__ == "__main__":
    main()
