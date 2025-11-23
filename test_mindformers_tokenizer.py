import argparse
import os
import sys

import mindspore as ms
import numpy as np
from mindformers import AutoModelForCausalLM
from transformers import AutoTokenizer as HFAutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="在 Ascend 上测试：用 transformers 构建 tokenizer + MindFormers 模型是否能协同工作")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="本地 HuggingFace tokenizer 目录（需包含 tokenizer.json / vocab.json 等）",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="MindFormers 模型或 checkpoint 目录（AutoModelForCausalLM 可识别）",
    )
    parser.add_argument(
        "--sample_text",
        type=str,
        default="Hello, MindFormers!",
        help="要编码并生成续写的示例文本",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=int(os.getenv("DEVICE_ID", 0)),
        help="Ascend 设备 ID，默认读取环境变量 DEVICE_ID，如未设置则为 0",
    )
    parser.add_argument(
        "--use_fast",
        action="store_true",
        help="若指定，则强制使用 transformers fast tokenizer",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="传递给 transformers.AutoTokenizer.from_pretrained 的 trust_remote_code 标志",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="MindFormers 生成阶段要追加的 token 数",
    )
    return parser.parse_args()


def _ensure_tokenizer_assets(path: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"未找到 tokenizer 目录: {path}")
    expected = [
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    ]
    if not any(os.path.exists(os.path.join(path, name)) for name in expected):
        raise RuntimeError(
            "目录中未发现标准 tokenizer 资产 (tokenizer.json / vocab.json / merges.txt / tokenizer.model 等)，"
            "请确认传入的是 HuggingFace 模型文件夹。"
        )


def _build_hf_tokenizer(args):
    tokenizer = HFAutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        use_fast=args.use_fast,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise RuntimeError("tokenizer 未设置 pad_token，MindFormers 生成需要 pad_token_id")
    return tokenizer


def _build_ms_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.set_train(False)
    return model


def main():
    args = parse_args()
    _ensure_tokenizer_assets(args.tokenizer_dir)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)

    print("[INFO] 使用 transformers 加载 tokenizer...")
    try:
        tokenizer = _build_hf_tokenizer(args)
    except Exception as err:
        print(f"[FAILED] transformers 无法加载 tokenizer: {err}")
        sys.exit(1)

    print("[INFO] 使用 MindFormers 加载模型...")
    try:
        model = _build_ms_model(args)
    except Exception as err:
        print(f"[FAILED] MindFormers 无法加载模型: {err}")
        sys.exit(1)

    encoded = tokenizer(
        args.sample_text,
        return_attention_mask=True,
        return_tensors="np",
        padding=True,
        truncation=False,
    )

    input_ids = ms.Tensor(encoded["input_ids"], dtype=ms.int32)
    attention_mask = ms.Tensor(encoded["attention_mask"], dtype=ms.int32)

    print("[INFO] 开始MindFormers生成...")
    try:
        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    except Exception as err:
        print(f"[FAILED] MindFormers generate 调用失败: {err}")
        sys.exit(1)

    if hasattr(sequences, "asnumpy"):
        sequences = sequences.asnumpy()
    elif isinstance(sequences, (list, tuple)):
        sequences = np.array([s.asnumpy() if hasattr(s, "asnumpy") else s for s in sequences])
    else:
        sequences = np.array(sequences)

    generated = sequences[:, encoded["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)

    print("[OK] 混合方案运行成功。")
    print(f"  tokenizer padding_side: {tokenizer.padding_side}")
    print(f"  vocab_size: {tokenizer.vocab_size}")
    print(f"  special_tokens_map: {tokenizer.special_tokens_map}")
    print("示例生成输出：")
    for idx, text in enumerate(decoded):
        print(f"  Sample {idx}: {text}")


if __name__ == "__main__":
    main()
