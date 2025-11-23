import argparse
import os
import sys

import mindspore as ms
from mindformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="在 Ascend 上测试 MindFormers 是否能复用本地 HuggingFace tokenizer")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="包含 HuggingFace tokenizer 资产的本地目录（需至少包含 tokenizer.json / vocab + merges 等文件）",
    )
    parser.add_argument(
        "--sample_text",
        type=str,
        default="Hello, MindFormers!",
        help="用于分词与解码回测的示例文本，默认: 'Hello, MindFormers!'",
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
        help="若指定，则强制尝试加载 fast 版 tokenizer（若目录不含 fast 资产会失败）",
    )
    parser.add_argument(
        "--padding_side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="设置 padding_side，需与模型推理保持一致，默认 left",
    )
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"未找到 tokenizer 目录: {path}")
    expected = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    if not any(os.path.exists(os.path.join(path, name)) for name in expected):
        raise RuntimeError(
            "目录中未发现常见 tokenizer 资产 (tokenizer.json / vocab.json / merges.txt 等)，"
            "请确认传入的是 HuggingFace 模型文件夹。"
        )


def main():
    args = parse_args()
    ensure_dir(args.tokenizer_dir)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir,
            use_fast=args.use_fast,
            padding_side=args.padding_side,
            local_files_only=True,
        )
    except Exception as err:
        print(f"[FAILED] 无法通过 MindFormers 读取 HuggingFace tokenizer：{err}")
        sys.exit(1)

    # 兜底补齐特殊 token（若 HF 资产缺失会在推理中报错，这里顺便提醒）
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(tokenizer, "pad_token", None) is None:
        print("[WARN] tokenizer 不包含 pad_token，后续推理需手动指定。")

    encoded = tokenizer(
        args.sample_text,
        return_attention_mask=True,
        padding="longest",
        truncation=False,
    )
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)

    print("[OK] 成功加载 tokenizer。关键属性如下：")
    print(f"  vocab_size: {tokenizer.vocab_size}")
    print(f"  padding_side: {tokenizer.padding_side}")
    print(f"  model_max_length: {tokenizer.model_max_length}")
    print(f"  special_tokens_map: {tokenizer.special_tokens_map}")
    print("示例分词结果：")
    print(f"  input_ids: {encoded['input_ids']}")
    print(f"  attention_mask: {encoded['attention_mask']}")
    print(f"  decode -> {decoded}")


if __name__ == "__main__":
    main()
