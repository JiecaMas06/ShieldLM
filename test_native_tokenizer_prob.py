#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""使用 MindFormers 原生管线加载 tokenizer / 模型并抽取首 token 概率。

此脚本完全复用了 run_mindformers.py 与 mindspore_trainer.Trainer 的初始化方式：
1. 通过 MindFormerConfig 读取 YAML，并在需要时覆盖 device / checkpoint。
2. 调用 build_context 与 Trainer 完成环境、模型、tokenizer 初始化。
3. 手动构造输入，前向获得 logits，并输出首 token 的概率 Top-K，方便排障。
"""
import argparse
import os
from typing import Any, Dict, Optional

import mindspore as ms
import numpy as np
from mindspore import Tensor, ops

from mindformers.tools import set_output_path
from mindformers.core.context import build_context
from mindformers.tools.register import MindFormerConfig, MindFormerRegister
from mindformers import Trainer, AutoTokenizer
from mindformers.tools.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MindFormers 原生 tokenizer / 模型加载 + 首 token 概率调试")
    parser.add_argument("--config", type=str, required=True, help="MindFormers YAML 配置文件")
    parser.add_argument("--device_id", type=int, default=None, help="可选：覆盖 YAML 中的 device_id")
    parser.add_argument("--device_target", type=str, default=None, help="可选：覆盖 device_target，例如 Ascend")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="可选：覆盖配置中的 load_checkpoint")
    parser.add_argument("--sample_text", type=str, required=True, help="需要计算首 token 概率的输入文本")
    parser.add_argument("--top_k", type=int, default=5, help="输出概率最高的前 K 个 token")
    parser.add_argument("--show_tokens", action="store_true", help="输出完整 token 序列 / mask 以便排查")
    return parser.parse_args()


def _resolve_config_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def _build_tokenizer_from_config(cfg: MindFormerConfig) -> Any:
    """优先按照 YAML 中 processor.tokenizer 的方式构建原生 tokenizer。"""
    if hasattr(cfg, "processor") and getattr(cfg.processor, "tokenizer", None) is not None:
        logger.info("使用 cfg.processor.tokenizer 定义构建原生 tokenizer ...")
        return MindFormerRegister.get_instance_from_cfg(cfg.processor.tokenizer, "tokenizer")
    if getattr(cfg, "tokenizer", None) is not None:
        logger.info("使用 cfg.tokenizer 定义构建原生 tokenizer ...")
        return MindFormerRegister.get_instance_from_cfg(cfg.tokenizer, "tokenizer")

    # 兜底：尝试根据 pretrained_model_dir 或 load_checkpoint 位置推断
    pretrained_dir = None
    if hasattr(cfg, "model") and hasattr(cfg.model, "model_config"):
        pretrained_dir = getattr(cfg.model.model_config, "pretrained_model_dir", None)
    if pretrained_dir is None:
        pretrained_dir = getattr(cfg, "pretrained_model_dir", None)
    if pretrained_dir is None:
        pretrained_dir = cfg.load_checkpoint

    if pretrained_dir is None:
        raise ValueError("无法根据配置推断 tokenizer 来源，请在 YAML 中提供 processor.tokenizer 或补充 pretrained_model_dir。")

    logger.warning("未在配置中找到 processor.tokenizer，退回到 AutoTokenizer.from_pretrained(%s)", pretrained_dir)
    return AutoTokenizer.from_pretrained(pretrained_dir)


def _extract_model_and_tokenizer(trainer: Trainer, cfg: MindFormerConfig):
    """优先复用 trainer 内部已经初始化好的 tokenizer / model。"""
    tokenizer = getattr(trainer.trainer, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(trainer, "tokenizer", None)
    if tokenizer is None:
        tokenizer = _build_tokenizer_from_config(cfg)
        logger.warning("未能从 Trainer 中直接获取 tokenizer，已根据配置手动构建。")

    model = trainer.model
    model.set_train(False)
    return model, tokenizer


def _to_tensor(value, dtype=ms.int32) -> Tensor:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    return Tensor(arr, dtype=dtype)


def _prepare_inputs(tokenizer, text: str) -> Dict[str, Tensor]:
    try:
        encoded = tokenizer(text, return_attention_mask=True, padding=True, return_tensors="np")
    except Exception:
        ids = tokenizer.encode(text)
        encoded = {
            "input_ids": np.array([ids], dtype=np.int32),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int32)
        }
    input_ids = _to_tensor(encoded["input_ids"], dtype=ms.int32)
    attention_mask = _to_tensor(encoded.get("attention_mask", np.ones_like(encoded["input_ids"])), dtype=ms.int32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "raw": encoded}


def _analyze_logits(logits: Tensor, tokenizer, top_k: int):
    last_step = logits[:, -1, :]
    softmax = ops.Softmax(axis=-1)
    probs = softmax(last_step)
    probs_np = probs.asnumpy()[0]

    topk = min(top_k, probs_np.shape[-1])
    top_indices = probs_np.argsort()[::-1][:topk]
    top_pairs = [(int(tok_id), float(probs_np[tok_id])) for tok_id in top_indices]

    lines = []
    for idx, (tok_id, prob) in enumerate(top_pairs):
        token_str = tokenizer.convert_ids_to_tokens(tok_id)
        lines.append(f"#{idx + 1}: id={tok_id:<6} prob={prob:.6f} token={token_str}")
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg_path = _resolve_config_path(args.config)
    cfg = MindFormerConfig(cfg_path, run_mode="predict")

    if args.device_id is not None:
        cfg.context.device_id = args.device_id
    if args.device_target is not None:
        cfg.context.device_target = args.device_target
    if args.load_checkpoint is not None:
        cfg.load_checkpoint = args.load_checkpoint

    logger.info("==== 构建 MindSpore 上下文 ====")
    set_output_path(cfg.output_dir)
    build_context(cfg)

    logger.info("==== 初始化 Trainer (run_mode=predict) ====")
    trainer = Trainer(cfg)
    model, tokenizer = _extract_model_and_tokenizer(trainer, cfg)

    logger.info("==== tokenizer / 模型信息 ====")
    logger.info("tokenizer: %s", tokenizer.__class__.__name__)
    logger.info("  vocab_size=%s pad_token=%s pad_id=%s eos_token=%s eos_id=%s",
                getattr(tokenizer, "vocab_size", "?"),
                getattr(tokenizer, "pad_token", None),
                getattr(tokenizer, "pad_token_id", None),
                getattr(tokenizer, "eos_token", None),
                getattr(tokenizer, "eos_token_id", None))
    logger.info("model: %s", model.__class__.__name__)

    inputs = _prepare_inputs(tokenizer, args.sample_text)
    if args.show_tokens:
        logger.info("input_ids=%s", inputs["raw"]["input_ids"])
        logger.info("attention_mask=%s", inputs["raw"].get("attention_mask"))

    logger.info("==== 前向推理，获取首 token 概率 ====")
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = getattr(outputs, "logits", None)
    if logits is None:
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        elif isinstance(outputs, dict):
            logits = outputs.get("logits")
    if logits is None:
        raise RuntimeError("模型输出中未找到 logits，无法计算概率。")

    report = _analyze_logits(logits, tokenizer, args.top_k)
    print("[OK] 首 token 概率 Top-{}:".format(args.top_k))
    print(report)


if __name__ == "__main__":
    main()
