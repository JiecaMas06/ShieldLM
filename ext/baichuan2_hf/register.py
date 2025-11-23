import os
import json
from typing import Any, Dict

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

# 依据 MindFormers 1.7.0：LLaMA 系列配置与模型
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama import LlamaForCausalLM


def _get_pretrained_model_dir(kwargs: Dict[str, Any]) -> str:
    """Try to obtain pretrained_model_dir from multiple possible entries."""
    # 1) 优先从 model_config 中获取（我们已在 YAML 的 model.model_config 同步配置）
    mc = kwargs.get("model_config", {}) if isinstance(kwargs.get("model_config", {}), dict) else {}
    pdir = mc.get("pretrained_model_dir", None)
    if pdir:
        return pdir
    # 2) 其次从顶层 kwargs 获取（如 MindFormers 直接传进来）
    pdir = kwargs.get("pretrained_model_dir", None)
    if pdir:
        return pdir
    # 3) 再尝试环境变量（作为保底，不强依赖）
    pdir = os.environ.get("PRETRAINED_MODEL_DIR", None)
    if pdir:
        return pdir
    raise ValueError("pretrained_model_dir is required for baichuan2_hf, but not found in kwargs or env.")


def _load_hf_config(pretrained_model_dir: str) -> Dict[str, Any]:
    """Load Hugging Face config.json into dict."""
    cfg_path = os.path.join(pretrained_model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"HF config.json not found at: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _map_hf_to_llama_config(hf_cfg: Dict[str, Any], extra_kwargs: Dict[str, Any]) -> LlamaConfig:
    """Map Baichuan2 HF config fields to MindFormers LlamaConfig (1.7.0)."""
    # HF Baichuan2 keys (示例来自用户提供的 config.json)
    hidden_size = hf_cfg["hidden_size"]
    num_hidden_layers = hf_cfg["num_hidden_layers"]
    num_attention_heads = hf_cfg["num_attention_heads"]
    vocab_size = hf_cfg.get("vocab_size", 32000)
    rms_norm_eps = hf_cfg.get("rms_norm_eps", 1e-6)
    intermediate_size = hf_cfg.get("intermediate_size", hidden_size * 4)
    model_max_length = hf_cfg.get("model_max_length", 4096)

    # LlamaConfig 参数名按 1.7.0 常见命名：hidden_size、num_layers、num_heads、vocab_size、rms_norm_eps、intermediate_size、seq_length
    # 其余保持默认（如 rope 参数、kv heads、attention impl 等）；需要可在此扩展。
    llama_cfg = LlamaConfig(
        hidden_size=hidden_size,
        num_layers=num_hidden_layers,
        num_heads=num_attention_heads,
        vocab_size=vocab_size,
        rms_norm_eps=rms_norm_eps,
        intermediate_size=intermediate_size,
        seq_length=model_max_length,
        **{k: v for k, v in extra_kwargs.items() if k not in ("model_config", "pretrained_model_dir")}
    )
    return llama_cfg


@MindFormerRegister.register(MindFormerModuleType.CONFIG, alias="baichuan2_hf")
def build_baichuan2_hf_config(**kwargs) -> LlamaConfig:
    """CONFIG builder for alias 'baichuan2_hf'.

    优先读取 YAML 中的 model.model_config.pretrained_model_dir，回退到顶层同名字段/环境变量。
    将 HF Baichuan2 的 config.json 字段映射为 LlamaConfig，以便直接复用 LLaMA 实现进行推理。
    """
    pretrained_model_dir = _get_pretrained_model_dir(kwargs)
    hf_cfg = _load_hf_config(pretrained_model_dir)
    return _map_hf_to_llama_config(hf_cfg, kwargs)


@MindFormerRegister.register(MindFormerModuleType.MODELS, alias="baichuan2_hf")
def build_baichuan2_hf_model(config: LlamaConfig) -> LlamaForCausalLM:
    """MODEL builder for alias 'baichuan2_hf' based on LLaMA CausalLM."""
    return LlamaForCausalLM(config)


