"""Qwen2 模型适配器

第五阶段新增：将 HF Qwen2/Qwen2.5 模型转换为自研前向所需的权重格式。
使用通用 TransformerWeights/TransformerLayerWeights 数据结构，
与通用 TransformerModelRunner 配合。
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from miniservellm.config import ModelConfig
from miniservellm.runtime.model_runner import TransformerLayerWeights, TransformerWeights


def _import_transformers():
    """延迟导入 transformers 库，避免模块加载时触发不必要的依赖检查"""
    try:
        import transformers  # type: ignore
        return transformers
    except ImportError as e:
        raise ImportError("Please install transformers: pip install transformers") from e


def _require_attr(obj: Any, path: str):
    """按点分路径获取嵌套属性，缺失时抛出明确的错误信息

    例: _require_attr(model, "model.layers.0.self_attn")
    等价于 model.model.layers[0].self_attn，但会检查每一步是否存在
    """
    cur = obj
    for name in path.split("."):
        if not hasattr(cur, name):
            raise AttributeError(f"Missing attribute path '{path}', at '{name}'")
        cur = getattr(cur, name)
    return cur


def _clone_to_cpu_contiguous(t: torch.Tensor) -> torch.Tensor:
    """将张量 detach、搬到 CPU、转为连续内存布局

    detach: 断开计算图，避免持有原始模型的引用导致显存无法释放
    cpu: 统一存到 CPU，后续由 move_weights_to_device 按需搬到 GPU
    contiguous: 确保内存连续，避免后续计算中的隐式拷贝
    """
    return t.detach().to("cpu").contiguous()


def _extract_proj(proj_module):
    """从 HF Linear 层中提取权重和可选的偏置

    Args:
        proj_module: HF 模型中的 nn.Linear 层

    Returns:
        (weight, bias) 元组，bias 可能为 None
    """
    weight = _clone_to_cpu_contiguous(proj_module.weight)
    bias = _clone_to_cpu_contiguous(proj_module.bias) if proj_module.bias is not None else None
    return weight, bias


class Qwen2Adapter:
    """Qwen2/Qwen2.5 模型适配器"""

    def load_tokenizer(self, model_name_or_path: str, trust_remote_code: bool = False) -> Any:
        """加载 tokenizer，优先从本地缓存读取

        Args:
            model_name_or_path: HuggingFace 模型标识符或本地路径
            trust_remote_code: 是否信任远程代码

        Returns:
            HF AutoTokenizer 实例
        """
        transformers = _import_transformers()
        return transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )

    def load_hf_config(self, model_name_or_path: str, trust_remote_code: bool = False) -> Any:
        """加载 HF 模型配置（config.json），提取模型结构参数

        Args:
            model_name_or_path: HuggingFace 模型标识符或本地路径
            trust_remote_code: 是否信任远程代码

        Returns:
            HF AutoConfig 实例
        """
        transformers = _import_transformers()
        return transformers.AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )

    def convert_hf_config(self, hf_config: Any) -> ModelConfig:
        """将 HF Config 转换为自研 ModelConfig

        从 hf_config 中提取模型结构参数，处理不同版本 config 格式差异：
        - num_key_value_heads: GQA 模型专有，不存在时退化为 num_attention_heads（即 MHA）
        - rope_theta: 可能在顶层属性、rope_parameters 或 rope_scaling 字典中
        - max_position_embeddings: 不存在时默认 32768
        - rms_norm_eps: 不存在时默认 1e-6

        Args:
            hf_config: HF AutoConfig 实例

        Returns:
            自研 ModelConfig 实例
        """
        vocab_size = int(hf_config.vocab_size)
        hidden_size = int(hf_config.hidden_size)
        intermediate_size = int(hf_config.intermediate_size)
        num_hidden_layers = int(hf_config.num_hidden_layers)
        num_attention_heads = int(hf_config.num_attention_heads)
        num_key_value_heads = int(getattr(hf_config, "num_key_value_heads", num_attention_heads))
        head_dim = hidden_size // num_attention_heads
        max_position_embeddings = int(getattr(hf_config, "max_position_embeddings", 32768))
        # rope_theta 可能在顶层属性，也可能在 rope_parameters/rope_scaling 字典中
        rope_theta = 10000.0
        if hasattr(hf_config, "rope_theta"):
            rope_theta = float(hf_config.rope_theta)
        elif hasattr(hf_config, "rope_parameters") and isinstance(hf_config.rope_parameters, dict):
            rope_theta = float(hf_config.rope_parameters.get("rope_theta", 10000.0))
        elif hasattr(hf_config, "rope_scaling") and isinstance(hf_config.rope_scaling, dict):
            rope_theta = float(hf_config.rope_scaling.get("rope_theta", 10000.0))
        rms_norm_eps = float(getattr(hf_config, "rms_norm_eps", 1e-6))
        model_type = str(getattr(hf_config, "model_type", "qwen2"))

        return ModelConfig(
            model_type=model_type,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )

    def load_hf_model(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> Any:
        """加载完整的 HF 模型（含权重）

        加载后切换到 eval 模式，关闭 dropout 等训练行为。

        Args:
            model_name_or_path: HuggingFace 模型标识符或本地路径
            device: 加载到哪个设备，默认 CPU
            torch_dtype: 指定精度，None 则使用模型默认精度
            trust_remote_code: 是否信任远程代码

        Returns:
            HF AutoModelForCausalLM 实例
        """
        transformers = _import_transformers()
        kwargs = {"trust_remote_code": trust_remote_code, "local_files_only": True}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model.to(device)
        model.eval()
        return model

    def extract_weights(self, hf_model: Any) -> TransformerWeights:
        """从 HF 模型中提取权重，转换为自研前向所需的格式

        提取的权重及其在网络结构中的对应关系：

        全局权重（TransformerWeights）：
        - embed_tokens [vocab_size, hidden_size]: 词嵌入层，token id → 隐藏向量
        - lm_head [vocab_size, hidden_size]: 输出投影层，隐藏向量 → 词表 logits
          （与 embed_tokens 共享权重，即 tied weights）
        - final_norm [hidden_size]: 最终 RMSNorm，lm_head 前的归一化
        - layers: 24 个 TransformerLayerWeights，每层结构如下：

        每层权重（TransformerLayerWeights）：
        Attention 部分（self_attn）：
        - q_proj [num_heads*head_dim, hidden_size]: Query 投影
        - k_proj [num_kv_heads*head_dim, hidden_size]: Key 投影
        - v_proj [num_kv_heads*head_dim, hidden_size]: Value 投影
        - o_proj [hidden_size, num_heads*head_dim]: Output 投影
        - q_proj_bias [num_heads*head_dim]: Query 偏置（可选）
        - k_proj_bias [num_kv_heads*head_dim]: Key 偏置（可选）
        - v_proj_bias [num_kv_heads*head_dim]: Value 偏置（可选）

        MLP 部分（SwiGLU FFN）：
        - gate_proj [intermediate_size, hidden_size]: SwiGLU 门控分支
        - up_proj [intermediate_size, hidden_size]: SwiGLU 上行分支
        - down_proj [hidden_size, intermediate_size]: SwiGLU 下行投影

        Norm 部分：
        - input_layernorm [hidden_size]: Attention 前的 RMSNorm
        - post_attention_layernorm [hidden_size]: FFN 前的 RMSNorm
        """
        core_model = _require_attr(hf_model, "model")
        embed_tokens = _clone_to_cpu_contiguous(_require_attr(core_model, "embed_tokens").weight)
        layers_mod = _require_attr(core_model, "layers")
        layers: List[TransformerLayerWeights] = []

        for i in range(len(layers_mod)):
            layer = layers_mod[i]
            self_attn = _require_attr(layer, "self_attn")
            mlp = _require_attr(layer, "mlp")

            # Attention 投影：Q/K/V 提取 weight + bias，O 只提取 weight
            q_w, q_b = _extract_proj(_require_attr(self_attn, "q_proj"))
            k_w, k_b = _extract_proj(_require_attr(self_attn, "k_proj"))
            v_w, v_b = _extract_proj(_require_attr(self_attn, "v_proj"))
            o_w, _ = _extract_proj(_require_attr(self_attn, "o_proj"))

            layers.append(
                TransformerLayerWeights(
                    q_proj=q_w,
                    k_proj=k_w,
                    v_proj=v_w,
                    o_proj=o_w,
                    q_proj_bias=q_b,
                    k_proj_bias=k_b,
                    v_proj_bias=v_b,
                    # SwiGLU MLP：gate/up/down 三个投影，只有 weight 没有 bias
                    gate_proj=_clone_to_cpu_contiguous(_require_attr(mlp, "gate_proj").weight),
                    up_proj=_clone_to_cpu_contiguous(_require_attr(mlp, "up_proj").weight),
                    down_proj=_clone_to_cpu_contiguous(_require_attr(mlp, "down_proj").weight),
                    # 两个 RMSNorm：Attention 前 和 FFN 前，只有 weight 没有 bias
                    input_layernorm=_clone_to_cpu_contiguous(_require_attr(layer, "input_layernorm").weight),
                    post_attention_layernorm=_clone_to_cpu_contiguous(
                        _require_attr(layer, "post_attention_layernorm").weight
                    ),
                )
            )

        # 全局最外层的 RMSNorm
        final_norm = _clone_to_cpu_contiguous(_require_attr(core_model, "norm").weight)
        # lm_head 与 embed_tokens 共享权重（tied weights）
        # 只存一份，节省 ~260 MB 显存
        embed_tokens_weight = _clone_to_cpu_contiguous(_require_attr(core_model, "embed_tokens").weight)

        return TransformerWeights(
            embed_tokens=embed_tokens_weight,
            layers=layers,
            final_norm=final_norm,
            lm_head=embed_tokens_weight,
        )

    def move_weights_to_device(
        self,
        weights: TransformerWeights,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TransformerWeights:
        """将 CPU 上的权重搬到指定设备并转换精度

        逐层移动所有权重张量，Optional 类型的 bias 在移动时保持 None。
        lm_head 与 embed_tokens 共享同一块显存（tied weights），
        只移动一次 embed_tokens，lm_head 指向同一对象。

        Args:
            weights: CPU 上的 TransformerWeights
            device: 目标设备（cuda/mps/cpu）
            dtype: 目标精度（float16/bfloat16/float32）

        Returns:
            搬到目标设备后的 TransformerWeights
        """
        def move(t: torch.Tensor) -> torch.Tensor:
            """将张量搬到目标设备和精度"""
            return t.to(device=device, dtype=dtype).contiguous()

        def move_optional(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            """移动 Optional 张量，None 保持不变"""
            return move(t) if t is not None else None

        moved_layers = []
        for layer in weights.layers:
            moved_layers.append(
                TransformerLayerWeights(
                    q_proj=move(layer.q_proj),
                    k_proj=move(layer.k_proj),
                    v_proj=move(layer.v_proj),
                    o_proj=move(layer.o_proj),
                    q_proj_bias=move_optional(layer.q_proj_bias),
                    k_proj_bias=move_optional(layer.k_proj_bias),
                    v_proj_bias=move_optional(layer.v_proj_bias),
                    gate_proj=move(layer.gate_proj),
                    up_proj=move(layer.up_proj),
                    down_proj=move(layer.down_proj),
                    input_layernorm=move(layer.input_layernorm),
                    post_attention_layernorm=move(layer.post_attention_layernorm),
                )
            )

        moved_embed = move(weights.embed_tokens)
        return TransformerWeights(
            embed_tokens=moved_embed,
            layers=moved_layers,
            final_norm=move(weights.final_norm),
            lm_head=moved_embed,  # tied weights: 与 embed_tokens 共享同一块显存
        )
