# Gemma 4 E2B 纯文本推理适配整改清单

## 0. 文档目的

本文档用于指导 `mini-serve-llm` 在保留现有 Qwen2.5-0.5B 支持的前提下，新增 Google Gemma 4 E2B（通常简称 Gemma 4 2B）的纯文本推理能力。

本文档不是“把模型名称换掉”的接入说明，而是针对当前自研 Transformer Runner、Paged KV Cache、Prefill/Decode 调度和 CUDA 优化链路的完整整改清单。

目标模型默认指：

```text
google/gemma-4-E2B-it
```

Gemma 4 E2B 的 “E2B” 表示有效参数规模约为 2.3B；包含 Per-Layer Embedding（PLE）和 embedding 后，静态权重总量约为 5.1B。后续所有显存和性能评估都必须按实际 checkpoint 的总权重规模计算，不能按普通 2B 模型估算。

---

## 1. 目标、范围和明确不做的内容

### 1.1 本阶段目标

- [ ] 在不破坏 Qwen2.5-0.5B 现有行为的情况下，加载 Gemma 4 E2B 指令模型。
- [ ] 支持纯文本 `input_ids` 输入。
- [ ] 支持 Gemma 4 自带的文本 tokenizer 和 chat template。
- [ ] 支持单请求文本 prefill。
- [ ] 支持单请求文本 decode。
- [ ] 使用现成 Ollama `gemma4:e2b` 作为端到端功能和性能基线。
- [ ] 使用独立 PyTorch eager reference 校验自研 Runner 与优化算子的数值误差。
- [ ] 支持 greedy 生成和常规 temperature/top-k/top-p 采样。
- [ ] 支持 EOS 停止和最大生成长度停止。
- [ ] 在正确性通过后，再接入当前的 batching、Paged KV Cache 和 CUDA 优化。

### 1.2 本阶段暂不支持

以下内容不属于本阶段的验收范围，不能为了“完整 Gemma 4”而混入第一版纯文本整改：

- [ ] Vision encoder。
- [ ] Audio encoder。
- [ ] Video 输入。
- [ ] 图像预处理和图像 token 替换。
- [ ] 音频特征提取和音频 token 替换。
- [ ] 多模态 attention mask。
- [ ] `pixel_values`、`input_features` 等多模态输入。
- [ ] Gemma 4 26B-A4B MoE。
- [ ] Gemma 4 E4B、12B、31B 的兼容性承诺。
- [ ] MLX Gemma4 后端。
- [ ] 一开始就实现 Gemma4 专用 CUDA 融合 kernel。

> 说明：E2B 本身是 dense 模型，当前官方配置中 `enable_moe_block=false`。本阶段不需要实现 MoE，但实现时不能把代码写死成“未来永远没有 MoE”。

### 1.3 兼容性原则

- [ ] Qwen2.5 继续使用现有 `Qwen2Adapter` 和 Qwen Runner。
- [ ] Gemma4 使用独立的 `Gemma4Adapter` 和 `Gemma4TextRunner`，或者使用清晰隔离的模型结构分支。
- [ ] 共享 Scheduler、Request、Sampler 和引擎生命周期接口。
- [ ] 不在 `Qwen2Adapter` 中堆积 Gemma4 的路径判断。
- [ ] 不通过修改 Qwen 默认值来“伪装支持” Gemma4。
- [ ] 不在未通过 HF 数值对齐前启用 CUDA Graph、量化和激进融合。

---

## 2. 验收前必须固定的 Gemma 4 E2B 模型事实

以下字段必须从实际 checkpoint 的 `config.json` 和 Transformers 解析后的 `text_config` 中读取，并在启动日志中打印。不能只依赖本文档中的默认值。

### 2.1 已知的 E2B 参考配置

- [ ] `model_type` 为 `gemma4`。
- [ ] 文本子配置的 `model_type` 为 `gemma4_text` 或当前 Transformers 版本对应的文本类型。
- [ ] `num_hidden_layers = 35`。
- [ ] `hidden_size = 1536`。
- [ ] `intermediate_size = 6144`。
- [ ] `num_attention_heads = 8`。
- [ ] 基础 `num_key_value_heads = 1`。
- [ ] 普通 sliding attention 层的 `head_dim = 256`。
- [ ] full/global attention 层使用 `global_head_dim = 512`。
- [ ] `vocab_size = 262144`。
- [ ] `max_position_embeddings = 131072`。
- [ ] `sliding_window = 512`。
- [ ] `rms_norm_eps = 1e-6`。
- [ ] `hidden_activation = "gelu_pytorch_tanh"`。
- [ ] `use_double_wide_mlp = true`。
- [ ] `num_kv_shared_layers = 20`。
- [ ] `final_logit_softcapping = 30.0`。
- [ ] `tie_word_embeddings = true`。
- [ ] `attention_bias = false`。
- [ ] `attention_k_eq_v = false`。
- [ ] `enable_moe_block = false`。
- [ ] `hidden_size_per_layer_input = 256`。
- [ ] `vocab_size_per_layer_input = 262144`。

### 2.2 按层解析要求

Gemma4 不能只保存一个全局 `head_dim`。必须读取或构造解析后的 per-layer 配置：

```text
layer_specs[i].attention_type
layer_specs[i].head_dim
layer_specs[i].num_attention_heads
layer_specs[i].num_key_value_heads
layer_specs[i].rotary_dim
layer_specs[i].rope_type
layer_specs[i].rope_theta
layer_specs[i].kv_source_layer
layer_specs[i].mlp_intermediate_size
layer_specs[i].use_double_wide_mlp
```

- [ ] 确认 `layer_types` 的实际长度等于 `num_hidden_layers`。
- [ ] 确认每个元素只能是支持的注意力类型。
- [ ] 确认最后一层是 `full_attention`。
- [ ] 确认 full 层的 `head_dim` 不是误读成 256。
- [ ] 确认每层 `q_proj/k_proj/v_proj/o_proj` 的 shape 与 `layer_specs[i]` 一致。
- [ ] 确认 `num_global_key_value_heads` 是否对当前 checkpoint 生效。
- [ ] 确认 `per_layer_config` 是从配置自动生成还是由 checkpoint 显式提供。
- [ ] 将最终解析结果序列化成调试 JSON，便于回归比较。

### 2.3 模型事实检查脚本

建议新增：

```text
scripts/inspect_gemma4_text_config.py
```

脚本至少输出：

- [ ] 外层 `model_type`。
- [ ] 文本模型类名。
- [ ] `text_config` 全量字段。
- [ ] 每层 attention 类型。
- [ ] 每层 Q/K/V/O 投影 shape。
- [ ] 每层 Norm shape。
- [ ] 每层 MLP shape。
- [ ] PLE 相关权重 shape。
- [ ] `lm_head` 与 embedding 是否共享。
- [ ] KV sharing 来源层映射。
- [ ] Transformers 版本。
- [ ] checkpoint 路径和权重文件列表。

脚本必须支持本地模型目录，避免在离线环境中误触发下载。

---

## 3. 当前代码缺口盘点

### 3.1 全局配置：`miniservellm/config.py`

当前存在以下 Qwen 假设：

- [ ] `MODEL_NAME` 固定为 Qwen2.5-0.5B。
- [ ] `ModelConfig` 只有一个全局 `head_dim`。
- [ ] `ModelConfig` 只有一个全局 `rope_theta`。
- [ ] `EngineConfig.create()` 的自动显存参数默认使用 Qwen 的层数、KV head 数和 head dimension。
- [ ] `compute_num_gpu_blocks()` 使用统一的 `[layers, kv_heads, head_dim]` KV Cache 公式。
- [ ] `EngineConfig.eos_token_id` 只允许单个 `int`，不能表达多个 EOS。

整改要求：

- [ ] 保留 Qwen 默认配置，但不得让 Gemma4 复用 Qwen 默认值。
- [ ] 增加模型架构类型或后端类型字段。
- [ ] 增加 `LayerSpec`、`RopeSpec`、`KVSharingSpec` 等内部数据结构。
- [ ] 让 `ModelConfig` 支持按层描述。
- [ ] 将 EOS 类型改为 `int | list[int] | set[int] | None`，或增加标准化的 `eos_token_ids`。
- [ ] 自动显存计算改为按实际每层 KV 存储需求求和。
- [ ] 对共享 KV 层只计算实际存储量，避免重复估算。
- [ ] 对 sliding/full 层分别计算可缓存 token 范围。
- [ ] 自动调优无法处理异构 cache 时必须明确报错或切换保守模式，不能静默使用 Qwen 公式。

推荐的核心数据结构：

```python
@dataclass
class LayerSpec:
    layer_idx: int
    attention_type: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_type: str
    rope_theta: float
    sliding_window: int | None
    kv_source_layer: int | None
    mlp_intermediate_size: int
    use_double_wide_mlp: bool
```

### 3.2 Adapter：`miniservellm/model_adapter/adapters/qwen2_adapter.py`

当前 Qwen Adapter 的以下逻辑不能直接用于 Gemma4：

- [ ] 配置字段从顶层读取，无法处理嵌套 `text_config`。
- [ ] `head_dim` 固定使用 `hidden_size // num_attention_heads`。
- [ ] RoPE 只支持一个 theta。
- [ ] 权重根路径固定为 Qwen 风格。
- [ ] 只提取两个 RMSNorm。
- [ ] 无 q/k/v norm。
- [ ] 无 PLE 权重。
- [ ] 无按层 head dimension。
- [ ] 无 KV sharing 映射。
- [ ] 无 Double-Wide MLP。
- [ ] 无 `final_logit_softcapping`。
- [ ] 无独立的 Gemma4 tokenizer/processor 处理。
- [ ] 无条件把 embedding 当作 lm_head。

整改要求：

- [ ] 新增 `miniservellm/model_adapter/adapters/gemma4_adapter.py`。
- [ ] Qwen Adapter 保持 Qwen 专用，不增加 Gemma4 分支。
- [ ] 新增 Adapter factory/registry。
- [ ] 根据 `model_type` 选择 Adapter。
- [ ] 对未知 `model_type` 给出明确错误。
- [ ] 加载配置时不依赖权重模型已加载。
- [ ] 先解析文本配置，再决定加载哪种文本模型类。

### 3.3 Runner：`miniservellm/runtime/model_runner.py`

当前 Runner 的核心假设是：

```text
所有层使用相同的 head_dim
所有层使用相同的 attention
所有层有独立 KV
每层只有两个 RMSNorm
MLP 永远是 SiLU + Mul
模型只有一个输入 embedding
RoPE 只有一套
```

这些假设对 Gemma4 E2B 均不成立，必须整改或隔离为新的 Runner。

### 3.4 算子：`miniservellm/runtime/nn_ops.py`

当前已有可复用的基础能力：

- [ ] 标准 RMSNorm。
- [ ] Linear。
- [ ] 基础 RoPE。
- [ ] GQA attention。
- [ ] Paged prefill attention。
- [ ] Paged decode attention。
- [ ] CUDA kernel fallback。

当前缺少或不兼容：

- [ ] partial rotary。
- [ ] proportional RoPE。
- [ ] 两种按层 RoPE。
- [ ] GELU-tanh gated MLP。
- [ ] q/k/v norm 组合。
- [ ] sliding-window mask。
- [ ] attention 按层 dispatch。
- [ ] 512 维 head 的 CUDA/SDPA fallback。
- [ ] 可选 attention logit soft cap。
- [ ] final logit soft cap。

### 3.5 KV Cache：`miniservellm/cache/kv_cache.py`

当前物理布局是统一的：

```text
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

Gemma4 需要支持：

- [ ] 不同层不同 `head_dim`。
- [ ] sliding/full 两类 cache 读取策略。
- [ ] 跨层 KV sharing。
- [ ] 共享层不重复写入或按官方规则写入。
- [ ] 来源层和消费层之间的映射。
- [ ] full 层的全上下文 KV。
- [ ] sliding 层的窗口 KV 或窗口 mask。
- [ ] 不同 cache layout 的 block 计数。

### 3.6 入口和引擎

以下文件仍然有 Qwen 入口假设：

- [ ] `scripts/run_stage5_demo.py`。
- [ ] `scripts/inspect_model.py`。
- [ ] `scripts/check_mlx_hf_parity.py`。
- [ ] `miniservellm/runtime/inference_engine.py`。
- [ ] `miniservellm/model_adapter/tokenizer_adapter.py`。

需要参数化模型名称、Adapter、Runner、EOS、Tokenizer 和验证脚本。

---

## 4. 模型适配层整改清单

### 4.1 新增 Adapter factory

建议新增：

```text
miniservellm/model_adapter/adapter_factory.py
```

- [ ] 提供 `create_adapter_from_config(hf_config)`。
- [ ] 支持 `qwen2`、`gemma4`。
- [ ] 支持外层 `gemma4` 和文本子配置 `gemma4_text` 的识别。
- [ ] 不根据模型名称字符串猜测架构。
- [ ] 输出 Adapter 类型和模型类型日志。
- [ ] 对嵌套配置使用文本子配置进行 Runner 构建。
- [ ] 保留显式传入 Adapter 的兼容路径。

推荐加载流程：

```text
model_path
    -> AutoConfig
    -> normalize_config
    -> create_adapter
    -> load tokenizer/processor
    -> load text model
    -> convert ModelConfig
    -> extract weights
    -> create Runner
```

### 4.2 Gemma4 配置转换

`Gemma4Adapter.convert_hf_config()` 必须完成：

- [ ] 提取外层 `Gemma4Config`。
- [ ] 提取嵌套 `text_config`。
- [ ] 读取 `layer_types`。
- [ ] 解析 `per_layer_config`。
- [ ] 读取 `head_dim` 和 `global_head_dim`。
- [ ] 不用 `hidden_size // num_attention_heads` 覆盖显式 head dimension。
- [ ] 解析 sliding/full 对应的 RoPE 参数。
- [ ] 解析 `partial_rotary_factor`。
- [ ] 解析 `sliding_window`。
- [ ] 解析 `num_kv_shared_layers`。
- [ ] 解析 PLE 维度。
- [ ] 解析 `hidden_activation`。
- [ ] 解析 `use_double_wide_mlp`。
- [ ] 解析 `final_logit_softcapping`。
- [ ] 解析 `tie_word_embeddings`。
- [ ] 解析 `attention_k_eq_v`。
- [ ] 解析 `enable_moe_block` 并在 E2B 为 false 时记录。
- [ ] 解析 PAD/BOS/EOS。

转换完成后执行校验：

- [ ] `len(layer_specs) == num_hidden_layers`。
- [ ] 每层 Q head 数可被 KV head 数整除。
- [ ] 每层 projection shape 与 spec 匹配。
- [ ] full 层 head dimension 与 `global_head_dim` 匹配。
- [ ] PLE 总维度等于 `num_hidden_layers * hidden_size_per_layer_input`。
- [ ] `lm_head` shape 为 `[vocab_size, hidden_size]`。
- [ ] 不支持的激活函数直接报错。
- [ ] 不支持的 attention 类型直接报错。

### 4.3 Gemma4 权重结构

建议不要继续复用只有 Qwen 字段的 `TransformerWeights`，而是新增结构或扩展为可选字段：

```python
@dataclass
class Gemma4LayerWeights:
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor

    q_norm: torch.Tensor
    k_norm: torch.Tensor
    v_norm: torch.Tensor

    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    pre_feedforward_layernorm: torch.Tensor
    post_feedforward_layernorm: torch.Tensor
    post_per_layer_input_norm: torch.Tensor

    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    per_layer_input_gate: torch.Tensor
    per_layer_projection: torch.Tensor
    layer_scalar: torch.Tensor | None
```

模型级权重至少包括：

```python
@dataclass
class Gemma4TextWeights:
    embed_tokens: torch.Tensor
    lm_head: torch.Tensor
    final_norm: torch.Tensor

    embed_tokens_per_layer: torch.Tensor
    per_layer_model_projection: torch.Tensor
    per_layer_projection_norm: torch.Tensor

    layers: list[Gemma4LayerWeights]
```

实际字段名称应以 checkpoint 的 `state_dict().keys()` 为准，不要把本文示例中的模块路径当成绝对路径。

### 4.4 权重根路径探测

Adapter 应支持对以下可能的根路径进行明确探测：

- [ ] `model.language_model`.
- [ ] `model`.
- [ ] 文本-only checkpoint 的直接文本根节点。
- [ ] `base_model.model` 等 Transformers 包装层。

探测规则：

- [ ] 优先使用模型类和 `base_model_prefix`。
- [ ] 探测失败时输出候选路径和实际模块树摘要。
- [ ] 不使用静默 fallback 到 Qwen 路径。
- [ ] 每个必需权重缺失时报告完整路径。

### 4.5 QKV 提取

- [ ] 分别提取 `q_proj`、`k_proj`、`v_proj`。
- [ ] 允许 q/k/v bias 全部为空。
- [ ] 如果只有部分 bias，明确拒绝或逐项保存，不能默认拼接。
- [ ] 每层分别拼接 QKV，不能使用全局固定维度。
- [ ] 保存原始 q/k/v shape 到 manifest。
- [ ] 验证 `o_proj.in_features == q_dim`。
- [ ] 验证 `q_dim`、`kv_dim` 与 `LayerSpec` 一致。
- [ ] 保留独立 q/k/v 权重的调试能力；数值对齐后再考虑合并存储。

### 4.6 PLE 权重提取

- [ ] 提取 `embed_tokens_per_layer`。
- [ ] 验证 shape 为 `[vocab_size_per_layer_input, layers * ple_dim]` 或官方等价布局。
- [ ] 提取 `per_layer_model_projection`。
- [ ] 提取 `per_layer_projection_norm`。
- [ ] 提取每层 `per_layer_input_gate`。
- [ ] 提取每层 `per_layer_projection`。
- [ ] 提取每层 `post_per_layer_input_norm`。
- [ ] 保存 PLE scaling 相关配置。
- [ ] 记录 PLE 权重的内存大小。
- [ ] 设计 PLE 的 CPU staging/offload/量化策略。

### 4.7 MLP 权重提取

- [ ] 提取 `gate_proj`。
- [ ] 提取 `up_proj`。
- [ ] 提取 `down_proj`。
- [ ] 记录每层实际中间维度。
- [ ] 判断该层是否启用 Double-Wide。
- [ ] 不把不同宽度的层强行拼成统一矩阵。
- [ ] 对量化 kernel 的偶数 output size 要求做 shape 检查。
- [ ] 对不支持的专家层直接报错，而不是当成普通 MLP。

### 4.8 Norm 权重提取

- [ ] 提取每层四个主 RMSNorm。
- [ ] 提取 q/k/v norm。
- [ ] 确认 v_norm 是否带可学习 scale。
- [ ] 确认权重初始化语义和实际 checkpoint 权重一致。
- [ ] 提取 final norm。
- [ ] 检查所有 Norm 维度。

### 4.9 Embedding 和 LM Head

- [ ] 根据 `tie_word_embeddings` 判断是否共享。
- [ ] 检查 `lm_head.weight` 是否真实存在。
- [ ] 检查 embedding 和 lm_head shape。
- [ ] 共享时只保存一份主 embedding。
- [ ] 不把 `embed_tokens_per_layer` 当成主 embedding。
- [ ] 量化时明确 lm_head 是否保持 FP16/FP32。

---

## 5. ModelConfig 和运行时结构整改清单

### 5.1 引入按层配置

- [ ] 新增 `LayerSpec`。
- [ ] 新增 `RopeSpec`。
- [ ] 新增 `KVSharingSpec`。
- [ ] `ModelConfig` 增加 `layer_specs`。
- [ ] `ModelConfig` 增加 `has_ple`。
- [ ] `ModelConfig` 增加 `final_logit_softcapping`。
- [ ] `ModelConfig` 增加 `hidden_activation`。
- [ ] `ModelConfig` 增加 `tie_word_embeddings`。
- [ ] `ModelConfig` 增加 `eos_token_ids`。
- [ ] 保留旧字段供 Qwen Runner 使用。
- [ ] 对 Qwen 自动生成只有一种 `LayerSpec` 的等价描述。

### 5.2 结构校验器

建议新增：

```text
miniservellm/model_adapter/config_validation.py
```

- [ ] 校验全局 hidden size。
- [ ] 校验层数。
- [ ] 校验每层 head 数。
- [ ] 校验每层 projection shape。
- [ ] 校验 attention 类型。
- [ ] 校验 RoPE 维度不超过 head dimension。
- [ ] 校验 sliding window 为正数。
- [ ] 校验 KV source layer 在合法范围。
- [ ] 校验 PLE shape。
- [ ] 校验 MLP gate/up/down shape。
- [ ] 校验 lm_head shape。
- [ ] 错误信息包含模型名、层号、字段名和实际 shape。

### 5.3 自动显存估算

当前统一 KV Cache 估算公式不适用于 Gemma4。应改为按层求和：

```text
bytes_per_token =
    sum(
        2 * num_kv_heads[i] * head_dim[i] * dtype_bytes
        for i in physical_kv_layers
    )
```

其中 `2` 表示 K 和 V。

- [ ] 对 full/sliding 层分别估算。
- [ ] 对 KV-shared 层只计入真实物理存储。
- [ ] 如果第一版使用保守的独立 cache，明确标记为上界估算。
- [ ] 估算时包含 PLE、embedding、lm_head、Norm 和临时 workspace。
- [ ] 不使用 `num_hidden_layers=24`、`num_kv_heads=2`、`head_dim=64` 等 Qwen 默认值。
- [ ] 3060 6GB 上显存不足时，在初始化阶段给出可解释错误。
- [ ] 不通过盲目减少 block 数掩盖权重本身无法加载的问题。

---

## 6. Gemma4TextRunner 前向整改清单

### 6.1 Runner 设计

建议新增：

```text
miniservellm/runtime/gemma4_text_runner.py
```

或者新增独立的：

```text
miniservellm/runtime/gemma4_model_runner.py
```

- [ ] 不破坏 `TransformerModelRunner` 的 Qwen 行为。
- [ ] 对外提供与现有 Runner 相同的 `forward_fresh_prefill()`。
- [ ] 对外提供与现有 Runner 相同的 `forward_incremental_prefill()`。
- [ ] 对外提供与现有 Runner 相同的 `forward_decode()`。
- [ ] 对外提供相同的 `device`、`dtype`、`has_cuda_graph` 等必要接口。
- [ ] 对外输出继续使用 `PrefillModelOutput` 和 `DecodeModelOutput`。
- [ ] 纯文本第一版优先支持 eager 路径。
- [ ] Gemma4 未通过数值对齐前禁用 compile/graph 快捷路径。

### 6.2 正确的层级前向顺序

实现时以官方 Transformers `Gemma4TextDecoderLayer.forward()` 为最终依据。概念流程应接近：

```text
h = 当前主 hidden state

1. Attention 子层
   residual = h
   x = input_layernorm(h)
   q = q_proj(x)
   k = k_proj(x)
   v = v_proj(x)
   q = q_norm(q)
   k = k_norm(k)
   v = v_norm(v)
   q/k 使用本层对应 RoPE
   attention = sliding 或 full attention
   attention = o_proj(attention)
   attention = post_attention_layernorm(attention)
   h = residual + attention

2. FFN 子层
   residual = h
   x = pre_feedforward_layernorm(h)
   x = Gemma4 MLP(x)
   x = post_feedforward_layernorm(x)
   h = residual + x

3. PLE 注入
   residual = h
   gate = per_layer_input_gate(h)
   gate = GELU_TANH(gate)
   x = gate * per_layer_input[i]
   x = per_layer_projection(x)
   x = post_per_layer_input_norm(x)
   h = residual + x
```

- [ ] 核对 PLE 注入的精确位置。
- [ ] 核对每个 residual 的来源。
- [ ] 核对每个 Norm 的输入和输出。
- [ ] 核对 `layer_scalar` 是否参与当前官方版本的 forward。
- [ ] 不使用当前 Qwen 的“Attention 后直接 fused_add_rms_norm + MLP”快捷流程替代 Gemma4 的四 Norm 结构。

### 6.3 PLE 前向

- [ ] 普通 embedding 输出 `[T, hidden_size]`。
- [ ] 根据 input IDs 查 PLE token identity。
- [ ] 执行官方要求的 `sqrt(ple_dim)` scaling。
- [ ] 执行 `per_layer_model_projection`。
- [ ] 执行 `1 / sqrt(hidden_size)` scaling。
- [ ] reshape 为 `[T, num_layers, ple_dim]`。
- [ ] 执行 `per_layer_projection_norm`。
- [ ] 两个分支按官方规则执行 `1 / sqrt(2)` 合并。
- [ ] 每一层取 `per_layer_inputs[:, layer_idx, :]`。
- [ ] prefill chunk 和 decode 单 token 都正确生成 PLE。
- [ ] batch prefill 不把不同请求的 PLE 混合。
- [ ] padding token 的 PLE 行为与官方实现一致。
- [ ] `inputs_embeds` 路径如果暂不支持，应明确拒绝，而不是错误地反查 token ID。

### 6.4 Attention 前向

- [ ] Q/K/V 投影分别使用每层权重。
- [ ] 根据 `LayerSpec` reshape Q/K/V。
- [ ] 对 Q/K/V 应用对应 Norm。
- [ ] 处理 V Norm 的无 scale/有 scale 变体。
- [ ] 使用官方 attention scaling。
- [ ] 不硬编码 Qwen 的 `D ** -0.5`，除非确认 Gemma4 当前实现确实使用该缩放。
- [ ] sliding 层使用局部窗口。
- [ ] full 层使用全局 causal attention。
- [ ] prefill 正确处理历史 token、当前 chunk 和 causal mask。
- [ ] decode 正确处理单 query。
- [ ] 支持 MQA/GQA 的 KV 广播。
- [ ] 不将跨层 KV sharing 和同层 MQA 混为一谈。

### 6.5 输出投影

- [ ] 依据本层 head dimension reshape attention output。
- [ ] 验证 `o_proj` 输入维度。
- [ ] full 层和 sliding 层分别使用正确的 O projection。
- [ ] 输出维度恢复到 `hidden_size=1536`。
- [ ] 保证不同层的输出都能进入统一 residual stream。

### 6.6 MLP 前向

- [ ] 实现 `gelu_pytorch_tanh`。
- [ ] 实现 gated GELU/GeGLU。
- [ ] 处理 gate/up 的正确顺序。
- [ ] 处理 Double-Wide 层。
- [ ] 根据实际权重 shape 计算 split 维度。
- [ ] 不调用 Qwen 专用 `silu_and_mul()`。
- [ ] 为不支持的激活函数添加显式错误。
- [ ] 检查 down projection 输入维度。

### 6.7 最终 logits

- [ ] 执行 final RMSNorm。
- [ ] 执行 lm_head。
- [ ] 在配置存在时应用 `final_logit_softcapping`。
- [ ] E2B 使用 `tanh(logits / cap) * cap` 的官方等价公式。
- [ ] 确认 soft cap 在采样前生效。
- [ ] 不在不同 prefill/decode 路径使用不同 logits 处理。
- [ ] 词表维度使用 262144，不能保留 Qwen 词表假设。

---

## 7. Attention 算子整改清单

### 7.1 RoPE 算子

建议扩展 `miniservellm/runtime/nn_ops.py`：

- [ ] `build_rope_cache()` 支持 `rotary_dim`。
- [ ] 支持 `rope_type="default"`。
- [ ] 支持 `rope_type="proportional"`。
- [ ] 支持 sliding/full 两套 cache。
- [ ] 支持不同 head dimension。
- [ ] 支持 full 层 `partial_rotary_factor=0.25`。
- [ ] 支持按 layer type 选择 cache。
- [ ] prefill/decode 使用相同的位置定义。
- [ ] 编写独立 RoPE reference test。

### 7.2 Attention scaling 和 soft cap

- [ ] 将 attention scaling 从函数内部常量改为参数。
- [ ] 按模型/层传入 scaling。
- [ ] 如果配置提供 attention logit soft cap，softmax 前应用。
- [ ] E2B 当前若没有 attention soft cap，不要强行添加。
- [ ] final logit soft cap 与 attention soft cap 分开实现。
- [ ] custom CUDA kernel 和 PyTorch fallback 使用同一公式。

### 7.3 Sliding-window mask

- [ ] prefill mask 支持 `key_pos >= query_pos - window + 1` 的局部可见范围。
- [ ] decode 只允许访问窗口内历史 token。
- [ ] chunked prefill 在跨 chunk 时保持全局位置。
- [ ] full 层不误套 sliding mask。
- [ ] batch 中不同请求的历史长度和窗口边界独立计算。
- [ ] padding token 不参与 attention。
- [ ] 验证窗口边界 `511/512/513` token。

### 7.4 512 维 full attention

当前自定义 CUDA kernel、FlashAttention 或其它 backend 可能不支持 `head_dim=512`。

- [ ] 检查 `mini_llm_kernels` 的 attention kernel 最大 head dimension。
- [ ] 检查 `decode_paged_attention` 的 shape 假设。
- [ ] 检查 `paged_prefill_attention` 的 shape 假设。
- [ ] 检查 QKV+RoPE 融合 kernel 是否写死 head dimension。
- [ ] 检查 CUDA Graph 是否允许不同层使用不同 shape。
- [ ] 第一版为 full 层提供 PyTorch SDPA/eager fallback。
- [ ] sliding 层可单独使用现有 kernel，但必须通过 shape gate。
- [ ] 日志打印每一层实际使用的 attention backend。
- [ ] 未支持 512 维时不能静默截断或 reshape 成 256。

---

## 8. KV Cache 重构清单

### 8.1 Cache 物理布局

当前统一五维 Tensor 不能直接承载异构层。可选方案：

#### 方案 A：每层独立 Tensor

```text
k_cache[layer_idx] -> [num_blocks_i, block_size, kv_heads_i, head_dim_i]
v_cache[layer_idx] -> [num_blocks_i, block_size, kv_heads_i, head_dim_i]
```

- [ ] 支持每层不同 shape。
- [ ] block allocator 管理共享的逻辑 block ID。
- [ ] 每层维护自己的物理 tensor。
- [ ] 写入和读取接口增加 `layer_idx`。

#### 方案 B：按 shape 分组

```text
cache_groups[(kv_heads, head_dim)] -> 多层 cache
```

- [ ] 适合多个层 shape 相同的情况。
- [ ] 需要 layer 到 group 的映射。
- [ ] 需要处理 full/sliding 的不同生命周期。

第一版建议优先选择可读性更好的方案 A，数值正确后再优化内存布局。

### 8.2 KV sharing

- [ ] 从官方实现推导每层 KV source mapping。
- [ ] 明确 source layer 和 consumer layer。
- [ ] source layer 必须在 consumer layer 使用前完成计算。
- [ ] consumer layer 不重复计算不存在的 K/V projection。
- [ ] consumer layer 读取正确的 source KV。
- [ ] prefill 和 decode 使用相同映射。
- [ ] batch 请求之间的 KV sharing 不能串请求。
- [ ] 请求结束时释放整套 source/consumer 关联资源。
- [ ] cache debug 信息打印 source layer mapping。
- [ ] 为 `num_kv_shared_layers=0` 保留普通模型兼容路径。

### 8.3 Sliding/full Cache 语义

- [ ] full 层保留官方要求的全上下文范围。
- [ ] sliding 层至少通过 mask 保证只能看到窗口范围。
- [ ] 第一版可暂时保留更多 KV 但使用窗口 mask，作为 correctness-first 方案。
- [ ] 后续再实现 sliding 层的实际窗口裁剪。
- [ ] 窗口裁剪不会破坏 source/consumer KV sharing。
- [ ] 位置编号在裁剪后仍使用全局绝对位置。

### 8.4 Cache API

建议新增或扩展：

```python
ensure_slots_for_request(req, num_new_tokens)
write_kv_for_tokens(layer_idx, ...)
read_kv_for_layer(layer_idx, ...)
read_shared_kv(source_layer_idx, ...)
get_layer_cache_spec(layer_idx)
get_visible_range(layer_idx, query_positions)
```

- [ ] API 不暴露 Qwen 的单一 `head_dim` 假设。
- [ ] `gather_kv_for_request()` 支持窗口范围。
- [ ] `gather_kv_decode_batch()` 支持按层 cache。
- [ ] block table 可以被不同 layer spec 复用。
- [ ] 释放请求时清理所有 layer state。
- [ ] 缓存容量不足时由 Scheduler 正确阻塞请求，不破坏已有 cache。

### 8.5 Cache 显存公式

普通独立 KV 的每 token 字节数：

```text
sum_i (
    2                       # K + V
    * num_kv_heads[i]
    * head_dim[i]
    * dtype_bytes
)
```

实施时还要考虑：

- [ ] KV sharing 后的唯一物理 source 数量。
- [ ] full 层和 sliding 层的有效长度。
- [ ] block padding 浪费。
- [ ] batch 中不同请求的上下文长度。
- [ ] CUDA kernel 临时 workspace。
- [ ] prefill 的 Q/K/V 临时张量。

---

## 9. Prefill、Decode 和 Batch 运行时整改清单

### 9.1 Prefill

- [ ] fresh prefill 支持 PLE。
- [ ] incremental prefill 支持 PLE。
- [ ] chunk 的绝对 position 正确传入。
- [ ] chunk 的 history length 正确传入。
- [ ] sliding mask 同时覆盖 history 和当前 chunk。
- [ ] full 层可以访问完整历史。
- [ ] source KV layer 在同一 prefill step 中正确完成。
- [ ] 只取每个请求最后有效 token 的 logits。
- [ ] 不为中间 chunk 错误采样。
- [ ] prefill 全量运行与分 chunk 运行结果一致。

### 9.2 Decode

- [ ] 单 token PLE lookup 正确。
- [ ] 每个请求的 query position 独立。
- [ ] sliding 层只看窗口内 KV。
- [ ] full 层看完整 KV。
- [ ] KV sharing 不因 decode 顺序变化而失效。
- [ ] batch decode 的不同请求不发生 cache 串扰。
- [ ] `head_dim=256` 和 `head_dim=512` 的层都能正常运行。

### 9.3 Batch

- [ ] 不同 prompt 长度的 batch 正确 padding。
- [ ] padding query 不产生有效 logits。
- [ ] batch 内每个请求的 PLE slice 正确。
- [ ] block-diagonal causal mask 扩展到 sliding/full。
- [ ] batch bucket 不改变实际结果。
- [ ] decode batch bucket 的 padding 行不写入真实请求 cache。
- [ ] batch 结束后所有临时状态释放。

### 9.4 调度器复用边界

以下组件原则上可以复用，但必须增加 Gemma4 专用测试：

- [ ] `Scheduler`。
- [ ] `RequestQueue`。
- [ ] `Request` 状态机。
- [ ] `FreshPrefillRunner`。
- [ ] `IncrementalPrefillRunner`。
- [ ] `DecodeRunner`。
- [ ] `Sampler`。

需要重新检查的接口：

- [ ] EOS 是否允许多个 ID。
- [ ] Cache 容量预检查是否使用新的 layer-aware 公式。
- [ ] CUDA Graph fast path 是否在 Gemma4 上禁用或改造。
- [ ] batch metadata 是否包含 layer-specific attention 信息。

---

## 10. Tokenizer 和纯文本输入整改清单

### 10.1 Tokenizer 加载

- [ ] 使用 Gemma4 官方 tokenizer/processor。
- [ ] 确认 Transformers 版本支持 Gemma4。
- [ ] 确认本地 checkpoint 包含 tokenizer 配置。
- [ ] 支持离线加载。
- [ ] 启动时打印 vocab size、BOS、PAD、EOS。

### 10.2 Chat template

- [ ] 不复用 Qwen 的 `<|im_start|>` 之类模板假设。
- [ ] 使用 `tokenizer.apply_chat_template()`。
- [ ] 支持 system/user/assistant 角色。
- [ ] 支持 `add_generation_prompt=True`。
- [ ] 验证 chat template 是否已经添加 BOS。
- [ ] 避免二次添加 special tokens。
- [ ] 增加 raw text 和 chat prompt 两类测试。
- [ ] 为 thinking/control token 的默认行为做明确配置。

### 10.3 EOS 和停止条件

- [ ] 将 tokenizer 的 EOS 标准化为集合。
- [ ] 支持单个 EOS。
- [ ] 支持多个 EOS。
- [ ] prefill 首 token 检查所有 EOS。
- [ ] decode 每一步检查所有 EOS。
- [ ] 达到 `max_new_tokens` 时正确释放 cache。
- [ ] decode 输出时跳过特殊 token。

建议把当前 `EngineConfig.eos_token_id: Optional[int]` 改为：

```python
eos_token_ids: frozenset[int] | None
```

并保留兼容旧调用方的单值转换逻辑。

---

## 11. 入口、模型注册和配置整改清单

### 11.1 `scripts/run_stage5_demo.py`

- [ ] 增加 `--model` 参数。
- [ ] 增加 `--backend` 或根据 config 自动选择 Runner。
- [ ] 不固定导入 `Qwen2Adapter`。
- [ ] 通过 Adapter factory 选择模型。
- [ ] 模型加载后再使用真实 `ModelConfig` 创建 EngineConfig。
- [ ] 不使用 Qwen 的固定 `num_gpu_blocks` 作为 Gemma4 默认值。
- [ ] 启动时打印模型摘要。
- [ ] 启动时打印每层 attention 统计。
- [ ] 启动时打印预计权重显存。
- [ ] 启动时打印预计 KV Cache 显存。
- [ ] Gemma4 未通过 eager parity 时默认关闭 compile/graph。

### 11.2 新增统一加载入口

建议新增：

```text
miniservellm/model_adapter/model_loader.py
```

统一完成：

- [ ] 本地/Hub 路径解析。
- [ ] AutoConfig 加载。
- [ ] Adapter 选择。
- [ ] tokenizer 加载。
- [ ] Ollama baseline 可用性和模型 tag 检查。
- [ ] 自研权重提取。
- [ ] 资源释放。
- [ ] Runner 创建。

### 11.3 日志与诊断

- [ ] 日志包含 `model_type`。
- [ ] 日志包含 checkpoint 路径。
- [ ] 日志包含 Transformers 版本。
- [ ] 日志包含每层 attention 类型计数。
- [ ] 日志包含 head dimension 分布。
- [ ] 日志包含 KV sharing 数量。
- [ ] 日志包含 PLE 权重大小。
- [ ] 日志包含最终 attention backend。
- [ ] 日志包含是否启用 quantization/compile/CUDA Graph。

---

## 12. 算子和 CUDA 优化整改清单

### 12.1 第一阶段只保证正确性

- [ ] Gemma4 默认使用 PyTorch eager/SDPA。
- [ ] 禁用未经验证的 QKV+RoPE fused kernel。
- [ ] 禁用未经验证的 CUDA Graph。
- [ ] 禁用未经验证的 paged attention kernel。
- [ ] 保留每个算子的 reference fallback。
- [ ] 每次 fallback 打印一次原因，不要每 token 重复打印。

### 12.2 RMSNorm

- [ ] 复用标准 RMSNorm 数学定义。
- [ ] 测试输入 fp16、bf16、fp32。
- [ ] 测试四个主 Norm。
- [ ] 测试 q/k/v Norm。
- [ ] 测试 V norm 无 scale 变体。
- [ ] 检查 fused add + norm 是否适用于 Gemma4 的每个位置。
- [ ] 不把 Qwen 两 Norm 的融合调用直接复制到 Gemma4。

### 12.3 MLP kernel

- [ ] 新增 `gelu_tanh_and_mul`。
- [ ] 明确 GELU approximate tanh 公式。
- [ ] 支持不同中间宽度。
- [ ] 检查 INT4 kernel 对每层 output shape 的要求。
- [ ] 没有 Gemma4 专用 fused MLP kernel 时走 PyTorch。

### 12.4 Attention kernel

- [ ] sliding 层可使用现有 kernel 前先验证 D=256。
- [ ] full 层 D=512 使用 SDPA/eager fallback。
- [ ] 验证 Q/K/V norm 后再进入 kernel。
- [ ] 验证 partial RoPE 后再进入 kernel。
- [ ] 验证 local mask 后再进入 kernel。
- [ ] 验证 KV source layer 映射后再进入 kernel。
- [ ] 后续再考虑单独实现 Gemma4 attention kernel。

### 12.5 CUDA Graph 和 torch.compile

- [ ] Gemma4 eager parity 通过前默认关闭。
- [ ] 解决每层异构 shape 对 graph capture 的影响。
- [ ] 解决 cache source mapping 对 static graph 的影响。
- [ ] 解决 sliding/full mask 的静态化。
- [ ] 解决 PLE layer slice 的 compile graph specialization。
- [ ] batch=1 固定 context 先做实验。
- [ ] graph replay 与 eager logits 对齐后再开放配置。

---

## 13. 显存和低显存整改清单

### 13.1 RTX 3060 6GB 的现实约束

Gemma4 E2B 总参数约 5.1B，FP16 权重理论大小约为：

```text
5.1e9 * 2 bytes ≈ 10.2 GB
```

这还没有计算 KV Cache、临时激活、CUDA runtime 和 workspace。因此：

- [ ] 不把 FP16 全量加载作为 3060 的默认方案。
- [ ] 不把“有效 2B”当成“只需 4GB 权重”。
- [ ] 单独统计 PLE embedding 的显存。
- [ ] 单独统计主 Transformer 权重的显存。
- [ ] 单独统计 lm_head 的显存。
- [ ] 单独统计 KV Cache。
- [ ] 单独统计 prefill 临时张量。

### 13.2 量化范围

当前 `TransformerModelRunner.quantize_weights()` 主要量化 Linear 权重，embedding、lm_head、Norm 保持浮点。对 Gemma4 这可能不够。

- [ ] 测量 PLE 表占用。
- [ ] 测量主 embedding 占用。
- [ ] 确认现有 INT4 是否支持 PLE embedding。
- [ ] 确认现有 INT4 是否支持巨大 vocab 的 lm_head。
- [ ] 评估 PLE 表的 INT8/INT4/分块量化。
- [ ] 评估 PLE CPU offload 或 mmap。
- [ ] 评估只将当前请求所需 PLE 行搬到 GPU。
- [ ] 评估主 Linear 使用 INT4、Norm 保持 fp16。
- [ ] 对量化前后 logits 做 parity。
- [ ] 对量化前后生成质量做最小回归。

### 13.3 KV Cache 预算

- [ ] 使用 layer-aware 公式计算每 token KV bytes。
- [ ] 分别测量 D=256 和 D=512 层。
- [ ] 测量独立 KV 与 shared KV 两种实现。
- [ ] 限制最大上下文长度。
- [ ] 限制最大 batch。
- [ ] 限制 prefill chunk size。
- [ ] 将 block size 作为可调参数。
- [ ] 记录不同 block size 的碎片浪费。
- [ ] OOM 前由调度器拒绝/等待请求，而不是运行中崩溃。

### 13.4 低显存验收

- [ ] 记录启动时权重显存。
- [ ] 记录首次 prefill 峰值显存。
- [ ] 记录 decode 峰值显存。
- [ ] 记录不同上下文长度的显存曲线。
- [ ] 记录 batch=1、batch=2、batch=4 的显存。
- [ ] 记录量化模式和 cache dtype。
- [ ] 在 3060 上至少完成一个可复现的文本推理配置。
- [ ] 如果无法在 6GB 完成，文档中明确列出所需 offload/量化，而不是宣称已支持。

---

## 14. 数值正确性测试清单

### 14.1 测试基准原则

- [ ] 使用现成 Ollama `gemma4:e2b` 作为端到端功能和性能 baseline。
- [ ] Ollama 请求默认使用模型原生 chat template，不混用 Qwen 模板。
- [ ] 数值测试使用项目内独立 PyTorch eager reference，不新增 Hugging Face 推理基线路径。
- [ ] 同一组原始 token IDs 输入 eager reference 和自研优化路径。
- [ ] eager reference 使用 FP32 或设备支持的高精度。
- [ ] 保存输入 token IDs、配置摘要和随机种子。
- [ ] 保存每层中间结果，便于定位第一处偏差。

### 14.2 配置和权重单元测试

- [ ] 外层 config 正确解析。
- [ ] 嵌套 `text_config` 正确解析。
- [ ] 35 层 layer spec 正确生成。
- [ ] full 层 head_dim=512。
- [ ] sliding 层 head_dim=256。
- [ ] PLE shape 正确。
- [ ] KV source mapping 正确。
- [ ] MLP width mapping 正确。
- [ ] q/k/v norm shape 正确。
- [ ] 四个主 Norm shape 正确。
- [ ] lm_head tie 状态正确。
- [ ] EOS 集合正确。

### 14.3 算子单元测试

- [ ] RMSNorm 与官方实现对齐。
- [ ] q_norm 对齐。
- [ ] k_norm 对齐。
- [ ] v_norm 对齐。
- [ ] GELU tanh 对齐。
- [ ] PLE token identity branch 对齐。
- [ ] PLE context branch 对齐。
- [ ] PLE combine scaling 对齐。
- [ ] PLE layer injection 对齐。
- [ ] sliding mask 对齐。
- [ ] full mask 对齐。
- [ ] default RoPE 对齐。
- [ ] proportional RoPE 对齐。
- [ ] partial rotary 对齐。
- [ ] final logit soft cap 对齐。

### 14.4 单层 parity

- [ ] 选取一个 sliding attention 层。
- [ ] 选取一个 full attention 层。
- [ ] 对比 Attention 前 hidden。
- [ ] 对比 Q/K/V projection。
- [ ] 对比 Q/K/V norm。
- [ ] 对比 RoPE 后 Q/K。
- [ ] 对比 attention score。
- [ ] 对比 attention output。
- [ ] 对比 O projection。
- [ ] 对比 FFN 前 hidden。
- [ ] 对比 MLP output。
- [ ] 对比 PLE gate。
- [ ] 对比 PLE injection。
- [ ] 对比层最终 hidden。

### 14.5 自研 eager 与优化路径的全模型 logits parity

- [ ] 空间较短的纯文本 prompt。
- [ ] prompt 长度 1。
- [ ] prompt 长度 2。
- [ ] prompt 长度跨越 sliding window 边界。
- [ ] prompt 长度超过 512。
- [ ] prompt 长度超过 1024。
- [ ] batch=1。
- [ ] batch>1。
- [ ] FP32 eager。
- [ ] FP16 eager。
- [ ] BF16 eager（设备支持时）。

建议记录：

```text
max_abs_error
mean_abs_error
relative_error
top1_match
topk_overlap
first_divergent_layer
first_divergent_token
```

初始验收阈值应先以项目内 FP32 eager reference 实测结果为基准，再固定到测试中。不要在没有 baseline 的情况下随意放宽误差。

### 14.6 Cache 等价性

- [ ] eager reference cache 与自研 decode 逐 token logits 对齐。
- [ ] eager reference 全量 forward 与自研 prefill 对齐。
- [ ] 一次性 prefill 与分 chunk prefill 对齐。
- [ ] prefill 后 decode 与完整串行 forward 对齐。
- [ ] sliding window 边界前后对齐。
- [ ] KV shared 与独立 reference 实现对齐。
- [ ] cache free 后新请求不读取旧数据。

### 14.7 生成 parity

- [ ] greedy 32 token 全部 token ID 一致。
- [ ] greedy 128 token 全部 token ID 一致或记录首个差异。
- [ ] temperature=0。
- [ ] temperature>0 且固定随机种子。
- [ ] top-k。
- [ ] top-p。
- [ ] repetition penalty。
- [ ] 多 EOS。
- [ ] max_new_tokens。
- [ ] chat template。

---

## 15. Qwen 回归测试清单

Gemma4 的改造不得破坏现有 Qwen2.5-0.5B。

- [ ] Qwen 配置转换测试仍通过。
- [ ] Qwen 权重提取测试仍通过。
- [ ] Qwen QKV shape 仍正确。
- [ ] Qwen gate/up 合并仍正确。
- [ ] Qwen RMSNorm parity 仍通过。
- [ ] Qwen RoPE parity 仍通过。
- [ ] Qwen prefill parity 仍通过。
- [ ] Qwen decode parity 仍通过。
- [ ] Qwen Paged KV Cache 测试仍通过。
- [ ] Qwen batch prefill 仍通过。
- [ ] Qwen batch decode 仍通过。
- [ ] Qwen CUDA kernel fallback 仍通过。
- [ ] Qwen CUDA Graph 路径仍通过。
- [ ] Qwen INT4 路径仍通过。
- [ ] Qwen 原有 demo 仍可运行。

---

## 16. 运行时功能测试清单

### 16.1 请求生命周期

- [ ] 单请求 fresh prefill。
- [ ] 长 prompt incremental prefill。
- [ ] prefill 转 decode。
- [ ] 首 token 是 EOS。
- [ ] decode 中间遇到 EOS。
- [ ] 达到 max_new_tokens。
- [ ] 多请求同时进入。
- [ ] 请求完成后释放全部 cache。
- [ ] 请求异常后释放全部 cache。

### 16.2 调度与批处理

- [ ] decode 优先策略不改变结果。
- [ ] prefill chunk 不改变结果。
- [ ] KV block 不足时请求等待。
- [ ] KV block 释放后等待请求继续。
- [ ] 不同长度请求混批。
- [ ] fresh 和 incremental prefill 混合。
- [ ] prefill 和 decode 同 step。
- [ ] batch bucket 开启/关闭结果一致。
- [ ] context bucket 开启/关闭结果一致。

### 16.3 输入边界

- [ ] 空字符串被明确拒绝或按约定处理。
- [ ] 单 token 输入。
- [ ] 长于 512 的输入。
- [ ] 长于 1024 的输入。
- [ ] 包含中文、英文、数字和特殊符号。
- [ ] 包含 system/user/assistant 多轮消息。
- [ ] 明确拒绝图像/音频参数。
- [ ] 超出最大上下文时给出明确错误。

---

## 17. 性能整改顺序

性能工作必须在 eager 数值正确之后进行，顺序不能倒置。

### P0：正确性基线

- [ ] Ollama `gemma4:e2b` 端到端功能和性能 baseline。
- [ ] 项目内 PyTorch eager 数值 reference。
- [ ] Gemma4 Adapter。
- [ ] PLE。
- [ ] 四个主 Norm。
- [ ] Q/K/V Norm。
- [ ] MLP。
- [ ] 双 RoPE。
- [ ] sliding/full attention。
- [ ] final soft cap。
- [ ] 普通 eager cache。

### P1：服务化

- [ ] 单请求 prefill/decode 接入引擎。
- [ ] EOS 和 chat template。
- [ ] request lifecycle。
- [ ] 基础 batch。
- [ ] 基础调度。

### P2：Cache 优化

- [ ] layer-aware cache。
- [ ] KV sharing。
- [ ] sliding cache/mask。
- [ ] paged storage。
- [ ] batch gather。

### P3：低显存

- [ ] Linear INT4。
- [ ] PLE 量化或 offload。
- [ ] embedding/lm_head 策略。
- [ ] KV cache dtype。
- [ ] block 数自动调优。

### P4：CUDA 优化

- [ ] D=256 sliding attention kernel。
- [ ] D=512 full attention fallback 或 kernel。
- [ ] GELU fused kernel。
- [ ] QKV + Norm + RoPE 融合。
- [ ] PLE injection 融合。
- [ ] paged attention。

### P5：图优化

- [ ] torch.compile eager parity。
- [ ] fixed-shape decode graph。
- [ ] dynamic context graph。
- [ ] batch bucket graph。
- [ ] CUDA Graph replay parity。

---

## 18. 涉及文件清单

### 18.1 新增文件建议

- [ ] `miniservellm/model_adapter/adapters/gemma4_adapter.py`
- [ ] `miniservellm/model_adapter/adapter_factory.py`
- [ ] `miniservellm/model_adapter/config_validation.py`
- [ ] `miniservellm/runtime/gemma4_text_runner.py`
- [ ] `miniservellm/runtime/gemma4_ops.py`（如果不希望继续扩大 `nn_ops.py`）
- [ ] `scripts/inspect_gemma4_text_config.py`
- [ ] `scripts/check_gemma4_hf_parity.py`
- [ ] `scripts/bench_gemma4_text.py`
- [ ] `tests/test_gemma4_config.py`
- [ ] `tests/test_gemma4_adapter.py`
- [ ] `tests/test_gemma4_ops.py`
- [ ] `tests/test_gemma4_ple.py`
- [ ] `tests/test_gemma4_cache.py`
- [ ] `tests/test_gemma4_parity.py`

### 18.2 需要修改的现有文件

- [ ] `miniservellm/config.py`
- [ ] `miniservellm/model_adapter/adapter_interface.py`
- [ ] `miniservellm/model_adapter/hf_loader.py`
- [ ] `miniservellm/model_adapter/tokenizer_adapter.py`
- [ ] `miniservellm/runtime/model_runner.py`（仅在选择共享抽象时修改）
- [ ] `miniservellm/runtime/nn_ops.py`
- [ ] `miniservellm/runtime/inference_engine.py`
- [ ] `miniservellm/runtime/runner.py`
- [ ] `miniservellm/runtime/attention_metadata.py`
- [ ] `miniservellm/runtime/prefill.py`
- [ ] `miniservellm/runtime/incremental_prefill.py`
- [ ] `miniservellm/runtime/decode.py`
- [ ] `miniservellm/cache/kv_cache.py`
- [ ] `miniservellm/cache/kv_cache_manager.py`（如果仍被使用）
- [ ] `miniservellm/cache/block_allocator.py`
- [ ] `miniservellm/runtime/cuda_graph_runner.py`
- [ ] `scripts/run_stage5_demo.py`
- [ ] `scripts/inspect_model.py`

### 18.3 第一阶段明确不修改或不接入

- [ ] `miniservellm/mlx/qwen2.py` 不在 CUDA 纯文本第一阶段改造。
- [ ] Vision/Audio encoder 不接入。
- [ ] MoE 路径不接入。
- [ ] 现有 Qwen 权重格式不改成 Gemma 专用格式。

---

## 19. 实施阶段与交付物

### 阶段 A：基线和模型事实确认

- [ ] 固定 Transformers 版本。
- [ ] 准备本地 Gemma4 E2B checkpoint。
- [ ] 运行模型配置检查脚本。
- [ ] 输出 state dict manifest。
- [ ] 保存 Ollama 固定短 prompt 的 greedy 输出和 token/耗时指标。
- [ ] 保存 Ollama batch=1 的 prefill/decode 性能数据。
- [ ] 记录权重、PLE 和 KV 的显存预算。

阶段出口：

```text
配置、模块路径、权重 shape、层级 forward 事实均已确认；
没有依赖猜测实现。
```

### 阶段 B：Adapter 和权重层

- [ ] 完成 Gemma4 Adapter。
- [ ] 完成 Adapter factory。
- [ ] 完成 config normalization。
- [ ] 完成权重提取。
- [ ] 完成权重 shape 校验。
- [ ] 完成 CPU staging。
- [ ] 完成 PLE 权重 manifest。

阶段出口：

```text
可以加载 checkpoint，并生成完整、可校验的 Gemma4TextWeights；
不执行自研前向。
```

### 阶段 C：eager 单请求 Runner

- [ ] 完成 PLE。
- [ ] 完成四个主 Norm。
- [ ] 完成 q/k/v norm。
- [ ] 完成双 RoPE。
- [ ] 完成 sliding/full attention。
- [ ] 完成异构 head_dim。
- [ ] 完成 MLP。
- [ ] 完成 final soft cap。
- [ ] 完成单请求 prefill。
- [ ] 完成单 token decode。

阶段出口：

```text
短 prompt 的端到端 greedy 输出与 Ollama baseline 对照完成，
单 token decode 与项目内 eager reference 数值对齐；
暂不承诺 Paged KV Cache、batch 和 CUDA Graph。
```

### 阶段 D：reference KV Cache

- [ ] 实现非 paged 的 layer-aware KV state。
- [ ] 实现 KV source mapping。
- [ ] 实现 sliding mask。
- [ ] 实现 full cache。
- [ ] 完成 prefill/decode cache 等价性。
- [ ] 完成长 prompt 测试。

阶段出口：

```text
一次性 prefill、chunked prefill、逐 token decode 的结果一致。
```

### 阶段 E：接入现有服务引擎

- [ ] 接入 `FreshPrefillRunner`。
- [ ] 接入 `IncrementalPrefillRunner`。
- [ ] 接入 `DecodeRunner`。
- [ ] 接入 Scheduler。
- [ ] 接入多 EOS。
- [ ] 接入 Gemma4 chat template。
- [ ] 完成 batch isolation。
- [ ] 完成请求释放。

阶段出口：

```text
Gemma4 可以通过统一 Stage5Engine 处理纯文本请求；
Qwen 回归测试全部通过。
```

### 阶段 F：Paged KV Cache

- [ ] 选择每层独立 cache 或 cache group 方案。
- [ ] 修改 block allocation。
- [ ] 修改 write/read/gather API。
- [ ] 修改 prefill attention。
- [ ] 修改 decode attention。
- [ ] 修改 batch metadata。
- [ ] 完成 paged/reference parity。

阶段出口：

```text
Paged 版本和 reference cache 的 logits、生成 token、停止原因一致。
```

### 阶段 G：低显存和量化

- [ ] 完成 Linear INT4 parity。
- [ ] 评估 PLE 量化/offload。
- [ ] 评估 embedding/lm_head。
- [ ] 评估 KV cache dtype。
- [ ] 在 RTX 3060 6GB 上确定可运行配置。
- [ ] 记录上下文长度和并发上限。

阶段出口：

```text
有一套可复现的 3060 运行配置，显存峰值、速度和精度都有记录。
```

### 阶段 H：性能优化

- [ ] sliding attention kernel。
- [ ] full attention D=512 backend。
- [ ] GELU/PLE 融合。
- [ ] paged attention kernel。
- [ ] torch.compile。
- [ ] CUDA Graph。
- [ ] batch=1 decode benchmark。
- [ ] 多请求 throughput benchmark。

阶段出口：

```text
每一项优化都有 eager/reference 对照和独立 benchmark，
任何 kernel fallback 都有日志和回归测试。
```

---

## 20. 验收标准（Definition of Done）

### 功能验收

- [ ] `google/gemma-4-E2B-it` 可以加载。
- [ ] 纯文本 prompt 可以完成生成。
- [ ] chat template 正确。
- [ ] raw token IDs 可以完成生成。
- [ ] 多 EOS 正确停止。
- [ ] max_new_tokens 正确停止。
- [ ] 请求结束后 cache 正确释放。
- [ ] 不传多模态输入时不会误触发 vision/audio 分支。

### 数值验收

- [ ] FP32 eager 与自研优化路径 logits 在固定阈值内。
- [ ] sliding 层和 full 层均完成单层 parity。
- [ ] prefill 和 decode 均完成 parity。
- [ ] chunked prefill 与一次性 prefill 一致。
- [ ] greedy 生成至少 32 token，并记录与 Ollama Q4 baseline 的输出差异。
- [ ] 量化版本的误差和生成差异有明确记录。

### 回归验收

- [ ] Qwen2.5-0.5B 原有测试全部通过。
- [ ] Qwen 原有 demo 可运行。
- [ ] Qwen CUDA kernel 路径不回归。
- [ ] Qwen INT4 路径不回归。
- [ ] Qwen CUDA Graph 路径不回归。

### 资源验收

- [ ] 启动时输出权重显存预算。
- [ ] 输出 PLE 显存预算。
- [ ] 输出 KV Cache 预算。
- [ ] 3060 上没有未说明的 OOM。
- [ ] 记录最大可用上下文。
- [ ] 记录 batch=1 decode tok/s。
- [ ] 记录 prefill TTFT。
- [ ] 记录 batch throughput。

### 工程验收

- [ ] Gemma4 与 Qwen 适配代码边界清晰。
- [ ] 不存在模型名硬编码导致的错误选择。
- [ ] 不存在静默 fallback 到 Qwen 结构。
- [ ] 不支持的结构会给出明确错误。
- [ ] 关键权重 shape 有启动校验。
- [ ] 关键算子有 reference fallback。
- [ ] 文档记录实际 Transformers 版本和 checkpoint 版本。
- [ ] 文档记录已完成项、未完成项和已知限制。

---

## 21. 风险、阻塞项和处理策略

### 风险 1：Transformers 版本不支持当前 Gemma4 checkpoint

- [ ] 在实施前固定并验证 Transformers 版本。
- [ ] 检查 `Gemma4TextModel`、`Gemma4ForCausalLM` 或当前对应类是否存在。
- [ ] 检查 `AutoConfig` 是否能解析 `gemma4`。
- [ ] 不通过 `trust_remote_code=True` 盲目绕过版本问题。
- [ ] 如果必须使用源码版本，记录 commit/版本号。

### 风险 2：PLE 权重导致 3060 无法加载

- [ ] 先计算 PLE 表单独显存。
- [ ] 验证现有 Linear-only INT4 是否足够。
- [ ] 设计 PLE 量化或按 token 搬运方案。
- [ ] 在未解决前，不宣称“Gemma4 已支持 3060”。

### 风险 3：full attention 的 head_dim=512 不被 CUDA kernel 支持

- [ ] full 层默认使用 SDPA/eager。
- [ ] 单独 benchmark full 层。
- [ ] 检查 RTX 3060 上的 SDPA 性能和显存。
- [ ] 后续再写 D=512 kernel。

### 风险 4：KV sharing 规则理解错误

- [ ] 以官方 Transformers forward 和 cache 实现为准。
- [ ] 从实际 config 生成 source mapping。
- [ ] 编写两层最小模型/人工配置测试。
- [ ] 禁止仅凭 `num_kv_shared_layers` 数字猜映射。

### 风险 5：把 E2B 当作普通 2B 模型

- [ ] 文档和日志使用 E2B/有效参数的准确表述。
- [ ] 权重显存按总参数计算。
- [ ] PLE 不能省略。
- [ ] 不能用普通 Qwen2.5 的显存自动调优公式。

### 风险 6：优化路径与 eager 路径不一致

- [ ] 每个 kernel 都有 reference 对比。
- [ ] 优化前保存 parity 基线。
- [ ] 优化后逐层检查。
- [ ] CUDA Graph 只在固定 shape 和 cache 语义稳定后开启。

---

## 22. 推荐的最小可行版本（MVP）

如果需要控制首轮工作量，MVP 应限定为：

- [ ] `google/gemma-4-E2B-it`。
- [ ] 纯文本。
- [ ] 单请求。
- [ ] PyTorch eager/SDPA。
- [ ] FP32 reference + FP16 推理。
- [ ] PLE。
- [ ] 四个主 Norm。
- [ ] Q/K/V Norm。
- [ ] sliding/full attention。
- [ ] 双 RoPE。
- [ ] 异构 head dimension。
- [ ] KV sharing 的 reference 实现。
- [ ] GELU-tanh gated MLP。
- [ ] final logit soft cap。
- [ ] HF logits parity。
- [ ] greedy token parity。

MVP 暂不包含：

- [ ] Paged KV Cache。
- [ ] Continuous batching。
- [ ] CUDA Graph。
- [ ] 自定义 attention kernel。
- [ ] PLE INT4。
- [ ] 多模态。

MVP 通过后，再逐项将 reference 实现替换为当前高性能运行时实现。

---

## 23. 实施记录

### 已完成

- [ ] 

### 进行中

- [ ] 

### 待处理

- [ ] 

### 已知限制

- [ ] 

### 关键 benchmark 记录

```text
模型：
设备：
dtype：
量化：
上下文长度：
batch：
prefill TTFT：
decode tok/s：
峰值显存：
HF parity：
备注：
```
