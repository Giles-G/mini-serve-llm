# Gemma4 C 阶段覆盖报告

日期：2026-10-20  
设备：NVIDIA RTX 3060 Laptop GPU 6GB（sm_86，WSL2）  
模型：`models/gemma-4-E2B-it`，INT4 group_size=64，Engine/eager 路径  
状态：**C PARITY PASS（FP16 和 BF16）**

## 1. 低层数值覆盖

### 1.1 自动测试

- kernels 全部测试：**644 passed**（含 INT4 边界、pair_vec8、pair_batch、fast dequant、safe/warp-split/全局 split-K，context 到 4096）。
- server 测试：**75 passed**（`tests/test_gemma4_mlx.py` 因未安装 `mlx` 跳过）。

边界用例参数化覆盖：

- 输入模式：zero、正负抵消、大值、小值、混合 scales；
- `N`：2、6、130、258（覆盖 warp 4 行/块、pair 偶数行对与 legacy 128 行/块的尾部）；
- `K`：64、128、1536；
- token 数：1、19；
- dtype：FP16、BF16；
- legacy、warp、pair、pair_vec 与 CPU FP32 oracle 对比；
- pair/pair_vec 空 batch、非默认 stream；
- group_size != 64 的 CPU/CUDA dispatch fallback；旧扩展缺少 pair_vec binding 时自动回退 scalar pair；
- 默认 dispatch 已是 pair_vec，`MINI_LLM_INT4_IMPL=legacy|warp|pair|pair_vec` 可选。
- pair_vec 额外通过 12 个真实 shape 的 exact/rounded 合同、3 seed × 12 shape 的 benchmark，以及 256-token 完整模型输出（与 scalar pair 逐 token 一致）。
- 本轮补做的 C 证据集中在低层边界与完整模型一致性；`check_gemma4_int4_parity.py` 的 277-label 真实激活逐项对比仍以 legacy/warp/pair 为主，pair_vec 的激活级逐 label 表未单独重跑。

### 1.2 合成与真实 shape

- `bench_gemma4_cuda.py --edge-cases --seeds 20261001 20261002`：**34 cases passed**（FP16），BF16 edge suite 17 cases passed。
- 真实 checkpoint，自动读取 shape，`--limit 12`：**12 cases passed**。覆盖：
  - `lm_head [262144,1536]`
  - `per_layer_model_projection [8960,1536]`（PLE）
  - `layer0.q/k/v/o_proj`、`gate/up/down_proj`
  - `layer0.per_layer_input_gate/projection`
  - `layer1.q_proj`
- 参考合同：
  - exact：去量化权重，FP32 matmul，TF32 显式关闭；
  - rounded：权重先舍入到 activation dtype，再 FP32 matmul；
  - fallback 独立检查；失败会 fail case，不静默通过；
  - 小矩阵额外用 CPU FP64 校验参考本身；
  - 报告 absolute/relative/P99/failure_ratio/NaN-Inf/worst/worst-over-budget；
  - TF32 设置在 `_tf32_disabled()` 中恢复（已有单测）。
- fallback 修复：`_int4_dequant_matmul_pytorch` 的 rounded-weight 合同改为 FP32 matmul 后再 cast，解决 BF16 fallback 在 `per_layer_model_projection` 上 107/170240 元素超预算的问题。

## 2. 真实激活与完整模型 parity

工具：`scripts/check_gemma4_int4_parity.py`

- 对每个真实 prompt 捕获进入每个 INT4 projection 的实际激活；
- 277 个 projection labels，覆盖 q/k/v/o、MLP、PLE、`lm_head`；
- 对比 legacy、warp、pair 与 chunked rounded reference；
- 对选定 label 同时检查 exact 和 rounded 两个 FP32 合同；
- 对比 prefill + 连续 decode 的完整 logits、top-1/top-2、margin；
- 对比 KV cache、共享层存储、位置、synthetic sliding、官方 512 sliding（32 logits）和 EOS/stop。

### 2.1 FP16

命令参数：`--capture-steps 2 --logits-steps 8 --official-sliding --official-sliding-tokens 520 --official-sliding-steps 32`

| prompt | tokens | warp flips | pair flips | max logit diff | worst impl activation diff | contracts |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 中文 KV Cache | 19 | 0 | 0 | 0.0913 | 0.0625 | pass |
| 英文 KV Cache | 22 | 0 | 0 | 0.1089 | 0.0313 | pass |
| 算术 27+45 | 25 | 0 | 0 | 0.0665 | 0.0625 | pass |

- 每个 prompt 554 条真实激活记录；top-2 全部一致；pair 与 warp 输出同源。
- KV sharing：层 0–14 存储，共享层 15–34 不存储；`shared_layers_stored=[]`。
- KV cache 的 legacy-vs-warp 最大差：K 0.0018，V 0.0391（短上下文）。
- synthetic sliding window=8：0 flips，max logit diff 0.0827。
- **官方 sliding window=512，520 tokens，32 个 logits 位置**：warp/pair 均 **0 flips**，max logit diff **0.1429**，cache K diff 0.0713。
- EOS/stop：算术 prompt 在两种实现下都在 step 2 停止；raw `"Hello"` 在 4 步内未触发 EOS（legacy/warp/pair 一致）。
- `C PARITY PASS`。

### 2.2 BF16

命令参数：`--capture-steps 2 --logits-steps 8 --official-sliding --official-sliding-tokens 520 --official-sliding-steps 32 --max-logit-diff 1.0 --max-activation-diff 0.5`

| prompt | tokens | warp flips | pair flips | max logit diff | worst impl activation diff | contracts |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 中文 KV Cache | 19 | 0 | 0 | 0.5621 | 0.2500 (lm_head) | pass |
| 英文 KV Cache | 22 | 0 | 0 | 0.7528 | 0.2500 (lm_head) | pass |
| 算术 27+45 | 25 | 0 | 0 | 0.5848 | 0.2500 (lm_head) | pass |

- 277 labels × 2 个激活 step；top-2 一致；pair 与 warp 输出同源。
- KV sharing 同样正确（共享层不存储）。
- synthetic sliding window=8：0 flips，0.7954。
- 官方 sliding window=512，520 tokens，32 logits：warp/pair 均 0 flips，0.7369；cache K diff 0.2656。
- BF16 的 logit/activation 门槛比 FP16 宽，这是 BF16 精度本身决定的；报告保留实测值和门槛。
- `C PARITY PASS`。

### 2.3 完整模型输出

- 128-token Engine 请求（同一 prompt）：legacy、warp、pair 的生成 token ID 和文本完全一致，`finish_reason=MAX_NEW_TOKENS`。
- 默认路径现在是 row-pair；`MINI_LLM_INT4_IMPL=legacy|warp` 仍可回退。
- 完整文本在 `/tmp/gemma4_audit/graph_pair128.json`；256-token pair 输出与 warp/eager 一致。

## 3. 计划 C 检查项对照

| 计划项 | 覆盖情况 |
| --- | --- |
| 两种数值合同 | 已覆盖（exact / rounded，报告 `reference_contracts`） |
| TF32 禁用并恢复 | 已覆盖（`_tf32_disabled` + 单测） |
| 小矩阵 CPU FP64 | 已覆盖 |
| 分组累加/归约/输出舍入预算 | 已覆盖（allowance + failure_ratio） |
| fallback 独立失败判定 | 已覆盖；BF16 fallback 修复后通过 |
| absolute/relative/P99/failure/NaN/worst | 已覆盖 |
| 多 seed、零输入、抵消、极值、scales | 已覆盖（边界测试 + 2 seeds） |
| N 尾部、不同 K、多 token、unsupported dispatch | 已覆盖 |
| FP16 必测 | 已覆盖 |
| BF16 单独验证 | 已覆盖 |
| 真实 checkpoint q/k/v/o/MLP/PLE/lm_head | 已覆盖（真实 shape + 277 labels） |
| 真实 prompt 激活 | 已覆盖 |
| prefill + 连续 decode logits/top-1/top-2/margin | 已覆盖（FP16 8 步，BF16 8 步） |
| KV sharing、位置、sliding、EOS/stop | 已覆盖（含官方 512 window × 32 logits） |
| paged attention kernel 长上下文 | 已覆盖（512/1024，D=256/512，FP16/BF16，window 0/512） |
| 低层 + 真实激活 + 完整模型输出 | 已覆盖 |

## 4. 仍未覆盖/需要说明的边界

- 真实激活对比使用一份权重 seed；多 seed 覆盖在低层/边界测试中。
- 官方 sliding 是 520 prompt + 32 logits（上下文到约 552），没有做 2k/4k 超长上下文生成。
- paged attention kernel 直接 parity 到 1024；没有 2k/4k，也没有 paged attention 的 Nsight 计数器。
- `EOS` 的 raw `"Hello"` 在 4 步内未触发；算术 prompt 的 EOS 已覆盖两种实现。
- 报告的 JSON 在 `/tmp/gemma4_audit/c_parity_fp16_boundary.json`、`c_parity_bf16_boundary.json`；本报告只保留摘要。

## 5. 复现命令

```bash
# 低层测试
./venv/bin/python -m pytest tests --ignore=tests/test_gemma4_mlx.py
/home/giles/work/mini-serve-llm/venv/bin/python -m pytest \
  /home/giles/work/mini-llm-kernels/tests

# 边界 + 真实 shape
./venv/bin/python scripts/bench_gemma4_cuda.py --edge-cases \
  --seeds 20261001 20261002 --iters 10 --skip-lm-head --skip-fallback
./venv/bin/python scripts/bench_gemma4_cuda.py --model models/gemma-4-E2B-it \
  --limit 12 --skip-fallback --chunk-rows 4096 --reference-chunk-rows 4096

# 真实激活 / logits / sliding / EOS
./venv/bin/python scripts/check_gemma4_int4_parity.py \
  --model models/gemma-4-E2B-it --dtype fp16 --capture-steps 2 --logits-steps 8 \
  --official-sliding --official-sliding-tokens 520 --official-sliding-steps 32 \
  --output-json /tmp/c_parity_fp16_boundary.json
./venv/bin/python scripts/check_gemma4_int4_parity.py \
  --model models/gemma-4-E2B-it --dtype bf16 --capture-steps 2 --logits-steps 8 \
  --official-sliding --official-sliding-tokens 520 --official-sliding-steps 32 \
  --max-logit-diff 1.0 --max-activation-diff 0.5 \
  --output-json /tmp/c_parity_bf16_boundary.json
```

注意：C 工具现在同时跑 legacy、warp、pair；报告里 `warp_flips`/`pair_flips` 均需为 0。
