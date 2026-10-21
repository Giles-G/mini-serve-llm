# A/B 缺口完成报告

日期：2026-10-20  
设备：RTX 3060 Laptop 6GB，sm_86；torch 2.13.0+cu130；INT4 group_size=64，FP16  
状态：**A/B 缺口已补齐；vectorized row-pair INT4 GEMV（pair_vec）设为默认，graph 中位数 101–102 tok/s（128 token）**

## A. profiling / 输出 / 元数据

### 已完成

- `--profile` / `--profile-summary` 包含：
  - `start_context_len`
  - `expected_decode_steps`、`actual_decode_steps`、`steps_match`
  - `record_shapes`（`--profile-shapes` 控制）
  - `categories`：int4_kernel、int4_label、attention_kernel、attention_math、norm、copy、sampler、elementwise、engine_label、other
  - 每类 `self_cpu_ms_per_step` / `self_cuda_ms_per_step` / calls
  - `hotspots`：每个 kernel 的 calls、self CUDA、us/call、ms/step
  - `parent_label_note`：明确 `int4::`/`engine::`/`direct::` label 不计入 raw kernel 求和
- **direct 与 engine 都支持 profile**：新增 `--profile-mode auto|direct|engine`。`--mode both` + `auto` 会在同一进程、同一权重上分别 profile direct eager 和 engine，输出 `backends`、`direct`、`comparison`。
  - `comparison` 只比较 raw kernel categories/hotspots，自动排除父 label，避免重复计时。
  - 两份产物：`reports/gemma4_profile_compare.json`（默认 graph engine）与 `reports/gemma4_profile_compare_eager.json`（`MINI_LLM_GEMMA4_GRAPH=0`，paged eager vs dense eager，更适合逐 kernel 对照）。
  - 3 个 profiled decode step 下，paged eager engine 的 raw kernel 归因示例：INT4 12.18 ms/step、attention 1.15、elementwise 3.46、norm 1.26、copy 0.58；direct eager：INT4 6.99、attention_math 0.35、copy 0.92、norm 0.76、elementwise 2.45。两者 launch 数和融合方式不同，这组数字用于归因，不代表 wall-clock 比值。
- **逐事件 shape 落盘**：`--profile-shapes` 现在写 `event_shapes`（按 name+shape 聚合，含 calls/self CUDA/self CPU）。engine profile 采到 27 组，direct profile 采到 40 组。
  - 边界：PyTorch 的 `record_shapes` 只给 CPU aten op 记录 `input_shapes`，CUDA kernel event 不携带 shape；因此 `event_shapes` 是 op 级形状，真实 INT4 projection 尺寸仍以 `metadata.projection_shapes` 的 317 项为准。summary 中已写 `event_shapes_note`。
- `--output-json` 包含 device/capability/VRAM、torch/CUDA/driver、git HEAD+dirty、EOS、INT4 backend/attention 状态、weights、317 个真实 projection shapes、prompt tokens/chat template、完整结果与 token IDs、`int4_matmul_calls`。
- direct 路径有 `finish_reason`（EOS / MAX_NEW_TOKENS）。
- 空可见文本有 `visible_text_empty`、`generation_status="empty_text"`，默认退出码 1；`--allow-empty-generation` 可关闭。
- **tokenizer / 后处理计时补齐**：
  - `metadata.text_tokenize_s`：chat template 之后的 tokenizer.encode。
  - 每个结果 `text_decode_s`：生成 token → 可见文本 + raw 文本的两次 decode。
  - 每个结果 `text_postprocess_s`：`strip()`/空文本判定。
  - `text_print_s`：输出生成文本的打印耗时。
  - 以上均不计入 decode 计时。
- `runner_backend` 记录 graph_mode/graph_captured/paged_attention，避免“实际走了哪个后端”不明。

### A 仍保留的边界

- `event_shapes` 只有 CPU aten op 的输入形状；CUDA kernel 级逐事件 shape 不受 PyTorch profiler 支持。真实 projection shape 由 metadata 的 317 项覆盖。
- direct/engine profile 是两次独立运行，per-step self time 受 profiler 开销、GPU 时钟和 launch 结构影响；`comparison` 明确标注为归因对照，不是 wall-clock 结论。
- tokenizer/后处理已单独计时，但没有把 tokenizer 放到独立进程或 GPU 侧。

## B. INT4 GEMV

### 已完成

- 保留 legacy（thread-per-row）、warp（warp-per-row）、pair（row-pair）三种实现。
- **新增 pair_vec（vectorized row-pair）**：
  - 一个 warp 仍计算相邻两行 (2p, 2p+1)，共享 packed byte 的 low/high nibble。
  - 每个 lane 用一次对齐的 32-bit load 读入 **4 个 packed 列**；16 个 lane 覆盖一个 64 列 group，warp 一次覆盖 2 个 group。
  - 相比 scalar pair 的每 lane 两次 8-bit load，weight load 指令约减少 4×；byte traffic 不变。
  - 全 warp 做 FP32 accumulation + shuffle reduction，每个 row 使用独立 scale。
- `MINI_LLM_INT4_IMPL` 支持 `legacy | warp | pair | pair_vec`；默认 **pair_vec**。
- pair_vec 对旧扩展兼容：若安装的 `_C` 没有 pair_vec binding，自动回退到 scalar pair，而不是禁用整个 CUDA INT4。
- pair_vec vs pair/legacy/oracle 边界测试：零输入、抵消、极值、混合 scales、N=2/6/130/258、K=64/128/1536/4096、多 token、FP16/BF16、空 batch、非默认 stream。server 测试 **73 passed**，kernels 测试 **456 passed**。
- **寄存器/occupancy 计数**（不依赖 Nsight，使用 `cudaFuncGetAttributes` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor`）：
  - pair：38 regs/thread；pair_vec：37 regs/thread。
  - 两者均 12 blocks/SM、1536 active threads/SM（RTX 3060 上限 2048，约 75% occupancy）。
  - 0 shared memory、0 local memory；sm_86 binary/PTX 86。
- `bench_gemma4_cuda.py`：
  - `--int4-impl legacy|warp|pair|pair_vec`
  - `--rounds`、逻辑权重带宽、GPU 时钟/温度/功耗/显存、failure_ratio、P99、exact/rounded 合同、kernel_attributes。
  - `--seeds` 支持多 seed；`--limit` 改为 **per-seed**，保证多 seed 覆盖同一组 projection。
- 真实 checkpoint 12 个代表 shape，seed 20261001，pair_vec vs pair（3 轮中位数）：
  - lm_head [262144,1536]：**1.028 vs 1.036 ms**（单独隔离 3 轮为 **1.003 vs 1.061 ms**），约 208 vs 206 GB/s。
  - PLE [8960,1536]：0.034 vs 0.038 ms。
  - q_proj [2048,1536]：0.015 vs 0.017 ms。
  - gate_proj [6144,1536]：0.024 vs 0.026 ms；up_proj：0.024 vs 0.027 ms。
  - down_proj [1536,6144]：0.027 vs 0.026 ms（pair_vec 略慢）。
  - k/v_proj [256,1536]：0.015 vs 0.016 ms。
- 多 seed 真实 shape：3 个 seed × 12 shape = **36 个 case**，pair_vec 全部通过，`max_abs=0.00781`，无 failure。
- 完整模型：
  - 128 token：**9.69 ms/token、103.14 tok/s**（decode_s 1.231 / 127）。
  - 256 token：**11.16 ms/token、89.60 tok/s**；同条件 scalar pair 为 89.25 tok/s，且 **256 个 token ID 完全一致**。
  - 正式验收（3 prompt × 3 轮 + 256 token）：pair_vec zh/en/math/long = **9.81 / 9.89 / 10.00 / 11.39 ms/token**；同条件 scalar pair = 9.88 / 10.02 / 10.13 / 11.62 ms/token。pair_vec 在四个 case 都略优，但差距约 0.4–2%。
  - 之后 P0/P1 的 split-K、快速反量化、pair_vec8、graph 预 warm 使最终验收变为 **7.15 / 7.19 / 6.82 / 7.45 ms/token**，prefill 0.060–0.069 s；见 `reports/gemma4_p1_report.md`。

### B 仍保留的边界

- 向量化收益集中在 lm_head 和中等 N 的 projection；k/v 等小 shape 主要受 launch/固定开销限制，差异落在噪声内，down_proj 本轮甚至略慢。
- x 激活仍是 4 次标量 half load，没有做 `uint2`/half2 向量 load；lm_head 约 1.0 ms，仍有继续优化空间。
- 没有 Nsight 的硬件计数器（DRAM sector、stall reason 等）；已用 CUDA 官方 attributes/occupancy 补上寄存器与 occupancy 计数。
- 多 seed 覆盖的是真实权重 + 不同随机激活，不是多个权重 seed。

## 验证命令

```bash
# A: direct + engine 双后端 profile（graph 默认）
./venv/bin/python scripts/bench_gemma4_speed.py \
  --model models/gemma-4-E2B-it --mode both --int4 \
  --max-new-tokens 2 --warmup 2 \
  --profile-tokens 5 --profile-shapes --profile-mode auto \
  --profile-summary reports/gemma4_profile_compare.json \
  --output-json /tmp/a_results.json

# A: paged eager vs direct eager（关闭 graph）
MINI_LLM_GEMMA4_GRAPH=0 ./venv/bin/python scripts/bench_gemma4_speed.py \
  --model models/gemma-4-E2B-it --mode both --int4 \
  --max-new-tokens 2 --warmup 0 \
  --profile-tokens 5 --profile-shapes --profile-mode auto \
  --profile-summary reports/gemma4_profile_compare_eager.json

# B: 真实 shape pair_vec vs pair，3 seed
./venv/bin/python scripts/bench_gemma4_cuda.py \
  --model models/gemma-4-E2B-it --limit 12 --skip-lm-head \
  --seeds 20261001 20261002 20261003 \
  --rounds 2 --iters 10 --skip-fallback --int4-impl pair_vec
./venv/bin/python scripts/bench_gemma4_cuda.py \
  --model models/gemma-4-E2B-it --limit 12 --skip-lm-head \
  --seeds 20261001 20261002 20261003 \
  --rounds 2 --iters 10 --skip-fallback --int4-impl pair

# 正式验收
./venv/bin/python scripts/bench_gemma4_acceptance.py \
  --model models/gemma-4-E2B-it --rounds 3 --warmup 8 \
  --output-json reports/gemma4_acceptance.json

# 测试
./venv/bin/python -m pytest tests --ignore=tests/test_gemma4_mlx.py
/home/giles/work/mini-serve-llm/venv/bin/python -m pytest \
  /home/giles/work/mini-llm-kernels/tests
