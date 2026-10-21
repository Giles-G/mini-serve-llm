# Gemma4 RTX 3060 推理性能与正确性整改计划

## 1. 目标与范围

优化 mini-serve-llm 与 mini-llm-kernels 在 NVIDIA RTX 3060 Laptop GPU 6GB 上运行 Gemma4 E2B 的性能，重点是 batch=1 decode。

本计划不混用 Apple M4 的 MLX/Metal 实现或性能结论，不以放宽数值阈值、遗漏同步或截断输出制造性能提升。

最终目标：

- 降低当前约 80.3 ms/token 的完整 Engine decode 延迟。
- 在正确性得到验证的前提下，逐步缩小与 Ollama 的系统级性能差距。
- 每次完整模型测试输出一次完整的实际生成结果，让用户可以直接观察回答内容。
- 提供体积小、可复制的性能摘要，默认不生成数百 MB 的 trace。
- 各阶段保留可回归基线，不提前承诺缺乏实测支持的 token/s。

本文是实施与验收计划；条目未完成前不能视为已经实现或已在 3060 验证。

### 当前工作区实施状态（2026-10-20）

- 已完成：正式请求完整文本/全部 token ID 输出、空文本 raw 解码诊断、结果 JSON 保存、direct/engine EOS 配置统一。
- 已完成：独立 decode-only profile、CPU/CUDA 小摘要、显式可选 trace、真实 checkpoint 投影尺寸报告。
- 已完成：TF32 设置恢复、分块 FP32 参考、小矩阵 CPU FP64 检查、fallback 独立失败判定。
- 已完成：batch=1 INT4 vectorized row-pair GEMV（pair_vec）并设为默认（legacy/warp/pair/pair_vec 可选）；真实 shape、边界输入、真实激活、三 prompt、FP16/BF16、官方 512 sliding、KV sharing、EOS 的 parity 全部通过。pair_vec 与 scalar pair 的 256-token 输出逐 token 一致。报告见 `reports/gemma4_c_parity.md`。
- 已完成：A 的 profile 起始 context、期望/实际步数、分组归因、每 step/per-call 时间、`--profile-shapes`；direct/engine 双后端 profile 对照；逐事件 `event_shapes`（CPU aten op 形状）；输出 JSON 的 device/driver/torch/CUDA/commit/EOS/backend/weights/317 shapes 元数据；direct finish_reason、空文本失败标记、tokenize/decode/postprocess/print 开销。
- 已完成：B 的 row-pair 与 pair_vec kernel、逻辑带宽、3 轮中位数/范围、GPU 时钟/温度/功耗记录、pair_vec/pair/warp/legacy 真实 shape 对比、3 seed × 12 shape、rows/block 2/4/8 调参、CUDA register/occupancy attributes；pair_vec 默认。报告见 `reports/gemma4_ab_gaps.md`。
- 已完成：融合 RMSNorm、RoPE 每步只算一次、addcmul RoPE、residual+RMSNorm 融合、QKV 合并、gate/up 合并、paged attention 未启用时跳过双写。
- 已完成：Gemma4 专用 paged decode attention safe kernel 的 parity 测试与启用；paged KV 路径。
- 已完成：CUDA Graph decode 原型并设为 CUDA batch=1 默认；graph 与 eager reference 在 256 token 上逐 token 一致。
- 已完成：D 的 residual/layer-scale 原地化、融合 qkv/gate-up 权重去重（graph 路径空闲显存 2.01→2.40 GiB）、block table tensor 缓存、`MINI_LLM_GEMMA4_PAGED=0` 显式 dense fallback gate；dense fallback 19+160 token 与 eager 逐 token 一致。
- 已完成：E 的 prompt 批量 prefill、快速 INT4→FP16/BF16 反量化 kernel + prefill GEMM（rows>=8）；1024/2048/4096 token prefill 约 0.55/1.04/2.37 s（2k 1966 tok/s），query-chunked prefill attention 把 4k 峰值从 4.18 降到 3.21 GiB。
- 已完成：P0 的 warp-split decode attention（8 warp 分段 KV、无 per-token barrier）；4k D512 单算子 4.18→0.915 ms（4.57×）。
- 已完成：全局 split-K attention（长上下文 full-attention，partial scratch + merge kernel）；4k D512 进一步降到 0.270 ms，2k/4k 完整模型 decode 约 18.1/29.4 ms/token；短上下文仍走 warp-split，不受影响。
- 已完成：4k 并发边界：4k+short、两个 4k batch=2 均完成；peak 3.29–3.36 GiB，free 1.20–1.22 GiB；大 prefill 由 scheduler token budget 串行化。
- 已完成：P1-3 pair_vec8（8-byte packed load）设为默认；lm_head 1.028→0.845 ms，128-token decode 8.20→7.15 ms/token；P1-4 启动预 warm 短/长 graph，双 graph 短上下文稳态约 6.86 ms/token。
- 已完成：正式验收 3 prompt × 3 轮 + 256-token；pair_vec8 + graph + batched prefill + split-K 中位数 7.15–7.45 ms/token（134.3–146.6 tok/s），prefill 0.060–0.069 s；长上下文稳态 1k/2k/4k 约 9.47/9.67/10.29 ms/token；测试 server 75、kernels 644。
- 已完成：true batched decode + batch 2/3/4 CUDA Graph + pair_batch；batch 2/4 吞吐约 169/224 tok/s（旧逐请求约 30 tok/s）；batch 2/4 token ID 与 eager 一致。
- 已完成：Ollama `gemma4:e2b` Q4_K_M 外部 baseline（1/2/4 并发，16/64 输出）：Ollama N4 decode 约 403–417 tok/s，项目约 199–226；项目 e2e 因 prefill 更快而领先。报告见 `reports/gemma4_ollama_baseline.md`。
- 已完成：P2-5 证据/工具。`scripts/check_gemma4_p2_p5.py` 统一跑 server/kernel 测试、batch parity、prefill 2k/4k、prefix-cache 命中 benchmark，输出 `reports/gemma4_p2_p5_evidence.json` / `.md`。
- 已完成（后续）：batch graph 2–16、Triton flash prefill、exact-prefix cache。19-token cold TTFT 约 60 ms；prefix-cache 命中 N=16 约 37 ms（4k prefill 1.65 s）。
- 未完成/已知问题：prefill 反量化仍是逐 projection 临时执行；sampler 最终 `.item()`、dense fallback 速度未优化。
- 当前所有改动仍未提交、未推送；等待用户检查。

### 当前验证入口


完整生成文本默认输出，无需额外开关。沿用原 speed 配置，可追加：

```text
--output-json results/gemma4_speed.json
--profile-summary results/gemma4_decode_profile.json
--profile-tokens 4
```

`--profile-tokens 4` 表示独立 profile 请求最多生成 4 个 token；排除首 token 的 prefill 与第一次 decode 后，正常情况下采集 2 次 decode。若提前 EOS 导致无采集窗口，会明确报错，不生成误导性摘要。

只有显式传入 `--profile-trace results/gemma4_trace.json` 才保存完整时间线。旧 `--profile PATH` 单独使用时写小摘要；它不再表示 Chrome trace。

低层真实权重检查可使用：

```text
python scripts/bench_gemma4_cuda.py --model models/gemma-4-E2B-it --device cuda --dtype fp16 --tokens 1 --group-size 64 --seeds 20261001 20261002 --iters 30
```

此命令测试真实权重与随机激活，不等于真实模型激活/logits parity。真实 checkpoint 全层检查耗时较长，可用 `--limit` 控制调试范围。

D/E 边界与 prefill/attention 对照：

```text
python scripts/check_gemma4_graph_boundaries.py   # 520 graph、复用、batch2、forced dense
python scripts/check_gemma4_e_boundaries.py       # 1k/2k/4k prompt + decode vs eager
python scripts/bench_gemma4_prefill.py            # GEMV vs 反量化 GEMM、TTFT、峰值显存
python scripts/bench_gemma4_attention.py          # safe attention vs dense reference
```

prefill 的 batch/GEMM 切换阈值由 `MINI_LLM_GEMMA4_PREFILL_GEMM_MIN_ROWS`（默认 64）控制；
`MINI_LLM_GEMMA4_PAGED=0` 可强制 dense fallback 做回归。

## 2. 当前基线与证据边界

### 2.1 环境与完整推理

- GPU：RTX 3060 Laptop GPU，6GB，compute capability 8.6。
- 权重与计算：项目自定义 INT4，group size=64，FP16 activation，FP16 scales。
- 推理路径：`--mode engine --int4 --int4-backend auto`。
- 示例 prompt：`请详细介绍KV Cache的作用和实现原理。`
- 项目 prompt token 数：19；最大生成 token 数：256。
- 最新完整测速：prefill 1.602 s；255 次 decode 用时 20.482 s；decode 12.45 token/s。
- 每 token decode 延迟：`20.482 / 255 ≈ 80.3 ms`。
- Gemma4 专用 CUDA attention 未启用。

Ollama 同机服务报告 decode 中位数约 107.16 token/s，端到端约 88.43 token/s。严格记录比较时还应确认 GPU offload、版本与运行参数。Ollama Q4_K_M 与项目 INT4 量化格式不同，prompt token 数也不同，属于系统级比较，不是完全相同计算的内核对照。

### 2.2 INT4 数值与微基准

当前合成测试 10 个算例均满足脚本的 fused 数值预算，包括 `lm_head`。它支持“所测输入下没有发现明显的打包、索引或数学实现问题”，不等于完整模型质量已经验证。

部分微基准结果：

| 算例 | fused 延迟 | 备注 |
| --- | ---: | --- |
| 合成 q_proj `[4096,1536]` | 0.166 ms | 不保证所有真实层采用该尺寸 |
| 合成 o/down `[1536,4096]` | 0.220～0.226 ms | 必须补充真实权重尺寸 |
| lm_head `[262144,1536]` | 4.245 ms | 大词表输出投影 |

相对 unpack + dequant + `F.linear` fallback 的加速，不等于相对 Ollama 的加速。

### 2.3 现有 profile 的限制

当前短 profile 混合了 19 个 prompt token 和 3 次 decode。INT4 CUDA kernel 累计 1.057 s，占所报告 CUDA kernel 时间约 90%，因此是明确热点候选；但这不能直接解释为“稳态 decode 的 90% 都在 INT4”。

- prefill 与 decode 混合，且 prompt 是逐 token 执行的工作负载。
- `engine::prefill` 百分比超过 100%，表中存在嵌套/关联事件，不能逐行相加。
- `int4::lm_head` 等标签与底层 INT4 kernel 可能描述同一工作，不能重复计时。
- profiler 本身有开销，记录 shapes 和完整 trace 会显著放大产物。
- `Self CPU` 与 `Self CUDA` 不能相加后当作请求 wall time。
- 当前仅测试一个默认种子；相同尺寸的多个算例会复用相同随机内容。

## 3. 新增强制要求：完整推理结果可见

### 3.1 输出行为

每次完整模型 benchmark 运行结束，至少输出一次正式测量请求的完整生成结果：

```text
=== GEMMA4 GENERATION BEGIN ===
prompt: 请详细介绍KV Cache的作用和实现原理。
path: stage5_engine
finish_reason: MAX_NEW_TOKENS
generated_tokens: 256
<该请求实际生成的全部 token 解码得到的文本，不截断、不只显示前若干 token>
=== GEMMA4 GENERATION END ===
```

- 输出的是实际计时请求的结果，不能额外生成一次不同回答冒充该结果。
- 完整指“本次请求实际生成的全部内容”，不是保证 256 token 内一定完成自然语言回答。
- 遇到 EOS/stop 需明确说明；遇到 token 上限需明确标记截断原因。
- 预热和独立 profiling 不重复打印完整答案，以免混淆结果。
- `--mode both` 时分别标注 direct 与 engine 的结果。
- 保留结构化性能输出，文本使用独立边界，便于复制和解析。
- 默认不流式逐 token 打印；在计时结束之后统一 decode、输出，避免终端 I/O 干扰速度。
- 输出文本不能占用正式 decode 的计时区间；另报告文本解码/写出开销（如需）。

### 3.2 空输出与诊断

不能把 `generated_tokens > 0` 但可见文本为空当作成功：

- 另提供 token IDs 诊断开关，支持保存全部生成 IDs。
- 可见文本为空时明确警告，并给出保留 special tokens 的解码结果。
- 明确 tokenizer、chat template、BOS/EOS/stop token 配置。
- 不静默丢弃 reasoning 或其他可见内容字段；如有区分则分别标注。
- 支持将完整结果和性能元数据保存为 UTF-8 小文件，方便用户审阅。

### 3.3 人工与自动验收

- 人工检查回答是否切题、语言连贯、是否乱码或重复、是否异常过早结束。
- 测试至少包含中文解释、英文解释、简单可核对的算术三个 prompt。
- 观察文本只是补充，不能替代 logits、数值误差和缓存行为测试。
- 新旧后端生成 token 不要求无条件完全相同；低精度误差可能在接近的 logits 间改变 greedy 选择，需要结合 top-1/top-2 margin 分析。
- INT4 算子微基准本身不生成语言；不能将其 `PARITY OK` 当作完整回答质量通过。

## 4. 阶段一：可信的 decode-only profiling

主要文件：`scripts/bench_gemma4_speed.py`。

### 工作项

1. 在 profiler 外完成 prefill 与少量 decode 预热。
2. 只采集后续固定数量 decode step，记录起始 context length 与实际采集步数。
3. 正式测速、数值诊断和 profiling 独立运行，不混用吞吐指标。
4. 默认输出小摘要与可选 JSON/文本报告；完整 Chrome trace 改为显式开关。
5. 默认关闭 shape/stack/memory 等详细记录；需要时单独开启。
6. 分别列出 CPU self time、CUDA kernel self time、调用次数、真实 shape、单次与每 token 延迟。
7. 按 INT4、attention、KV、norm/elementwise、sampler 与 Engine 调度分组，区分标签与底层 kernel，避免重复求和。
8. 对比 direct 与 engine，统一 prompt、生成长度、EOS 行为与计时边界。
9. 正常测速结束输出完整实际生成结果，符合第 3 节。

### 验收

- 不下载巨大 trace 也能定位主要 GPU 热点和 CPU 调度候选。
- 报告明确区分“原始 kernel 时间”“父级标签时间”“请求 wall time”。
- 不再用混合 prefill 的百分比代替 decode 归因。
- 可根据摘要决定优先优化计算、launch、同步还是 KV/attention。

## 5. 阶段二：重写 batch=1 INT4 GEMV

主要位置：mini-llm-kernels 的 `csrc/int4_matmul.cu` 与 Python dispatch。

### 现状与设计

当前每线程负责一个输出行、沿 K 串行累加；相邻线程读取不同权重行，并反复使用 block 同步。新实现优先评估 warp 协作与连续 K 访存：

```text
当前：线程 → 独立输出行 → 串行 K 循环
候选：warp → 相邻输出行对 → 连续 K 读取 → FP32 局部累加 → warp 归约
```

### 工作项

- 第一轮保留现有 `[N/2,K]` packed layout 和 `[N,K/64]` scales，避免同时改变权重格式。
- 共享读取 packed byte，分别解出偶数/奇数输出行的 nibble。
- 保持每输出行独立 scale，不能把相邻行的 scale 混用。
- 让 lane 沿 K 连续读，评估每 warp 一对或多对输出行。
- 评估不同 warps/block、寄存器用量、归约方式与 occupancy。
- 减少不必要的 shared memory 和 block-wide barrier。
- 为真实 MLP/down projection 与大 N `lm_head` 分别测试调度配置。
- 明确仅对 M=1 或已验证的小 M 启用新路径，多 token prefill 单独评估。
- 保留旧内核对照；数值与完整模型验证通过后再设为默认。
- 保持 current stream、device guard、同设备、dtype、contiguous、shape 和 launch error 检查。

### 指标

除延迟外，记录逻辑有效读取带宽：

$$
B_{\mathrm{logical}}=
\frac{NK/2+2NK/64}{t}
$$

对于 `lm_head`，packed 与 scales 约 214 MB，按 4.245 ms 计算约 50 GB/s。该数字不是硬件实际 DRAM 流量；不能用它替代 Nsight 计数器，也不能单凭 TFLOPS 评价 GEMV。

### 验收

- 真实主要 shape 逐项比较，不只选择表现最好的 q_proj。
- 记录 GPU 时钟、功耗/温度状态及多轮波动。
- 不允许以主要真实 shape 严重退化换取单个尺寸加速。
- 最终以无 profiler 的完整模型 decode 收益验收，不能只交微基准。

## 6. 阶段三：数值、真实权重与完整模型验证

主要文件：`scripts/bench_gemma4_cuda.py`、新增测试及完整模型 parity 工具。

### 数值参考

- 明确“量化权重数学值”与“权重先舍入至 activation dtype”两个不同合同。
- 参考 FP32 matmul 显式禁用 TF32，记录相关设置，并在退出时恢复。
- 对小矩阵使用 CPU FP64 验证参考计算本身。
- 将分组累加、参考归约与最终输出舍入纳入误差预算；不能声称 FP32 参考是精确实数结果。
- fallback 数值预算独立验证，明确警告与失败策略，不隐藏失败。
- 报告绝对误差、相对误差、P99、失败比例、NaN/Inf、最差元素与超预算最严重元素。

### 覆盖

- 多随机种子、零输入、正负抵消、不同 scales、可表示范围内的大/小输入。
- N 尾部不足完整 block、不同合法 K、多 token 与非支持条件 dispatch。
- FP16 必测；BF16 如宣称支持则单独验证。
- 真实 checkpoint 的 q/k/v/o、MLP、PLE 与 lm_head；自动读取 shape。
- 从真实 prompt 获取激活，不仅使用随机输入。
- 对比 prefill logits 及连续多个 decode step 的 logits，记录 top-1/top-2 与 margin。
- 同时核对 KV sharing、位置、sliding-window 和 EOS/stop 行为。

### 验收

新内核必须通过低层数值测试、真实激活测试和完整模型输出检查。不能只以某一次合成 `PARITY OK` 放行。

## 7. 阶段四：小算子融合、CPU 调度与 KV

以 decode-only profile 为依据，不预先假定所有项目都值得实施。

- 优先评估 RMSNorm、scale、residual、激活等高频小算子融合。
- 评估适合合并的 Q/K/V、gate/up 调用，计算权重重排和临时内存成本。
- 检查逐 token `.item()`、Python int 转换、采样、debug-state 构造与重复同步。
- 保留必要的计时同步，不能通过异步计时漏算 GPU 工作来宣称提速。
- 稳定 KV buffer、block table 和 metadata，减少每层临时分配。
- 在 paged attention 未启用时评估避免双份 KV 写入；不能破坏 fallback 或共享层语义。
- 明确 cache lifecycle 与请求结束释放策略。

CUDA Graph 在动态分配、cache 地址、metadata 和控制流稳定后单独评估；当前不直接启用。

## 8. 阶段五：attention 与 prefill 独立整改

### Attention

- 重新核查专用 CUDA attention 实现，不能仅因为 symbol 可导入就启用。
- 覆盖 D=256/512、GQA、scale=1、不同 context length、sliding window、KV sharing、paged block 边界。
- 检查 warp participation、shared memory race、barrier 与 current stream。
- 通过 parity 后再与现有 attention 比较性能；只在验证过的条件下 dispatch。
- 若没有收益则保留现有路径，不把“用了自研算子”当作验收目标。

### Prefill

- 批量处理可并行的 prompt 运算，减少逐 token 执行。
- 为多 token 输入评估 GEMM/反量化缓存方案，而不是默认复用 decode GEMV。
- 单独报告 TTFT/prefill token/s、峰值显存。
- prefill 提速不冒充稳态 decode 提速。

## 9. 执行顺序与仓库边界

| 批次 | 主要交付 | 本地可完成 | 必须在 3060 验证 |
| --- | --- | --- | --- |
| A | 完整生成输出、decode-only 小摘要 profiler | 工具和单测 | 输出内容、采集行为、耗时分布 |
| B | 新 INT4 GEMV、dispatch、数值工具 | CUDA 源码与测试准备 | 编译、数值、真实 shape 延迟 |
| C | 真实激活和模型 parity | 比对工具 | logits、生成、显存 |
| D | 按热点融合、调度和 KV 改进 | 分阶段实现 | 完整 decode 收益 |
| E | attention、prefill、可选 Graph | 独立实现与 gate | 正确性、稳定性、性能 |

- Server 仓库负责工具、Engine 接入、输出与模型验证。
- Kernels 仓库负责 CUDA 内核和相应 binding/dispatch。
- 两仓库沿用同名分支 `gemma4_adaptation_20260826`，每个可运行阶段独立提交。
- 当前会话只声明了 server 可写目录；修改 kernels 前需具备该目录写入权限，不能绕过限制。
- Mac 能完成代码和非 CUDA 验证，不能冒充完成 sm_86 编译或 GPU parity。
- 推送到 GitHub按用户授权执行；交付时列出两个仓库各自的提交与实际验证范围。

## 10. 统一性能与正确性验收

### 测试条件

- 同一 checkpoint、prompt、chat template、量化配置和固定生成长度。
- 记录实际 prompt tokens、生成 tokens、EOS/stop 与 finish reason。
- 无 debugpy，使用运行而不调试或终端直接启动。
- 卸载其他 GPU 模型，避免 Ollama 与项目同时占用 GPU。
- 充分预热，至少三轮正式测试，报告中位数与范围。
- 记录设备、驱动、PyTorch/CUDA、扩展版本和代码 commit。
- 正式测速不启用 profiler；独立 profiler 结果不作为吞吐验收值。

### 必须交付的结果

1. prefill/TTFT、decode token/s、ms/token、端到端速度及显存。
2. 新旧内核真实 shape 微基准和数值报告。
3. decode-only 小摘要及实际调用后端。
4. 至少一次正式请求的完整生成文本、prompt 和结束原因。
5. 真实激活/logits 检查与已知差异。
6. 未验证场景、回退条件及下一阶段建议。

### 收益判断

若某模块占原始单 token 延迟比例为 p，优化加速比为 s，则理想总加速上界为：

$$
S=\frac{1}{(1-p)+p/s}
$$

必须先实测 p，不能拿混合 prefill 的 90% 直接代入。任何局部优化最终都以完整 decode 的实测收益与生成正确性验收。

## 11. 建议优先实施范围

优先执行批次 A，再依据 decode-only 结果推进批次 B/C：

1. 首先让用户能直接看到完整推理结果，并得到小体积、可信的耗时摘要。
2. 对 INT4 GEMV 做有数值门槛的新实现与对照，优先真实热点尺寸。
3. 再决定小算子、KV、attention 与 Graph 的顺序。

不提前承诺 30、50 或 100 token/s；每批交付均给出可复现结果和是否继续的依据。
