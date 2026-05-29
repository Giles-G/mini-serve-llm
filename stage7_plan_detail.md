# 第七阶段详细规划文档（基于第六阶段现状）

## 1. 背景与目标

本项目在第六阶段已经完成了“纯 Python / PyTorch 路线下”的核心重构：
- 批量 decode / prefill 主路径向量化
- Paged KV 存储结构与批量 gather
- 采样向量化
- 基准脚本与阶段计时
- 调度稳定性增强（KV 预留水位、非阻塞排队、deadlock breaker）

第七阶段不做自定义 CUDA kernel（第八阶段再做），目标是：

1. 在不改算法语义的前提下，进一步降低 Python 调度和 eager 执行开销
2. 提高图执行命中率（为 torch.compile / graph 化做好形状管理）
3. 在“可复现 benchmark 口径”下拿到稳定可解释的吞吐收益
4. 保持第六阶段已有稳定性机制，不引入卡死或精度回退

---

## 2. 第六阶段回顾（上一阶段干了什么）

### 2.1 计算路径改造

- decode attention：从“逐请求循环”改为“批量路径”
  - 现状是 `gather -> padded batched attention`，并不是 vLLM 真 paged kernel
  - GQA 不物化 `repeat_kv`，减少显存复制
- prefill attention：支持批量 prefill + block-diagonal causal mask
- cross-layer context：把 positions / slot 索引等跨层复用，减少重复构造

### 2.2 采样路径改造

- `sample_batch` 完整向量化，避免请求级 Python 循环

### 2.3 工程能力补齐

- `scripts/bench_engine.py`：
  - 支持多 batch、多次运行取中位数
  - 输出 phase breakdown（forward_decode / prefill / sample）
  - 已支持输出样例推理结果
- 命名与注释修正：明确当前 attention 实现不是 fused paged kernel

### 2.4 稳定性机制

- KV 不足时不再直接崩溃退出，改为非阻塞排队
- 调度器加入 KV 预算预检查（避免本步聚合超配）
- 引入 decode KV 预留水位，避免 prefill 抢空 KV
- 引入 deadlock breaker（连续无进展时临时放宽水位）

### 2.5 阶段价值

第六阶段的核心价值是：
- 从“能跑”走到“批量化高性能雏形 + 基本服务稳定性”
- 为第七/第八阶段奠定结构基础（尤其是 batch 结构和 metadata 组织）

---

## 3. 为什么还需要第七阶段

第六阶段主要解决的是“算法路径正确 + 向量化改造”。
但在执行层仍然存在这些问题：

1. 仍以 eager 执行为主，Python 调度/函数边界开销明显
2. 动态 shape 波动导致图优化难以充分命中
3. 没有系统化区分“短 prompt / 长 prompt / 高并发”的最优运行策略
4. benchmark 虽可用，但还缺“回归门槛”和更标准化口径

因此第七阶段的关键词是：
- 执行时优化（Execution-time optimization）
- 图优化命中率（Compile/Graph hit rate）
- 性能可复现与可回归（Reproducible performance）

---

## 4. 第七阶段详细任务

## 4.1 Task A：Torch Compile 引入与分层落地

### 要做什么

1. 对 decode 主路径尝试 `torch.compile`（优先）
2. prefill 路径单独尝试 compile（次优先）
3. 采用开关式接入（可配置启用/禁用），避免一次性替换

### 建议顺序

- A1: 只编译 decode block 前向链路
- A2: 验证数值一致性 + 性能增益
- A3: 再尝试 prefill compile

### 风险点

- 动态 shape 导致 recompile 频繁
- 某些设备/版本下 compile 反而退化

### 验收标准

- compile 开启时比 eager 有稳定增益（至少在 N=4/N=8）
- 不出现功能回归（输出可接受范围内一致）

---

## 4.2 Task B：形状分桶（Bucketing）

### 要做什么

对 decode/prefill 引入桶化策略，减少 shape 抖动：
- batch bucket：N in {1, 2, 4, 8, 16}
- context bucket：按长度区间分段
- prefill chunk bucket：固定常用 chunk 大小

### 为什么必要

compile/graph 最怕形状频繁变化；桶化能显著提高命中率，降低图失效开销。

### 验收标准

- benchmark 中同一 profile 下 shape 切换次数显著下降
- compile 热身后吞吐更稳定

---

## 4.3 Task C：执行缓冲区复用（减少小对象创建）

### 要做什么

1. 减少 step 内临时 tensor 重复创建
2. 尽可能复用索引/中间缓冲区
3. 降低 host->device 小块 metadata 传输频次

### 预期收益

- 降低 Python 与 allocator 开销
- 减少小 kernel / 小张量管理抖动

### 验收标准

- 关键 phase 的 avg ms 降低
- profiler 中小对象创建热点下降

---

## 4.4 Task D：调度策略性能化微调（基于第六阶段稳定性机制）

### 要做什么

1. 把 `kv_decode_block_reserve` 作为负载相关参数调优
2. 调整 `kv_reserve_relax_after_no_progress_steps` 阈值
3. 区分“吞吐优先配置”和“稳定性优先配置”

### 建议策略

- 吞吐优先：较低 reserve
- 稳定优先：较高 reserve + 更积极 breaker

### 验收标准

- 不崩溃、不活锁
- 在既定稳定性约束下取得更优 tok/s

---

## 4.5 Task E：Benchmark 体系升级（性能回归门禁）

### 要做什么

1. 固定测试 profile（short / medium / long）
2. 固定采样口径（greedy vs non-greedy）
3. 增加结果输出字段：
   - tok/s
   - 完成率
   - phase breakdown
   - 样例输出
4. 建立回归阈值（例如 tok/s 回退 > 8% 报警）

### 验收标准

- 同一 commit 重复运行波动可控
- 能快速定位“是 decode 退化还是 prefill 退化”

---

## 5. 预期性能提升区间（第七阶段）

说明：以下是合理预期，不是硬保证，受设备与工作负载影响。

- 保守：+10% ~ +20%
- 中位：+20% ~ +40%
- 理想（桶化 + compile 命中率高）：+40% ~ +70%

第七阶段的价值不仅在“峰值 tok/s”，更在于：
- 同口径下更稳定
- 波动更小
- 便于第八阶段 kernel 替换

---

## 6. 与第八阶段边界（避免阶段目标混淆）

第七阶段不做：
- 自定义 CUDA fused attention kernel
- 自定义 sampler kernel
- 内核级 shared memory / warp-level 手工优化

这些都放在第八阶段。

第七阶段要做的是：
- 把现有 PyTorch 路径压到“执行框架层”的上限
- 为第八阶段提供稳定、可对比、可回归的基线

---

## 7. 推荐执行顺序（最小返工）

1. A（decode compile）
2. B（shape bucketing）
3. C（缓冲区复用）
4. D（水位参数调优）
5. E（benchmark 回归体系）

原因：先把主路径 compile 命中做起来，再做策略调优，最后固化 benchmark 门禁。

---

## 8. 第七阶段完成定义（DoD）

满足以下条件视为第七阶段完成：

1. 主 benchmark 口径下 tok/s 相比第六阶段有稳定提升
2. 无正确性回归（对齐基准测试通过）
3. 无稳定性回归（低 KV 压测不崩溃、不活锁）
4. benchmark 脚本可一键复现并输出完整分析结果
5. compile/策略参数可配置，能按场景切换

---

## 9. 附：建议基准命令模板

```bash
python scripts/bench_engine.py \
  --batch-list 1,2,4,8 \
  --max-new 48 \
  --runs 3 \
  --greedy \
  --kv-decode-block-reserve 2 \
  --kv-reserve-relax-after-no-progress-steps 8
```

非 greedy 场景：
```bash
python scripts/bench_engine.py \
  --batch-list 1,2,4,8 \
  --max-new 48 \
  --runs 3 \
  --kv-decode-block-reserve 2 \
  --kv-reserve-relax-after-no-progress-steps 8
```

---

如果你认可这份阶段定义，下一步可以直接按这个文档拆成第七阶段实施 checklist，然后进入实际编码。