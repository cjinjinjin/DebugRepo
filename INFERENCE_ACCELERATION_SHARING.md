# 推理加速实战经验分享：ZImage × Gemma4

> 从 12+ 种优化技巧到生产部署的完整决策路径

---

## Part 1: 通用推理加速技巧框架

推理加速的所有技巧归纳为 **3 个大方向**：

### 方向一：让每次计算更快

> 核心思路：同样的计算量，用更高效的方式执行

| 技巧 | 原理 | 适用场景 | 前提条件 |
|------|------|---------|---------|
| **torch.compile** | Python → Triton/C++ 编译，算子融合，消除 Python overhead | 几乎所有 PyTorch 模型 | PyTorch 2.0+；模型中不含动态控制流或不支持的算子 |
| **TensorRT** | NVIDIA 专用推理引擎，层融合、精度校准、kernel auto-tuning | CNN、标准 Transformer | 所有算子需 TRT 支持；不支持 complex64、某些 scatter 操作 |
| **ONNX Runtime** | 跨平台推理引擎，图优化 + 多后端 | 可导出为静态图的模型 | torch.export 成功；无嵌套动态输入；编码器需 optimum 支持 |
| **FlashAttention** | 分块计算 attention，减少 HBM 访问 | head_dim ≤ 256 的标准 attention | head_dim 限制 (FA v2 ≤ 256)；PyTorch SDPA 已自动选择最优 kernel |
| **量化 (INT8/FP8/4-bit)** | 降低权重/激活精度，减少内存带宽和计算量 | LLM (decode-bound, 内存带宽瓶颈) | 需对应硬件支持（FP8→H100, INT8→需优化 kernel）；精度敏感模型可能质量崩溃 |

### 方向二：减少计算次数

> 核心思路：跳过不必要的计算步骤或 token

| 技巧 | 原理 | 适用场景 | 前提条件 |
|------|------|---------|---------|
| **First Block Cache (FBC)** | 相邻 denoising step 中间特征相似时跳过后续 block | Diffusion 模型 | diffusers 0.38+；模型需适配 `_set_context` 接口 |
| **TeaCache** | 用 timestep embedding 变化量预测是否可跳过整个 transformer | Diffusion 模型 | 需针对具体模型校准多项式系数 |
| **DeepCache** | 缓存 UNet 高层特征，低层每步计算 | UNet-based Diffusion 模型 | **仅限 UNet 架构**，Transformer-based 不可用 |
| **减少推理步数** | 9→7→5 步，线性降低延迟 | Diffusion 模型 | 需验证图像质量下限 |
| **No-CoT / No-Think** | 减少 LLM 输出 token 数量 | LLM (decode-bound) | 仅适用于 decode-bound 场景；prefill-bound 无效 |
| **输入截断** | 限制输入长度减少 prefill 和 KV cache | LLM | 仅在 prefill-bound 或 KV cache 受限时有效 |
| **Two-Step 生成** | 拆分任务为规划+执行两步，用 stop_strings 截断 | LLM 结构化输出 | 任务可拆分；batch decode 比串行高效 |

### 方向三：提高并行吞吐

> 核心思路：同一时间处理更多请求

| 技巧 | 原理 | 适用场景 | 前提条件 |
|------|------|---------|---------|
| **Continuous Batching** | 动态组 batch，请求完成即释放，新请求即加入 | LLM serving | 需 vLLM / TGI 等 serving 框架 |
| **Tensor Parallel (TP)** | 模型切分到多卡，减少单卡计算量 | 大模型 (>单卡 VRAM) | TP 数不是越大越好——通信开销可能超过计算收益 |
| **并发扩展** | 增加同时处理的请求数 | 高吞吐场景 | 以增加单请求延迟为代价换吞吐 |
| **量化→多副本** | 量化减少 VRAM → 同卡部署更多副本 | 吞吐优先场景 | 量化质量损失可接受 |

---

## Part 2: ZImage Case Study — Diffusion Transformer 的优化困境

### 2.1 模型简介

- **ZImage**: 基于 Diffusion Transformer 的文生图模型 (ZImagePipeline)
- 架构：30 层 Transformer blocks（非 UNet），Qwen3 文本编码器，VAE 解码器
- 部署平台：DLIS (A100/A6000)
- 推理流程：9 步 denoising，每步完整过 30 层 Transformer

### 2.2 Profiling：瓶颈在哪里？

| 组件 | 占比 | 说明 |
|------|------|------|
| **Transformer** | **95.5%** | 绝对主导，优化必须集中在这里 |
| VAE | 2.9% | 优化无意义 |
| Text Encoder | 1.6% | 优化无意义 |

**CUDA 时间分解：**

| 类别 | 占比 | 含义 |
|------|------|------|
| **GEMM (aten::mm)** | **40.3%** | 493 次矩阵乘法 (Q/K/V/Out + FFN)，量化的目标 |
| **Command Buffer Full** | **36.8%** | CPU 来不及发 kernel → torch.compile 的核心价值 |
| SDPA (attention) | 7.7% | 已用 mem_efficient kernel，FlashAttention 空间小 |
| dtype cast (copy_) | 8.3% | bf16↔fp32 转换，难以消除 |

**关键发现：** 30 个 Transformer blocks 完全均匀 (~19ms/step)，没有单一热点 block。

### 2.3 技巧选型决策树

#### ✅ 成功路线：编译优化 + 缓存跳步

| 技巧 | 加速比 | 质量 (PSNR) | 备注 |
|------|--------|------------|------|
| torch.compile (reduce-overhead) | **1.30x** | 无损 | CUDA Graphs 消除 Command Buffer Full 瓶颈 |
| torch.compile (default) | **1.25x** | 无损 | 无 CUDA Graphs，但算子融合有效 |
| FBC (t=0.3) | **1.20x** | 32.3dB ✅ | 相邻步特征相似时跳过后续 block |
| TeaCache (t=0.20) | **1.24x** | 33.8dB ✅ | 校准后的多项式系数预测可跳步 |
| MagCache | **1.21x** | — | 硬编码校准比例 |
| **torch.compile + FBC** | **1.55x** | 30.6dB ✅ | ⭐ **最佳组合**，必须用 mode='default'（CUDA Graphs 与 FBC 冲突） |
| 减少步数 9→7 | **1.28x** | 需验证 | 线性关系，最简单的加速 |

#### ❌ 失败路线及原因

| 技巧 | 分类 | 失败原因 | 教训 |
|------|------|---------|------|
| **TensorRT** | 计算加速 | ZImage RoPE 使用 `torch.complex64`，TRT 不支持；修补为 Real RoPE 后又遇 `aten.scatter` bf16/int32 类型不匹配 | 新架构的非标准算子是 TRT 的硬伤 |
| **FlashAttention** | 计算加速 | ZImage head_dim=512，超过 FA v2 上限 256；PyTorch SDPA 已自动选择 mem_efficient kernel | 非标准 head_dim 直接堵死 FA |
| **ONNX Runtime** | 计算加速 | ① complex64 无法 export ② 嵌套 `list[Tensor]` 输入不兼容 ③ optimum 不支持 Qwen3 编码器 ④ PyTorch 2.11 移除了 onnxrt 后端 | 多重阻碍，短期无解 |
| **DeepCache** | 缓存跳步 | 需要 UNet 架构，ZImage 用 Transformer（`'ZImagePipeline' has no attribute 'unet'`） | 架构差异导致方法完全不适用 |
| **INT8 W8A8 (torchao)** | 量化 | **2.1x 更慢**——A100 (SM80) 没有优化的 INT8 GEMM kernel，只有 H100 (SM90a) 有 | ⚠️ 量化不等于加速，取决于硬件 kernel 支持 |
| **INT8 (bitsandbytes)** | 量化 | 速度 1.09x 微增，但 **PSNR 9.6dB 质量完全崩溃**；每个 GEMM 被替换为 5-6 个 quant/dequant kernel | ⚠️ Diffusion 模型对精度极度敏感 |
| **torch.compile + INT8** | 量化+编译 | **155x 更慢** (786,511ms)——AffineQuantizedTensor 子类破坏 Triton 算子融合 | 量化与编译可能严重冲突 |
| **FP8** | 量化 | A6000/A100 硬件不支持 FP8 (需 H100/Ada SM90+) | 硬件代际限制 |
| **Channels-Last (NHWC)** | 内存布局 | 4-4.6% 更慢，格式转换开销 > 收益 | 不是所有"教科书优化"都有效 |
| **TF32 matmul** | 精度 | 无效果——模型已在 BF16，TF32 仅影响 FP32 matmul | 了解模型实际运行精度 |
| **TaylorSeer** | 缓存跳步 | Hook 不匹配 ZImage block 结构，0.97x 反而更慢 | 方法与模型结构强耦合 |

#### 🔬 重点展开：INT8 量化在 Diffusion 模型上为什么全军覆没？

**三条路都试了，三条路都失败：**

1. **torchao W8A8**: A100 没有 SM90a 的 INT8 GEMM 优化 kernel → 实际走了慢速路径 → 2.1x 更慢
2. **bitsandbytes INT8**: 虽然速度微增 1.09x，但每个 GEMM 变成 5-6 个 kernel (量化→INT8 GEMM→反量化)，而且 **PSNR 从 40+dB 暴跌到 9.6dB**——对 diffusion 模型来说，每步 denoise 的微小误差会累积放大
3. **torch.compile + INT8**: AffineQuantizedTensor 是 PyTorch 子类，完全破坏了 Triton 的算子融合 → 155x 更慢

**根因：** Diffusion 模型的 9 步 iterative denoise 会**累积量化误差**，不像 LLM 的 autoregressive decode 每步独立。加上 A100 缺乏优化 INT8 kernel，量化在 diffusion 上目前不可行。

### 2.4 最终部署方案

```
torch.compile(mode='default') + FBC(threshold=0.3) = 1.55x 加速
```

- 延迟：4666ms → ~3010ms (A100)
- 质量：PSNR 30.6dB（可接受）
- 无需模型修改，纯运行时优化
- 备选：TeaCache(t=0.20) = 1.24x，质量更好 (PSNR 33.8dB, min 32.2dB)

---

## Part 3: Gemma4 Case Study — LLM 的 Serving 架构优化

### 3.1 模型简介

- **Gemma4 26B-A4B-it**: Google MoE LLM，26B 总参数 / 3.8B 激活参数
- 用途：广告 Landing Page → 广告文案生成
- 部署平台：DLIS (A100/A6000)，vLLM serving
- 评估：Zero-shot format compliance 95.9% (vs Qwen3 47.9%)

### 3.2 Profiling：瓶颈在哪里？

| 指标 | 值 | 说明 |
|------|-----|------|
| **Decode 占比** | **76-77%** | 绝对瓶颈，优化 decode 吞吐是关键 |
| TTFT (首 token) | 0.02s | prefill 可忽略 |
| 延迟 vs token 数 | 线性关系 | 1024 tok / 632 tok ≈ 1.62x → 82.65s / 52.0s ≈ 1.59x |
| MoE routing 开销 | 可忽略 | Dense 4.5B (E4B) 12.9 tok/s ≈ MoE 3.8B (26B-A4B) 12.1 tok/s |

**关键发现：** 因为是 decode-bound，**减少输出 token 数 = 线性减少延迟**。

### 3.3 逐步优化：每步都有明确收益

#### Step 1: Tensor Parallel 调优 — 最大单项加速

| 配置 | 延迟 | 原因 |
|------|------|------|
| TP=4, CoT, C=8 | ~200s | 4 卡间通信开销巨大，远超计算收益 |
| **TP=2, No-CoT, C=8** | **42s** | 通信减半 + 输出 token 减少 |

> **教训：TP 不是越大越好。** 对于 3.8B 激活参数的 MoE 模型，2 卡足够放下，4 卡的 AllReduce 通信反而成为瓶颈。

#### Step 2: vLLM Continuous Batching — 吞吐量飞跃

| 框架 | 190 条记录耗时 | Decode 吞吐 | 加速比 |
|------|-------------|------------|--------|
| HF Transformers (单条) | ~67 min | 30.5 tok/s | 1x |
| **vLLM (1×A100)** | **1.4 min** | **869.5 tok/s** | **28x** |
| vLLM (2×A100) | 1.1 min | 1,075 tok/s | 35x |

#### Step 3: Two-Step 生成 + stop_strings — 减少无效 token

| 阶段 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| Step 2 输出 token | 1041 | 311 | **-70%** |
| Step 2 decode 时间 | 30.3s | 10.2s | **-66%** |
| 单条总时间 | 39.6s | 21.2s | **-46%** |

Two-Step vs Single-Prompt (vLLM, 190 samples):

| 指标 | Single-Prompt | Two-Step | 改善 |
|------|--------------|----------|------|
| 总时间 | 122.7s | **68.8s** | **-44%** |
| 总输出 token | 159,474 | 74,003 | **-54%** |

#### Step 4: AWQ 4-bit 量化 — 用 VRAM 换吞吐

| 配置 | 总时间 (190条) | Decode 吞吐 | Format 合规 | VRAM |
|------|-------------|------------|------------|------|
| BF16, 1×A100 | 84.9s | 869.5 tok/s | 99.5% | ~52GB |
| **AWQ 4-bit, 1×A100** | **70.0s** | **1,105.2 tok/s** | 98.9% | **~19GB** |
| BF16, 2×A100 | 68.8s | 1,075 tok/s | 100% | ~52GB×2 |

> **单卡 AWQ 匹配双卡 BF16！** 原因：VRAM 节省 → KV cache pool 更大 → batch capacity 更高 → 吞吐更高。

#### Step 5: 并发扩展

| 并发 | 吞吐 | 单请求延迟 |
|------|------|----------|
| C=8 | 0.16 req/s | 42.0s |
| C=16 | 0.24 req/s | 50.1s |
| **C=32** | **0.32 req/s** | 58.2s |

吞吐翻倍，延迟增加 39%——经典的吞吐-延迟权衡。

### 3.4 边际/失败的尝试

| 技巧 | 结果 | 原因 |
|------|------|------|
| LP 截断至 2000 字符 | 无明显加速 | decode-bound，输入长度对延迟影响 <5% |
| E2B 小模型 (2B) | 最快 (15.9 tok/s) 但 **20% format 合规率** | 模型能力不足，不可用 |
| GPTQ 4-bit | 单条更慢 (-14% tok/s) 但 VRAM -75% | 纯 dequant 开销，不如 AWQ (有 fused kernel) |

### 3.5 最终部署方案

```
AWQ 4-bit + vLLM + Two-Step + TP=2 + C=32
```

- 单条延迟：~0.36s (A100) / ~1.7s (A6000)
- 190 条批处理：68.8s (vs HF Transformers 67 min = **59x 加速**)
- Format 合规：98.9%
- 单卡 VRAM：~19GB（可在 A6000 48GB 上部署）

---

## Part 4: 对比总结

### 技巧 × 模型类型 兼容性矩阵

| 优化技巧 | ZImage (Diffusion Transformer) | Gemma4 (MoE LLM) | 差异原因 |
|---------|-------------------------------|-------------------|---------|
| **torch.compile** | ✅ 1.30x (核心优化) | ⚪ 不需要 (vLLM 内置) | Diffusion 有严重 Command Buffer Full 问题 |
| **TensorRT** | ❌ complex64 + scatter 不支持 | ⚪ 不需要 (vLLM 内置) | 非标准算子阻断 |
| **FlashAttention** | ❌ head_dim=512 > 256 限制 | ✅ vLLM 自动启用 | 架构参数不满足前提 |
| **INT8/FP8 量化** | ❌ 质量崩溃 + 无优化 kernel | — 未测试 | Diffusion 迭代累积误差 |
| **4-bit 量化 (AWQ/GPTQ)** | — 未测试 | ✅ AWQ 1.27x 吞吐提升 | LLM decode-bound，带宽收益 > dequant 开销 |
| **FBC / TeaCache** | ✅ 1.20-1.24x | ❌ 不适用 | 仅限 diffusion 多步 denoise |
| **DeepCache** | ❌ 需要 UNet 架构 | ❌ 不适用 | 架构限制 |
| **Continuous Batching** | ⚪ 单请求场景 | ✅ **28-59x** (最大杠杆) | LLM serving 生态成熟 |
| **Tensor Parallel** | ⚪ 单 GPU 够用 | ✅ TP=2 最优 (TP=4 反而慢) | 通信开销与模型大小的平衡 |
| **减少输出量** | ✅ 减少步数 (线性) | ✅ No-CoT -18%, Two-Step -44% | decode-bound 模型收益最大 |

### 核心 Takeaway

#### 1. 模型架构决定优化路线

- **Diffusion Transformer**: 优化生态不成熟（TRT/FA/ONNX/量化全部失败），只有 torch.compile + 缓存跳步可走
- **MoE LLM**: 优化生态成熟，Serving 架构 (vLLM + batching + TP) 是最大杠杆

#### 2. Profiling 先行，不要盲试

- ZImage profiling 发现 Command Buffer Full 36.8% → 直接指向 torch.compile
- Gemma4 profiling 发现 decode 77% + TTFT 0.02s → 直接指向 reduce output tokens + batching
- **不做 profiling 就做优化 = 在黑暗中射箭**

#### 3. "教科书优化"不一定有效

- INT8 量化："理论上应该更快" → 实际 2.1x 更慢 (无优化 kernel) 或质量崩溃 (9.6dB)
- TP=4："更多卡应该更快" → 实际比 TP=2 慢 5x (通信开销)
- Channels-Last："NHWC 应该更快" → 实际慢 4% (转换开销)
- **永远要实测，永远要看数据**

#### 4. 组合优化 ≠ 简单相加

- torch.compile(reduce-overhead) + FBC → 冲突（CUDA Graphs vs 动态 hooks）
- torch.compile(default) + FBC → **1.55x**（需要选对 compile mode）
- torch.compile + INT8 → **155x 更慢**（子类破坏融合）
- **组合前必须理解每个优化的实现机制**

#### 5. 最终加速倍数

| 模型 | 优化前 | 优化后 | 总加速 |
|------|--------|--------|--------|
| **ZImage** | 4666ms / 请求 | ~3010ms / 请求 | **1.55x** |
| **Gemma4** | 67 min / 190 条 (HF) | 68.8s / 190 条 (vLLM) | **59x** |
