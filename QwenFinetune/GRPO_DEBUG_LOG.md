# GRPO 训练调试日志

## 环境
- 机器：8× A100-SXM4-80GB，node-0（Azure ML Singularity）
- 常驻进程：pid=1804 python gpu.py，每卡占 1.2GB，实际可用 ~78GB/卡
- 模型：Qwen3-30B-A3B（MoE），BF16 ~60GB，4bit ~15-18GB
- 框架：ms-swift 4.0.3 + torch 2.6.0 + deepspeed 0.18.8 + vllm 0.11.0（降级自 0.18.1）
- 起点：SFT LoRA checkpoint（rank 64），已 merge 到 base model

## 已确认可用的配置
- reward 注册：`orms['format_quality'] = FormatQualityReward`，子进程里也生效 ✓
- 数据格式：messages 字段，1101 条，全部有 system prompt ✓
- SFT baseline reward 均值：0.211（19 条 sample）
- merged model 路径：`/vc_data/.../qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50/merged_model`

---

## 失败方案汇总

### 1. ZeRO-3 + ds3_gather_for_generation=true
- **失败原因**：rank0 单卡 allgather 全量 30B 参数，死锁
- **结论**：放弃

### 2. ZeRO-3 + ds3_gather_for_generation=false + offload_param=cpu
- **失败原因**：NCCL 死锁
- **结论**：放弃

### 3. vllm colocate 模式
- **失败原因**：NCCL OOM，torch 2.6.0 + NCCL 2.27.5 不兼容
- **结论**：放弃

### 4. vllm server 模式 + 分开设置 CUDA_VISIBLE_DEVICES
- **失败原因**：训练侧 `CUDA_VISIBLE_DEVICES=0-5`，server `CUDA_VISIBLE_DEVICES=6,7`，两边互不可见，NCCL 无法建立跨进程通信组（swift server 模式需要训练进程和 vllm worker 之间直接通过 NCCL 同步 LoRA 权重）
- **关键发现**：server 端不应传 `--adapters`，base model 即可；adapter 初始化在训练侧通过 `--adapters` 传入
- **关键发现**：两边都不设 `CUDA_VISIBLE_DEVICES`，swift 自行协调 GPU 分配
- **结论**：放弃分开设置，改为不设置

### 5. vllm server 模式 + 不限制 CUDA_VISIBLE_DEVICES（vllm 0.18.1）
- **失败原因**：`NCCL error: invalid usage`
- **根本原因**：vllm 0.18.x 引入 v1 EngineCore 架构，Worker_TP0 通过 `collective_rpc`
  被调用时，vllm 内部 NCCL comm 和 swift 的 `PyNcclCommunicator` 在同一 CUDA device
  上并发初始化，NCCL 不支持重入，报 invalid usage
- **尝试的修复**：
  - `VLLM_USE_V1=0`：vllm 0.18.1 不支持此变量，无效
  - `time.sleep(0.1)` → `time.sleep(15)`：不是时序问题，无效
  - patch rollout.py 加 `torch.cuda.set_device`：无效
  - 降级 vllm 到 0.11.0：v1 engine 依然存在，问题未解决
- **结论**：swift 4.0.x 的 server 模式与 vllm 0.11+ 的 v1 engine 存在根本兼容性问题

### 6. ZeRO-2 + QLoRA
- **失败原因**：OOM。ZeRO-2 每卡需要完整优化器状态，30B 模型放不下
- **结论**：放弃

### 7. ZeRO-3 + QLoRA(4bit) + 无 vllm + 加载旧 SFT adapter
- **失败原因**：OOM。`train_swift_grpo.sh` 里硬编码了 SFT adapter 默认路径，加载 rank 64 的 adapter 时 OOM
- **修复**：`export SFT_ADAPTER=""`，`train_swift_grpo.sh` 默认值改为空

### 8. ZeRO-3 + QLoRA(4bit) + 无 vllm + merged model（无 adapter）
- **失败原因**：`TypeError: output tensor must have the same type as input tensor`
- **根本原因**：ZeRO-3 generation 时调用 `deepspeed.zero.GatheredParameters` allgather，
  BNB 4bit tensor 无法直接 allgather 成 BF16，类型不兼容
- **结论**：ZeRO-3 + BNB 4bit 根本不兼容，必须用 BF16

### 9. ZeRO-3 + BF16 + 无 vllm + merged model，top_k=-1
- **失败原因**：`ValueError: top_k has to be a strictly positive integer, but is -1`
- **根本原因**：swift GRPO 默认 top_k=-1，transformers generation 不接受
- **修复**：`train_swift_grpo.sh` 里加 `--top_k 50 --temperature 0.7`

### 10. ZeRO-3 + BF16 + 无 vllm + merged model + top_k=50（当前方案）
- **状态**：运行中
- **显存**：每卡 ~72GB / 80GB，8 卡全部正常
- **NCCL**：8 rank 全部 Init COMPLETE ✓
- **进度**：Train 0/34，step 0 generation 中（ZeRO-3 无 vllm 需逐层 allgather，每步耗时长，非死锁）

## 重要参数发现

| 参数 | 错误用法 | 正确用法 | 来源 |
|------|---------|---------|------|
| LoRA 类型 | `--train_type lora` | `--tuner_type lora` | 官方文档 |
| deepspeed 配置 | `--deepspeed ./ds_zero3.json` | `--deepspeed zero3` | 官方文档内置字符串 |
| vllm server 端 | 传 `--adapters` | 只传 base model，不传 adapter | 官方文档 |
| CUDA 分配 | 手动分 GPU 给 server/训练 | 两边都不设 CUDA_VISIBLE_DEVICES | 实验验证 |

---



## 当前方案

**ZeRO-3 + BF16 + 无 vllm + merged model + top_k=50**

```bash
bash run_grpo_stable_canary.sh
```

关键配置：
- `GRPO_PRESET=stable_grpo_zero2_qlora`
- `DEEPSPEED_CONFIG=zero3`（内置字符串）
- `USE_VLLM=false`
- `LOAD_IN_4BIT=false`（BF16）
- `SFT_ADAPTER=""`（merged model 直接训，不加载旧 adapter）
- `MODEL_PATH` 指向 merged model
- `--top_k 50 --temperature 0.7`（在 train_swift_grpo.sh 里）
- LoRA rank 16 / alpha 32

Generation 方式：ZeRO-3 native，逐层 allgather，速度慢但能跑。

---

## 环境修改记录

| 文件 | 修改内容 |
|---|---|
| `vllm_client.py:204` | `time.sleep(0.1)` → `time.sleep(15)`（无效但未还原）|
| `rollout.py:109-113` | 加了 `torch.cuda.set_device` / `torch.cuda.current_stream`（无效但未还原）|
| vllm | 从 0.18.1 降级到 0.11.0 |

---

## 待办
1. 验证当前方案能否跑过 step 1
2. 若成功：对比 GRPO 训后 reward 与 SFT baseline（0.211）
3. 若需要加速 generation：考虑升级 swift 到修复了 vllm server 兼容性的版本
