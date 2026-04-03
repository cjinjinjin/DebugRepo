# GRPO 训练调试日志

## 环境
- 机器：8× A100-SXM4-80GB，node-0（Azure ML Singularity）
- 常驻进程：pid=1804 python gpu.py，每卡占 1.2GB，实际可用 ~78GB/卡
- 模型：Qwen3-30B-A3B（MoE），BF16 ~60GB，4bit ~15-18GB
- 框架（旧）：ms-swift 4.0.3 + torch 2.6.0 + deepspeed 0.18.8 + vllm 0.11.0（降级自 0.18.1）
- 框架（新，2026-04-03 重建）：通过 `setup_envs.sh swift_train` 重建环境 → ms-swift 4.1.0.dev0（GitHub main）+ torch 2.8.0 + vllm 0.19.0 + trl 0.28.0 + transformers 4.57.6
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

### 6. ZeRO-2 + QLoRA（第一次尝试，run_grpo_stable_scaleup.sh）
- **失败原因**：OOM。表面看是 ZeRO-2 optimizer state 问题，实际根因是脚本硬编码了旧 SFT adapter 默认路径，加载 rank64 adapter 时 GPU 0 已被 base model 占满
- **真实错误**：`torch.OutOfMemoryError` 发生在 `load_peft_weights → safe_load_file`，GPU 0 只剩 2.44MB，rank0 进程独占 37GB
- **修复**：`export SFT_ADAPTER=""`，MODEL_PATH 改为 merged model，与 canary 脚本保持一致
- **结论**：未真正测试 ZeRO-2 的显存极限，需重跑验证

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

### 10. ZeRO-3 + BF16 + 无 vllm + merged model + top_k=50
- **状态**：✅ 跑通（能完成完整 step），但速度不可接受
- **显存**：每卡 ~69.3GB / 80GB，8 卡全部正常，NCCL Init COMPLETE ✓
- **Step 1 数据**：
  - step_time: 6387s（~1.8小时/step），34 steps 预计 **4天21小时**
  - reward: 0.091（低于 SFT baseline 0.211，step 1 正常）
  - completions/clipped_ratio: 1.0 —— 所有生成全被截断到 MAX_COMPLETION_LENGTH=512
  - kl: 0.0 —— 异常，policy 与 reference model 完全一致，GRPO 未在学习
- **结论**：正确性待验证（kl=0.0 异常），速度不可用（4天21小时），必须解决 generation 加速问题
- **对应脚本**：`run_grpo_stable_canary.sh`

### 11. ZeRO-2 + QLoRA（第二次，修复 adapter 加载问题后）
- **状态**：待测试
- **配置**：`run_grpo_stable_scaleup.sh`，ZeRO-2 + QLoRA 4bit，无 vllm，merged model，无旧 adapter
- **关键修复**：`SFT_ADAPTER=""`，`MODEL_PATH` 指向 merged model，补充 `LORA_RANK=64` / `LORA_ALPHA=128`
- **待观察**：preflight 输出的 ZeRO-2 Adam states 估算；OOM 是否再次出现及位置

### 环境兼容性问题（旧环境，已通过重建环境解决）
- **现象**：`vllm/_C.abi3.so: undefined symbol` 导致 `trl.GRPOTrainer` import 崩溃
- **触发条件**：trl 的 GRPOTrainer 在模块加载时 import vllm 内部路径，即使 `USE_VLLM=false` 也会触发
- **验证**：`python -c "import vllm"` 正常，问题在 trl 加载时触发的特定 vllm 子模块
- **影响**：之前部分"OOM"失败实际是被此错误打断，并非真实显存 OOM
- **解决**：通过 `setup_envs.sh swift_train` 重建环境解决（见下方"环境重建记录"）

### 12. 环境重建过程中的依赖冲突（2026-04-02）
- **背景**：决定重建 swift_train 环境，使用 `setup_envs.sh swift_train`
- **安装后 pip 警告**（非致命）：
  - `datasets 3.6.0` 要求 `dill<0.3.9`，实际 `dill 0.4.1`
  - `datasets 3.6.0` 要求 `fsspec<=2025.3.0`，实际 `fsspec 2026.3.0`
  - `gradio 5.50.0` 要求 `pillow<12.0`，实际 `pillow 12.2.0`
  - `ms-swift 4.0.1` 要求 `transformers<5.3.0`，实际 `transformers 5.5.0`（中间版本）
- **结论**：pip 警告暂不影响运行。ms-swift 从 GitHub main 安装为 4.1.0.dev0，绕过了 PyPI 上 4.0.1 的 transformers 限制

### 13. trl 版本兼容性排查（2026-04-02）
- **初始状态**：旧环境 trl 版本未知，`from trl import GRPOTrainer` 报错 `No module named 'vllm_ascend'`
- **根因**：新版 trl 对昇腾 NPU（vllm_ascend）的支持代码未做 try/except 保护，CUDA 环境下也会尝试 import
- **排查过程**：
  - 安装 `trl==0.12.2` → 报错 `cannot import name 'GRPOTrainer'`（GRPOTrainer 在 0.15+ 才引入）
  - 安装 `trl==0.15.2` → 报错 `cannot import name 'AutoVideoProcessor' from 'transformers'`（当时 transformers 版本 4.46.3 太旧）
- **最终解决**：对齐到 canary 已验证可用的版本组合 `transformers==4.57.6 + trl==0.28.0`
- **验证**：`python -c "from trl import GRPOTrainer; print('OK')"` 通过

### 14. ZeRO-2 + QLoRA scaleup 运行异常（2026-04-02）
- **配置**：`run_grpo_stable_scaleup.sh`，ZeRO-2 + QLoRA 4bit + 无 vllm
- **现象**：运行 1.5 小时未到达 Step 1，GPU 显存仅 30G/80G（~37%）
- **分析**：
  - 显存 30G/80G 异常偏低（QLoRA 4bit 30B 模型应占 ~50-60G/卡），说明模型未正常加载或初始化卡住
  - 可能死因：
    1. `LOAD_IN_4BIT=true` + ZeRO-2 + adapter 分离加载组合导致 bitsandbytes 初始化卡死
    2. 后台 vllm 导入错误导致 silent failure，外层进程空等
    3. NCCL 死锁
- **与 canary 的关键差异**：
  | 参数 | canary（成功） | scaleup（卡住） |
  |------|---------------|----------------|
  | MODEL_PATH | merged_model | 预训练模型 + SFT adapter 分开加载 |
  | DEEPSPEED_CONFIG | `zero3`（内置字符串） | `ds_zero2.json` |
  | LOAD_IN_4BIT | `false` | `true` |
  | MAX_COMPLETION_LENGTH | 512 | 1024 |
  | LORA_RANK | 16 | 64 |
- **待排查**：
  1. 先临时设 `LOAD_IN_4BIT=false` + `MAX_COMPLETION_LENGTH=512` 复现 canary 成功路径
  2. 逐步恢复参数定位根因
  3. 确认 SFT_ADAPTER 路径是否真实存在

## 重要参数发现

| 参数 | 错误用法 | 正确用法 | 来源 |
|------|---------|---------|------|
| LoRA 类型 | `--train_type lora` | `--tuner_type lora` | 官方文档 |
| deepspeed 配置 | `--deepspeed ./ds_zero3.json` | `--deepspeed zero3` | 官方文档内置字符串 |
| vllm server 端 | 传 `--adapters` | 只传 base model，不传 adapter | 官方文档 |
| CUDA 分配 | 手动分 GPU 给 server/训练 | 两边都不设 CUDA_VISIBLE_DEVICES | 实验验证 |

---



## 当前方案

**ZeRO-2 + QLoRA(4bit) + 无 vllm + merged model（新环境，待重新测试）**

```bash
bash run_grpo_stable_scaleup.sh
```

关键配置：
- `GRPO_PRESET=stable_grpo_zero2_qlora`
- `DEEPSPEED_CONFIG` 指向 `ds_zero2.json`
- `USE_VLLM=false`
- `LOAD_IN_4BIT=true`（QLoRA 4bit）
- `SFT_ADAPTER=""`（merged model 直接训，不加载旧 adapter）
- `MODEL_PATH` 指向 merged model
- `LORA_RANK=64` / `LORA_ALPHA=128`
- `--top_k 50 --temperature 0.7`

ZeRO-2 generation 不需要逐层 allgather，速度比 ZeRO-3 快一个数量级。
核心风险：ms-swift 是否正确 freeze 基座参数，若未正确 freeze 则 ZeRO-2 会为 30B 参数分配 Adam state（~480GB），导致 OOM。
preflight 脚本会在启动时估算并输出风险警告。

---

## 环境修改记录

| 文件 | 修改内容 |
|---|---|
| `vllm_client.py:204` | `time.sleep(0.1)` → `time.sleep(15)`（无效但未还原）|
| `rollout.py:109-113` | 加了 `torch.cuda.set_device` / `torch.cuda.current_stream`（无效但未还原）|
| vllm | 从 0.18.1 降级到 0.11.0（旧环境）|
| `train_swift_grpo.sh` | 加 preflight 参数估算脚本（训练前自动运行，估算 ZeRO-2 optimizer state 风险）|
| `run_grpo_stable_scaleup.sh` | 清除硬编码 SFT adapter 路径；MODEL_PATH 改为 merged model；补充 LORA_RANK/LORA_ALPHA |
| `run_grpo_stable_canary.sh` | `SFT_ADAPTER=""`，MODEL_PATH 指向 merged model，DEEPSPEED_CONFIG=zero3 |
| `setup_envs.sh` | 新增 swift_train 环境重建脚本：ms-swift 4.1.0.dev0 + torch 2.8.0 + vllm 0.19.0 + trl 0.28.0 + transformers 4.57.6 |

### 环境重建记录（2026-04-02）
通过 `bash setup_envs.sh swift_train` 从零重建 swift_train conda 环境：
- **旧版本**：ms-swift 4.0.3, torch 2.6.0, deepspeed 0.18.8, vllm 0.11.0, trl 未知, transformers 4.46.3
- **新版本**：ms-swift 4.1.0.dev0 (GitHub main), torch 2.8.0, vllm 0.19.0, trl 0.28.0, transformers 4.57.6
- **动机**：旧环境 trl/vllm/transformers 版本冲突严重，无法正常 import GRPOTrainer
- **验证**：`from trl import GRPOTrainer` 通过

---

## 待办
1. 在新环境下重跑 `run_grpo_stable_scaleup.sh`（ms-swift 4.1.0.dev0 可能已修复旧环境的兼容性问题）
2. 如果仍卡住：先用 Gemini 建议的冒烟测试参数（`MAX_COMPLETION_LENGTH=128`, `GRADIENT_ACCUMULATION_STEPS=1`）验证是否能到 Step 1
3. 如果冒烟测试也卡：临时设 `LOAD_IN_4BIT=false` + `MAX_COMPLETION_LENGTH=512` 复现 canary 成功路径，逐步恢复参数定位根因
4. 若 ZeRO-2 OOM：确认是 optimizer state 问题，再加 CPU offload 到 ds_zero2.json
5. 若成功：对比 GRPO 训后 reward 与 SFT baseline（0.211）
6. 若需要加速 generation：ms-swift 4.1.0.dev0 可能已修复 vllm server 兼容性，尝试 `USE_VLLM=true` + vllm 0.19.0
