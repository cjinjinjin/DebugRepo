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
- **状态**：✅ 跑通（已运行 10/34 steps，checkpoint-10 已保存），但存在严重的训练效果问题
- **显存**：每卡 ~69.3GB（step 1-2）→ 70.07GB（step 3+），8 卡全部正常，NCCL Init COMPLETE ✓
- **对应脚本**：`run_grpo_stable_canary.sh`
- **训练曲线（10 steps 完整数据）**：

  | Step | Reward | Loss | KL | clipped_ratio | frac_reward_zero_std | step_time(s) |
  |------|--------|------|----|---------------|----------------------|-------------|
  | 1 | 0.0911 | -0.0881 | 0.0 | 1.0 | 0.03125 | 6388 |
  | 2 | 0.0704 | -0.1090 | 0.0 | 1.0 | 0.0625 | 6248 |
  | 3 | 0.0394 | -0.0840 | 0.0093 | 1.0 | 0.09375 | 6148 |
  | 4 | 0.0994 | 0.0051 | 0.0079 | 1.0 | 0.03125 | 6081 |
  | 5 | 0.0772 | -0.1058 | 0.0076 | 1.0 | 0.0625 | 6120 |
  | 6 | 0.0839 | -0.0387 | 0.0083 | 0.984375 | 0.03125 | 6090 |
  | 7 | 0.0621 | 0.1919 | 0.0079 | 1.0 | 0.046875 | 6141 |
  | 8 | 0.0659 | 0.0383 | 0.0078 | 1.0 | 0.109375 | 6145 |
  | 9 | 0.0890 | -0.0398 | 0.0082 | 1.0 | 0.09375 | 6223 |
  | 10 | 0.1044 | 0.0709 | 0.0075 | 1.0 | 0.03125 | 6163 |

- **问题 1：Reward 无上升趋势**：
  - 10 步 reward 在 0.039~0.104 之间波动，无明确上升趋势（均值 ~0.075）
  - 远低于 SFT baseline（0.211），经过 10 步训练仍未恢复到 baseline 水平
  - GRPO 训练正常情况下应该看到 reward 逐步上升

- **问题 2：completions/clipped_ratio ≈ 1.0（根因）**：
  - 几乎所有 step 的 clipped_ratio = 1.0（仅 step 6 为 0.984），mean_length = max_length = 512
  - **说明模型在 512 tokens 内根本无法完成回复，所有生成均被截断**
  - 截断的不完整回复无法获得合理的 format/quality reward
  - reward signal 噪声大、信息量低，模型无法学到"好回复"的信号
  - 这是 reward 不涨的根本原因

- **问题 3：KL 极低且平坦**：
  - Step 1-2 KL=0.0（policy 与 reference 完全一致）
  - Step 3+ KL 稳定在 ~0.008，几乎不增长
  - 说明策略更新幅度极小，模型几乎没有从 reference policy 偏移

- **问题 4：Loss 剧烈波动**：
  - Loss 从 -0.109 到 +0.192，正负交替，说明训练信号非常嘈杂
  - 与问题 2 一致：截断回复给出的 reward 方差大但信息量低

- **问题 5：frac_reward_zero_std 偶尔偏高**：
  - Step 8 达到 0.109375（~11% 的 prompt 所有 generation 获得相同 reward）
  - 当一个 prompt 的所有 generation 都被截断且 reward 相同时，该 prompt 对梯度无贡献（GRPO 依赖 group 内 reward 方差来学习）

- **速度**：~6100-6400s/step（~1.7h/step），34 steps 预计 **4天21小时**
- **已保存 checkpoint**：`checkpoint-10`（`/vc_data/.../grpo_zero2_qlora_novllm_canary_len2048_comp512_gen2/v3-20260403-014147/checkpoint-10`）

- **修复方案**：将 `MAX_COMPLETION_LENGTH` 从 512 提高到 1024
  - `run_grpo_stable_canary.sh` 已更新：`MAX_COMPLETION_LENGTH="1024"`，实验名改为 `..._comp1024_gen2`
  - 期望：clipped_ratio 显著下降，模型能生成完整回复，reward signal 更干净，训练出现上升趋势
  - 风险：completion 变长会增加 generation 时间和显存，step_time 可能从 ~6100s 增加到 ~8000-10000s
  - 如果 1024 仍不够（clipped_ratio 仍很高），考虑提高到 2048 并相应增大 MAX_LENGTH

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
| `run_grpo_stable_canary.sh` | `MAX_COMPLETION_LENGTH` 从 512 提高到 1024；实验名改为 `..._comp1024_gen2`（2026-04-04，修复 clipped_ratio=1.0 问题）|
| `setup_envs.sh` | 新增 swift_train 环境重建脚本：ms-swift 4.1.0.dev0 + torch 2.8.0 + vllm 0.19.0 + trl 0.28.0 + transformers 4.57.6 |

### 环境重建记录（2026-04-02）
通过 `bash setup_envs.sh swift_train` 从零重建 swift_train conda 环境：
- **旧版本**：ms-swift 4.0.3, torch 2.6.0, deepspeed 0.18.8, vllm 0.11.0, trl 未知, transformers 4.46.3
- **新版本**：ms-swift 4.1.0.dev0 (GitHub main), torch 2.8.0, vllm 0.19.0, trl 0.28.0, transformers 4.57.6
- **动机**：旧环境 trl/vllm/transformers 版本冲突严重，无法正常 import GRPOTrainer
- **验证**：`from trl import GRPOTrainer` 通过

---

### 15. ZeRO-2 + QLoRA + 无 vllm — NCCL ALLREDUCE 超时（2026-04-03）
- **配置**：`run_grpo_stable_scaleup.sh`，ZeRO-2 + QLoRA 4bit，`USE_VLLM=false`，`MAX_COMPLETION_LENGTH=1024`
- **现象**：训练启动正常，preflight 通过，但第一个 step 的 generation 阶段 NCCL ALLREDUCE 在 600s 超时
- **错误**：`WorkNCCL(SeqNum=1, OpType=ALLREDUCE, Timeout(ms)=600000)` + `NCCL operation timed out after 600000 ms`
- **根因分析（深入研究后确认）**：
  - **Generation 非同步问题**：8 个 GPU 独立运行 `model.generate()`，各自随机采样（top_k=50, temperature=0.7），产生不同长度的输出
  - 快的 GPU 先完成 generation，进入 gradient allreduce 阶段等待；慢的 GPU 还在 generate，超过 600s 默认超时
  - 这不是 allreduce 本身的问题，而是 generation 耗时差异导致的等待超时
  - BNB 4bit 反量化比 BF16 慢 2-3x，加剧了 generation 时间
  - ms-swift issue #6029 和 trl issue #3119 确认这是 GRPO + 大模型的已知问题
- **修复**：
  - 添加 `--ddp_timeout 7200` 到 swift 命令（HuggingFace Trainer 级别的 NCCL 超时）
  - 设置 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200`（PyTorch watchdog 超时）
  - 设置 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`（不能与 `TORCH_NCCL_BLOCKING_WAIT` 共用）
  - 降低 `MAX_COMPLETION_LENGTH` 从 1024 到 512 减少 generation 时间差异
- **重要发现**：`TORCH_NCCL_BLOCKING_WAIT` 和 `TORCH_NCCL_ASYNC_ERROR_HANDLING` 在某些 PyTorch 版本中互斥
- **结论**：Plan B（ZeRO-2 + QLoRA + 无 vllm）理论上可跑通，但每 step ~30-60 分钟，作为验证方案可用；需要 Plan A 加速

### 16. VLLM_MODE 环境变量泄露导致 colocate 模式误启动（2026-04-03）
- **现象**：`run_grpo_stable_scaleup.sh` 设了 `USE_VLLM=false`，但 debug 日志显示 `--use_vllm true --vllm_mode colocate`
- **根因**：
  - `train_swift_grpo.sh` 的 preset `stable_grpo_zero2_qlora` 默认 `VLLM_MODE_DEFAULT="colocate"`
  - VLLM_MODE 被设置后传给 swift 命令行，swift 忽略了 `USE_VLLM=false` 而读取了 `VLLM_MODE`
  - 此外 shell 环境可能残留旧的 `export USE_VLLM=true`，覆盖了默认值
- **修复**：
  1. `run_grpo_stable_scaleup.sh` 显式 `export USE_VLLM="false"` 和 `unset VLLM_MODE/VLLM_SERVER_*`
  2. `train_swift_grpo.sh` 在 `USE_VLLM=false` 分支添加 VLLM_* 环境变量清除逻辑
- **教训**：swift 会读取 VLLM_* 环境变量自行初始化，即使命令行传了 `--use_vllm false`

### 17. vLLM 0.8.5 FusedMoE `_load_w2` bug — Plan A server 模式失败（2026-04-03，新机器）
- **配置**：Plan A — `start_rollout_server.sh`（GPUs 0,1, TP=2）+ `run_grpo_server_mode.sh`（GPUs 2-7）
- **Server 端错误**：
  ```
  IndexError: start (0) + length (384) exceeds dimension size (1)
  ```
  发生在 `vllm/model_executor/layers/fused_moe/layer.py` 的 `FusedMoE._load_w2` 方法
- **根因**：vLLM 0.8.5 的 `_load_w2` 对 Qwen3MoE 的 expert weight scale tensor 调用 `narrow()`，但 scale tensor 维度为 1（未分片），`narrow(0, 384)` 越界
- **Trainer 端连锁错误**：server crash 后 trainer 侧 NCCL ALLGATHER 等待权重同步，7200s 超时
  ```
  WorkNCCL(SeqNum=19, OpType=ALLGATHER, Timeout(ms)=7200000)
  ```
- **修复方案**：升级 vllm 到 0.10.2+（~~已确认 v0.9.0+ 修复了此 bug~~ 实际未修复，见 #18）
- **版本约束发现**：
  - trl 0.28.0 要求 `vllm>=0.10.2,<0.13.0`，所以 vllm 0.9.2 不满足
  - vllm 0.10.2 需要 torch 2.8.0，torch 2.8.0+cu126 可用（CUDA driver 12080 兼容）
  - ~~最终确定升级路径：**vllm 0.8.5 → 0.10.2, torch 2.6.0 → 2.8.0+cu126**~~ 0.10.2 仍有 bug，见 #18

### 两步走策略确立（2026-04-03）
- **Plan B（保底，原机器）**：ZeRO-2 + QLoRA + `USE_VLLM=false`，超时 7200s，`MAX_COMPLETION_LENGTH=512`
  - 脚本：`run_grpo_stable_scaleup.sh`
  - 预计速度：~30-60 分钟/step，慢但能验证训练逻辑正确性
- **Plan A（目标，新机器）**：外部 vLLM server（TP=2, GPUs 0-1）+ ZeRO-2 + QLoRA 训练（GPUs 2-7）
  - 脚本：`start_rollout_server.sh` + `run_grpo_server_mode.sh`
  - 预计速度：generation 快 10-50x
  - **阻塞项**：vLLM 版本升级（0.8.5 → ~~0.10.2~~ 0.19.0），已更新 `setup_envs.sh`

### 18. vLLM 0.10.2 仍未修复 FusedMoE._load_w2 bug（2026-04-04，新机器第二次尝试）
- **配置**：Plan A — `start_rollout_server.sh`（GPUs 0,1, TP=2）+ `run_grpo_server_mode.sh`（GPUs 2-7）
- **环境**：torch 2.8.0+cu126, vllm 0.10.2（从 0.8.5 升级后重建）
- **现象**：与 #17 完全相同 — `start (0) + length (384) exceeds dimension size (1)`
- **根因深挖**：
  - vllm 0.10.2~0.18.1 只有 PR #33173 的 `ndim > 0` guard，保护的是 0 维标量 tensor
  - Qwen3MoE 的问题 tensor 是 ndim > 0 但 shard 维度 size=1，guard 被绕过，narrow() 仍然越界
  - **真正修复在 vllm 0.19.0**：PR #37010（2026-03-31 合并）添加了完整的 bounds checking：
    - 计算 `available = loaded_weight.shape[shard_dim] - start_offset`
    - `available <= 0` 时 early return
    - 使用 `min(shard_size, available)` 调用 narrow
    - 新增 `_narrow_expert_data_for_padding()` 处理 padded hidden dimensions
- **版本追溯（完整）**：

  | vllm 版本 | torch 要求 | _load_w2 bug | 说明 |
  |----------|-----------|-------------|------|
  | 0.8.5 | 2.6.0 | **有 bug** | 原始版本 |
  | 0.10.2 | 2.8.0 | **有 bug** | 无修复 |
  | 0.12.0~0.13.0 | 2.9.0 | **有 bug** | 无修复 |
  | 0.14.0~0.16.x | 2.9.1 | **有 bug** | 仅 ndim>0 guard (PR #33173)，不够 |
  | 0.17.0~0.18.1 | 2.10.0 | **有 bug** | 同上 |
  | **0.19.0** | **2.10.0** | **修复** | PR #37010，完整 bounds checking |

- **修复方案**：升级到 vllm 0.19.0 + torch 2.10.0+cu126
  - torch 2.10.0+cu126 在 PyTorch 官方 whl 仓库可用（Python 3.10, Linux x86_64）✓
  - CUDA driver 12080 兼容 cu126 ✓
  - trl 0.28.0 要求 vllm<0.13.0 — 与 0.19.0 冲突 → 安装顺序：先装 trl，最后装 vllm 覆盖
  - ms-swift GRPO 不走 trl 的 vllm 集成，trl 版本限制不影响实际运行

---

## 环境修改记录（补充 2026-04-03~04）

| 文件 | 修改内容 |
|---|---|
| `train_swift_grpo.sh` | 添加 `--ddp_timeout 7200`；添加 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200`；移除 `TORCH_NCCL_BLOCKING_WAIT`；添加 VLLM_* 环境变量清除；添加 server 模式 `--vllm_server_timeout` 透传 |
| `run_grpo_stable_scaleup.sh` | 添加 `export USE_VLLM="false"`；降低 `MAX_COMPLETION_LENGTH` 到 512；`unset VLLM_MODE/VLLM_SERVER_*`；改实验名 |
| `run_grpo_server_mode.sh` | **新建**：Plan A trainer 端启动脚本，GPUs 2-7，连接 vLLM server |
| `start_rollout_server.sh` | **新建**：Plan A vLLM rollout server 启动脚本，`swift rollout`，TP=2 |
| `ds_zero2.json` | 还原为原始版本（移除临时添加的 `communication_data_type` 和 `comms_config`） |
| `setup_envs.sh` | swift_train 环境升级：torch 2.6.0+cu124 → 2.10.0+cu126, vllm 0.8.5 → 0.19.0（FusedMoE._load_w2 TP 修复 PR #37010；中间版本 0.10.2~0.18.1 均未修复） |

### 环境升级记录（2026-04-04，第三次 — 最终版）
`setup_envs.sh` swift_train 环境升级：
- **旧版本**：torch 2.8.0+cu126, vllm 0.10.2（第二次升级，仍有 _load_w2 bug）
- **新版本**：torch 2.10.0+cu126, vllm 0.19.0
- **其他不变**：ms-swift 4.1.0.dev0 (GitHub main), deepspeed (latest), trl 0.28.0, transformers 4.57.6, bitsandbytes (latest)
- **动机**：vllm 0.10.2 仍有 Qwen3MoE FusedMoE._load_w2 TP>1 bug，真正修复在 0.19.0 (PR #37010)
- **兼容性**：torch 2.10.0+cu126 兼容 CUDA driver 12080 ✓；trl 0.28.0 限制 vllm<0.13.0 但 ms-swift GRPO 不走 trl vllm 集成

---

## 2026-04-04: DPO Format-Preference 训练 OOM 排查与修复

### 背景
GRPO 在 Qwen3-30B-A3B (MoE) 上因 ZeRO-3 allgather 死锁无法跑通，改用 DPO format-preference 训练替代。

### DPO 数据管线已完成
- `prepare_dpo_format.py`：12 种 corruption 策略合成格式错误的 rejected（missing tags, think violations, structural violations, length violations, repetition）
- `combine_dpo_data.py`：合并 format DPO (~1800 条) + quality DPO (74 条)
- 输出：`dpo_combined_train_cot.jsonl` + `dpo_combined_eval_cot.jsonl`
- 每条 pair 用 `reward_fn()` 交叉验证 chosen reward > rejected reward

### ms-swift API 变更
- `--train_type` 在新版 ms-swift 中已改为 `--tuner_type`
- 报错：`ValueError: remaining_argv: ['--train_type', 'lora']`
- 修复：所有脚本统一改为 `--tuner_type lora`

### 清理 14b/27b 脚本
- 已删除 10 个不再使用的 14b/27b 训练/评估/推理脚本

### DPO 训练 OOM
- **环境**：8×80G GPU
- **症状**：`torch.OutOfMemoryError: CUDA out of memory`，即使停掉其他 infer 进程（释放 ~10G/卡）仍然 OOM
- **原因分析**：
  - `ds_zero3.json` 原始配置没有 optimizer offload，optimizer states 全在 GPU 上
  - 原始 DPO 只有 74 条数据能跑通，但当前 ~1900 条数据量不是 OOM 直接原因（逐 batch 处理）
  - DPO 需要同时保留 reference model + policy model 的 logprobs，显存需求比 SFT 大
  - `max_length=8192` + DPO 双模型 forward 是显存峰值的主要来源
- **修复**：在 `ds_zero3.json` 中添加 `offload_optimizer`：
  ```json
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  }
  ```
- **关于 max_length**：曾考虑降到 4096，但统计 quality DPO 数据字符长度（mean=10711, p90=15742）后发现会截断大量样本，保持 8192
- **状态**：已提交推送（commit `2239a95`），待在训练机上重试

### DPO 训练数据路径解析错误
- **报错**：`Exception: Invalid repo_id: dataset, must be of format namespace/name`
- **根因**：`train_swift_dpo.sh` 中 `DATA_DIR="./data"` 是相对路径，ms-swift 的 DPO `_get_dataset` 将其误认为 ModelScope repo ID
- **修复**：`DATA_DIR` 改为基于脚本目录的绝对路径：
  ```bash
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  DATA_DIR="${SCRIPT_DIR}/data"
  ```
- 同步修复了 `train_swift_cot.sh`
- **状态**：已提交推送（commit `8110f3b`），待在训练机上重试

### DPO 训练当前状态
- OOM 修复（offload_optimizer）+ 路径修复（绝对路径）已推送
- 等待训练机 `git pull` 后重试 `bash train_swift_dpo.sh`

---

## 2026-04-04: Inference + Constrained Decoding 排查

### SFT 模型格式合规率基线
- SFT checkpoint-50 merged model 的格式合规率约 **30%**（用户实测）
- 70% 的输出格式不合规（缺少 tag、think block 问题等）
- 这是 DPO format-preference 训练的动机：提升格式合规率

### Constrained Decoding 方案演变

**方案 1：outlines RegexLogitsProcessor（原实现，失败）**
- 使用 `outlines` 库的 `RegexLogitsProcessor`，在 decode 时通过 FSM 约束 token 生成
- **问题**：原始正则 `[\s\S]{10,3000}` 中大范围量词导致 FSM 状态爆炸
  - Qwen3-30B-A3B 词表 ~150K tokens
  - FSM 状态数 = O(量词范围 × 词表大小)，编译时间极长
  - 实际现象：卡在 "Initializing constrained decoding with outlines ..." 一整夜无响应

**方案 2：简化正则 + outlines（当前实现）**
- 将 `[\s\S]{10,3000}` 替换为 `[^<]+(?:<(?!/tag>)[^<]*)*`
  - `[^<]+` 匹配非 `<` 字符（大多数文本 token）
  - `(?:<(?!/tag>)[^<]*)*` 允许非闭合标签的 `<` 出现
- FSM 状态数从 O(3000 × vocab_size) 降到 O(tag_count × vocab_size)
- 添加 120 秒编译超时保护（SIGALRM），超时自动 fallback 到无约束模式
- **待验证**：简化后的 pattern 在 150K 词表上 FSM 编译是否能在 120s 内完成

**放弃的方案**：
- vLLM guided decoding：vLLM 在此模型上有 FusedMoE TP bug（见 #17/#18），引入 vLLM 做 inference 环境复杂度高
- 后处理验证+重试：30B 模型单条推理慢，重试会成倍增加时间
- 纯后处理统计（不约束）：不满足需求，用户需要的是推理时实时提升合规率

### Inference 脚本改动（`inference.py`）
- 简化 `CONSTRAINED_PATTERN` 正则
- `_init_constrained_decoding()` 添加 120s 超时保护
- Batch 模式输出格式合规率统计 + 每条结果附带 `format_compliant` 字段
- 保留 `--constrained` flag，不影响无约束推理

---

## 待办
1. ~~在新机器上执行环境升级（0.10.2）~~（已完成但 bug 未修复）
2. 在新机器上重建环境：vllm 0.19.0 + torch 2.10.0+cu126
3. 验证 vllm 0.19.0 + Qwen3MoE TP=2 能否正常启动 `swift rollout`
4. 重跑 Plan A：`start_rollout_server.sh` + `run_grpo_server_mode.sh`
5. **重跑方案十（canary）**：`MAX_COMPLETION_LENGTH=1024`，观察 clipped_ratio 是否下降、reward 是否出现上升趋势
6. 如果 Plan A 仍有问题，在原机器跑 Plan B：`run_grpo_stable_scaleup.sh`（已有超时修复）
7. 若成功：对比 GRPO 训后 reward 与 SFT baseline（0.211）
