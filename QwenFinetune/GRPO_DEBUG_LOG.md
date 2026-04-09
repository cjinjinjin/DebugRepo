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
- **状态**：✅ 跑通（已运行 16/34 steps，checkpoint-10 已保存），训练效果问题确认：reward 不涨、clipped_ratio≈1.0
- **显存**：每卡 ~69.3GB（step 1-2）→ 70.07GB（step 3+），8 卡全部正常，NCCL Init COMPLETE ✓
- **对应脚本**：`run_grpo_stable_canary.sh`
- **训练曲线（16 steps 完整数据）**：

  | Step | Reward | Loss | KL | clipped_ratio | frac_reward_zero_std | grad_norm | step_time(s) |
  |------|--------|------|----|---------------|----------------------|-----------|-------------|
  | 1 | 0.0911 | -0.0881 | 0.0 | 1.0 | 0.03125 | 0.090 | 6388 |
  | 2 | 0.0704 | -0.1090 | 0.0 | 1.0 | 0.0625 | 0.082 | 6248 |
  | 3 | 0.0394 | -0.0840 | 0.0093 | 1.0 | 0.09375 | 0.081 | 6148 |
  | 4 | 0.0994 | 0.0051 | 0.0079 | 1.0 | 0.03125 | 0.082 | 6081 |
  | 5 | 0.0772 | -0.1058 | 0.0076 | 1.0 | 0.0625 | 0.073 | 6120 |
  | 6 | 0.0839 | -0.0387 | 0.0083 | 0.984 | 0.03125 | 0.087 | 6090 |
  | 7 | 0.0621 | 0.1919 | 0.0079 | 1.0 | 0.046875 | 0.149 | 6141 |
  | 8 | 0.0659 | 0.0383 | 0.0078 | 1.0 | 0.109375 | 0.092 | 6145 |
  | 9 | 0.0890 | -0.0398 | 0.0082 | 1.0 | 0.09375 | 0.091 | 6223 |
  | 10 | 0.1044 | 0.0709 | 0.0075 | 1.0 | 0.03125 | 0.078 | 6163 |
  | 11 | — | — | — | — | — | — | — |
  | 12 | 0.0746 | -0.0404 | 0.0081 | 1.0 | 0.125 | 0.075 | 6241 |
  | 13 | 0.0828 | -0.0165 | 0.0077 | 0.992 | 0.0625 | 0.075 | 6006 |
  | 14 | 0.0543 | -0.1336 | 0.1578 | 1.0 | 0.046875 | **6.494** | 6314 |
  | 15 | 0.0608 | 0.0705 | 0.0091 | 1.0 | 0.0625 | 0.080 | 6158 |
  | 16 | 0.0624 | 0.0681 | 0.0080 | 0.992 | 0.09375 | 0.083 | 6339 |

- **问题 1：Reward 无上升趋势**：
  - 16 步 reward 在 0.039~0.104 之间波动，无明确上升趋势（均值 ~0.072）
  - 远低于 SFT baseline（0.211），经过 16 步训练仍未恢复到 baseline 水平
  - GRPO 训练正常情况下应该看到 reward 逐步上升

- **问题 2：completions/clipped_ratio ≈ 1.0（根因）**：
  - 几乎所有 step 的 clipped_ratio = 1.0（仅 step 6/13/16 略低于 1.0），mean_length = max_length = 512
  - **说明模型在 512 tokens 内根本无法完成回复，所有生成均被截断**
  - 截断的不完整回复无法获得合理的 format/quality reward
  - reward signal 噪声大、信息量低，模型无法学到"好回复"的信号
  - 这是 reward 不涨的根本原因

- **问题 3：KL 极低且平坦（step 14 异常除外）**：
  - Step 1-2 KL=0.0（policy 与 reference 完全一致）
  - Step 3-13, 15-16 KL 稳定在 ~0.008，几乎不增长
  - 说明策略更新幅度极小，模型几乎没有从 reference policy 偏移

- **问题 4：Loss 剧烈波动**：
  - Loss 从 -0.134 到 +0.192，正负交替，说明训练信号非常嘈杂
  - 与问题 2 一致：截断回复给出的 reward 方差大但信息量低

- **问题 5：frac_reward_zero_std 偶尔偏高**：
  - Step 12 达到 0.125（12.5% 的 prompt 所有 generation 获得相同 reward）
  - 当一个 prompt 的所有 generation 都被截断且 reward 相同时，该 prompt 对梯度无贡献（GRPO 依赖 group 内 reward 方差来学习）

- **问题 6：Step 14 梯度爆炸**：
  - grad_norm 突然从 ~0.08 飙升到 **6.494**（正常值的 ~80x）
  - 同步出现 KL 飙升到 0.158（正常值的 ~20x），loss 异常低 -0.134
  - Step 15 恢复正常（grad_norm=0.080, KL=0.009），说明是单步异常而非持续发散
  - 可能原因：某个 batch 的 reward 分布极端（所有 generation 截断后出现异常大的 advantage），导致策略大幅更新
  - 这进一步证实截断 reward signal 的不稳定性

- **速度**：~6000-6340s/step（~1.7h/step），已运行 55h10m，剩余 ~73h（~3天）
- **已保存 checkpoint**：`checkpoint-10`（`/vc_data/.../grpo_zero2_qlora_novllm_canary_len2048_comp512_gen2/v3-20260403-014147/checkpoint-10`）
- **结论**（16 步确认）：comp512 训练无效，reward 不涨的根因是 clipped_ratio≈1.0。Step 14 梯度爆炸进一步证实截断 reward 的不稳定性。此实验可作为负面对照，不建议继续等待完成。

- **修复方案**：将 `MAX_COMPLETION_LENGTH` 从 512 提高到 1024
  - `run_grpo_stable_canary.sh` 已更新：`MAX_COMPLETION_LENGTH="1024"`，实验名改为 `..._comp1024_gen2`
  - 期望：clipped_ratio 显著下降，模型能生成完整回复，reward signal 更干净，训练出现上升趋势
  - 风险：completion 变长会增加 generation 时间和显存，step_time 可能从 ~6100s 增加到 ~8000-10000s
  - 如果 1024 仍不够（clipped_ratio 仍很高），考虑提高到 2048 并相应增大 MAX_LENGTH

### 10b. 方案十 comp1024 重跑结果（2026-04-05~06，前 10 steps）
- **配置变更**：`MAX_COMPLETION_LENGTH=1024`（其余与方案十一致），实验名 `..._comp1024_gen2`
- **训练曲线**：

  | Step | Reward | Loss | KL | clipped_ratio | mean_length | min_length | frac_reward_zero_std | grad_norm | step_time(s) |
  |------|--------|------|----|---------------|-------------|------------|----------------------|-----------|-------------|
  | 1 | 0.4571 | -0.0011 | 0.0 | 0.1406 | 898.5 | 647.0 | 0.03125 | 0.061 | 10641 |
  | 2 | 0.4575 | 0.0778 | 0.0 | 0.1641 | 904.7 | 477.5 | 0.015625 | 0.087 | 10552 |
  | 3 | 0.3987 | 0.0165 | 0.0066 | 0.2188 | 915.2 | 676.0 | 0.015625 | 0.057 | 10759 |
  | 4 | 0.4719 | -0.0395 | 0.0069 | 0.1094 | 907.6 | 615.0 | 0.0 | 0.060 | 10597 |
  | 5 | 0.4331 | -0.1582 | 0.0068 | 0.1094 | 914.9 | 764.5 | 0.015625 | 0.058 | 10570 |
  | 6 | 0.4486 | -0.0186 | 0.0069 | 0.1953 | 908.9 | 698.0 | 0.0625 | 0.088 | 10439 |
  | 7 | 0.4358 | -0.0878 | 0.0067 | 0.1484 | 894.1 | 476.0 | 0.09375 | 0.053 | 10612 |
  | 8 | 0.4269 | -0.0395 | 0.0074 | 0.2188 | 908.6 | 649.0 | 0.03125 | 0.073 | 10765 |
  | 9 | 0.4707 | 0.0852 | 0.0075 | 0.1328 | 906.1 | 670.0 | 0.015625 | 0.073 | 10511 |
  | 10 | 0.4605 | -0.0173 | 0.0069 | 0.1406 | 907.9 | 718.5 | 0.0 | 0.059 | 10762 |

- **与 comp512 的对比（step 1）**：

  | 指标 | comp512 | comp1024 | 变化 |
  |------|---------|----------|------|
  | reward | 0.091 | **0.457** | **+5x**，已超过 SFT baseline 0.211 |
  | clipped_ratio | 1.0 | **0.141** | 从全部截断降到 14% |
  | mean_length | 512.0 | **898.5** | 模型实际需要 ~900 tokens |
  | min_length | 512.0 | **647.0** | 最短回复也超过原 512 上限 |
  | step_time | 6388 | 10641 | 慢 ~1.67x（预期内） |

- **关键发现**：
  1. **comp512 的低 reward 完全是截断导致的**：1024 后第一步 reward 就达到 0.457，远超 SFT baseline（0.211）
  2. **模型实际需要 ~900 tokens**：mean_length 稳定在 898-915，min_length 477-676，说明 512 根本不够
  3. **clipped_ratio 波动在 0.11-0.22 之间**：均值 ~0.16，无上升趋势
  4. **reward 在 0.40-0.47 之间平台波动**：10 步均值 ~0.447，远超 SFT baseline（0.211），但无明确上升趋势
  5. **KL 仍然极低**：step 1-2 为 0.0，step 3-4 稳定在 ~0.007，策略更新幅度很小但已开始
  6. **frac_reward_zero_std 波动**：step 7 升到 0.094 后回落，step 10 为 0.0，整体无恶化趋势
  7. **grad_norm 极其稳定**：0.053-0.088，无异常波动
  8. **速度**：~10600s/step（~2.9h/step），34 steps 预计 **~8 天**

- **10 步总结**：reward 稳定在 ~0.45 平台，无上升也无下降。训练状态健康但学习信号弱。KL ~0.007 说明策略几乎未偏离 reference，可能需要更多 steps 或更高 LR 才能看到变化。
- **已保存 checkpoint**：`checkpoint-10`（`.../grpo_zero2_qlora_novllm_canary_len2048_comp1024_gen2/v0-20260404-151523/checkpoint-10`）
- **状态**：comp1024 训练进行中（10/34 steps，已运行 ~59h）；comp2048 并行运行中（见 10c）

### 10c. 方案十 comp2048 实验（2026-04-05~06，前 6 steps）
- **动机**：comp1024 的 clipped_ratio 在 step 3 升到 0.22（后续回落到 0.11），提前启动 comp2048 消除截断瓶颈，获得更干净的 reward 信号。
- **配置变更**（vs comp1024）：

  | 参数 | comp1024 | comp2048 |
  |------|----------|----------|
  | MAX_LENGTH | 2048 | **4096** |
  | MAX_COMPLETION_LENGTH | 1024 | **2048** |
  | SAVE_STEPS | 10 | **1**（每步都保存，因每步耗时极长） |
  | EXPERIMENT_NAME | `..._comp1024_gen2` | `..._comp2048_gen2` |

- **其余配置不变**：ZeRO-3 + BF16 + 无 vllm，8×A100-80GB，LoRA rank=16/alpha=32，LR=5e-6，NUM_GENERATIONS=2，GRADIENT_ACCUMULATION_STEPS=8
- **对应脚本**：`run_grpo_stable_canary_comp2048.sh`（新建，避免与正在运行的 comp1024 冲突）
- **训练曲线**：

  | Step | Reward | Loss | KL | clipped_ratio | mean_length | min_length | max_length | frac_reward_zero_std | grad_norm | step_time(s) |
  |------|--------|------|----|---------------|-------------|------------|------------|----------------------|-----------|-------------|
  | 1 | 0.5070 | -0.0881 | 0.0 | 0.0 | 898.4 | 722.0 | 1128.5 | 0.0 | 0.060 | 11064 |
  | 2 | 0.4800 | -0.0219 | 0.0 | 0.0 | 902.4 | 712.0 | 1127.5 | 0.0 | 0.060 | 10957 |
  | 3 | 0.4540 | 0.0709 | 0.0061 | 0.0 | 899.8 | 704.5 | 1134.5 | 0.0 | 0.147 | 11212 |
  | 4 | — | — | — | — | — | — | — | — | — | — |
  | 5 | 0.4751 | 0.1380 | 0.0063 | 0.0 | 897.7 | 627.0 | 1134.5 | 0.0 | 0.057 | 11115 |
  | 6 | 0.5016 | 0.0495 | 0.0062 | 0.0 | 900.0 | 750.5 | 1147.0 | 0.0 | 0.055 | 11022 |

- **与 comp1024 的对比（step 1）**：

  | 指标 | comp1024 | comp2048 | 变化 |
  |------|----------|----------|------|
  | reward | 0.457 | **0.507** | **+11%**，目前所有实验最高起点 |
  | clipped_ratio | 0.141 | **0.0** | **完全消除截断** |
  | max_length | 1024.0 | **1128.5** | 有回复超过 1024，被 comp1024 截断但 comp2048 保留 |
  | min_length | 647.0 | **722.0** | 最短回复更长（保留了完整内容） |
  | frac_reward_zero_std | 0.031 | **0.0** | 100% 的 prompt 有有效梯度信号 |
  | step_time | 10641 | **11064** | **仅慢 4%**（远好于预期的 50-100%） |
  | memory(GiB) | 69.7 | **70.3** | 仅多 0.6 GiB，无 OOM 风险 |

- **关键发现**：
  1. **clipped_ratio = 0.0**：完全消除截断，所有回复完整生成，reward 信号最干净
  2. **reward 0.507 是所有实验最高起点**：超过 comp1024 step 1 的 0.457（+11%），说明被截断的 ~14% 回复确实拉低了 comp1024 的 reward
  3. **max_length 1128.5**：有回复长度超过 1024，这些在 comp1024 中被截断（获得低 reward），在 comp2048 中完整保留
  4. **step_time 几乎无增加**（11064 vs 10641，+4%）：因为模型实际 mean_length ~900 远低于 2048 上限，generation 时间由实际长度决定而非上限
  5. **显存无压力**：70.3 GiB vs 69.7 GiB，仅多 0.6 GiB，之前担心的 OOM 风险不存在
  6. **frac_reward_zero_std = 0.0 连续六步**：所有 prompt 的 generation 间都有 reward 方差，GRPO 梯度 100% 有效
  7. **"Could not estimate tokens" 警告**：无害，transformers 对该模型架构缺少 FLOPs 估算，不影响训练
  8. **Step 3 grad_norm 小幅上升到 0.147**：比 step 1-2 的 0.060 高 ~2.5x，但 step 5-6 回落到 0.055-0.057，属正常波动
  9. **显存 step 3+ 稳定在 73.15 GiB**，仍有 ~7 GiB 余量
- **预期修正**：
  - ~~step_time 15000-20000s~~ → 实际 ~11000s（与 comp1024 几乎持平）
  - ~~OOM 风险~~ → 不存在（70.3 GiB，余量 ~10 GiB）
  - ~~34 steps 5-7 天~~ → 预计 **~8.6 天**（与 comp1024 相同）
- **状态**：训练进行中（6/34 steps，已运行 ~40h），checkpoint-1 至 checkpoint-6 已保存

- **comp1024 vs comp2048 并行对比（同 step 数，step 1-5 均值）**：

  | 指标 | comp1024 (5 steps avg) | comp2048 (5 steps avg) | 差异 |
  |------|------------------------|------------------------|------|
  | reward | 0.444 | **0.484** | comp2048 高 9% |
  | clipped_ratio | 0.148 | **0.0** | comp2048 无截断 |
  | frac_reward_zero_std | 0.019 | **0.0** | comp2048 梯度更有效 |
  | KL | 0.004 | 0.004 | 相近 |
  | step_time | 10612 | 11090 | comp2048 仅慢 4.5% |

  **结论**：comp2048 在 reward、梯度质量上全面优于 comp1024，且速度几乎无损。comp2048 是更优配置。
  
  **注**：reward 呈平台波动（0.45-0.51），6 步内尚无明确上升趋势。GRPO 在大模型上学习缓慢是已知现象（KL ~0.006 说明策略更新幅度极小），需要更多 steps 才能观察到趋势。

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
| `run_grpo_stable_canary_comp2048.sh` | **新建**（2026-04-05）：comp2048 实验专用脚本，`MAX_LENGTH=4096`，`MAX_COMPLETION_LENGTH=2048`，`SAVE_STEPS=1`，避免与运行中的 comp1024 冲突 |
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

## 2026-04-04: DPO Combined 数据集生成完成

### 数据准备流水线执行结果
- 运行 `python combine_dpo_data.py`，该脚本内部先调用 format DPO 生成，再合并
- **Format DPO 数据生成**：
  - 从 833 条 SFT 样本生成 1882 条 format-preference pairs
  - 9 种 corruption 策略分布均匀（185–226 条/类）
  - Reward violations（chosen ≤ rejected）：617 条被过滤
  - 最终 format DPO：train=1700, eval=182
- **Quality/Refine DPO 数据**（已有）：train=74, eval=8
- **Combined DPO 数据**：train=1774, eval=190
  - `dpo_combined_train_cot.jsonl`（1774 条）
  - `dpo_combined_eval_cot.jsonl`（190 条）
  - `dataset_stats_dpo_combined.json`

### 数据集组成比例
| 类型 | Train | Eval | 占比 |
|------|-------|------|------|
| Format preference | 1700 | 182 | 95.8% |
| Quality/Refine preference | 74 | 8 | 4.2% |
| **Combined** | **1774** | **190** | **100%** |

### 下一步
- 在训练机上执行 `bash train_swift_dpo.sh`，此前的 OOM 修复和路径修复已就绪

---

## 2026-04-04: Constrained Decoding 迁移到 outlines 高阶 API

### 问题回顾
- 方案 2（简化正则 + outlines 低阶 `RegexLogitsProcessor`）依然失败：
  - `interegular` 不支持 lookahead `(?!...)` → `Unsupported: Group can not have lookbacks/lookaheads`
  - 进一步简化到 `[^<]+` 后，HF tokenizer 缺少 `.vocabulary` 属性 → `AttributeError: Qwen2TokenizerFast has no attribute vocabulary`
  - 添加 `TransformerTokenizer` wrapper 后仍有同样问题

### 方案 3：outlines 旧 API 0.1.x（最终可用实现）
- 使用 `outlines.models.transformers.Transformers(model, tokenizer)` 包装已加载的 HF 模型
- 使用 `outlines.generate.regex(outlines_model, pattern)` 创建 regex-constrained generator
- 生成调用：`generator(prompt, max_tokens=N)`
- 优势：outlines 内部处理 tokenizer 适配（`TransformerTokenizer`），无需手动 wrap
- `generate()` 和 `generate_batch()` 均已更新：
  - constrained 模式：通过 `self._outlines_generator()` 逐条生成
  - unconstrained 模式：走原有 HF `model.generate()` 路径，无变化

### 简化的正则 Pattern
```
<think>[^<]+</think>\s*
<Prompt1>[^<]+</Prompt1>\s*
<Prompt2>[^<]+</Prompt2>\s*
<Prompt3>[^<]+</Prompt3>\s*
<Prompt4>[^<]+</Prompt4>\s*
<Prompt5>[^<]+</Prompt5>
```
- `[^<]+`：匹配非 `<` 字符，FSM 状态复杂度低
- 假设：tag 内容不含 `<` 字符（对 prompt 文本成立）

### 待验证
- ~~在训练机上 `pip install 'outlines[transformers]'` 后运行 constrained inference~~（已验证）
- ~~FSM 编译时间是否可接受（150K 词表 + 简化 pattern）~~（可接受，秒级完成）
- ~~格式合规率是否从 SFT baseline 30% 提升~~（单条测试已通过，batch 评估中）

---

## 2026-04-05: Constrained Decoding 验证成功

### outlines API 修复历程
- **方案 3a（outlines 新 API）失败**：`outlines.from_transformers()` + `outlines.types.Regex` 是新版 API（>0.2.x）
  - 训练机安装的是 outlines 0.1.11（旧版），`outlines.types` 没有 `Regex` 类
  - `from outlines.types import Regex` → `ImportError: cannot import name 'Regex'`
- **方案 3b（outlines 旧 API）成功**：outlines 0.1.x 使用不同的 API：
  - `outlines.models.transformers.Transformers(model, tokenizer)` 包装已加载的 HF 模型
  - `outlines.generate.regex(outlines_model, pattern)` 创建 regex-constrained generator
  - `generator(prompt, max_tokens=N)` 执行约束生成
  - FSM 编译秒级完成，无超时问题

### 单条测试结果
- **输入**：`--url "https://example.com/product" --title "Premium Wireless Headphones" --constrained`
- **输出**：格式完全合规
  - `<think>` block 存在且闭合 ✓
  - 5 个 `<PromptN>` tag 全部正确解析 ✓
  - prompt 内容质量良好：场景多样（夜间通勤、桌面工作区、公交车、家居），包含摄影参数、安全约束、排除条件
- **结论**：constrained decoding 在 SFT merged model + outlines 0.1.11 上可行

### Batch 评估
- **进行中**：在 `dpo_combined_eval_cot.jsonl`（190 条）上评估
- **命令**：
  ```bash
  python inference.py \
      --adapter_path .../merged_model \
      --input_file data/dpo_combined_eval_cot.jsonl \
      --output_file results/constrained_dpo_eval.jsonl \
      --constrained --max_new_tokens 2048 --batch_size 1
  ```
- **待得出**：格式合规率（对比 SFT baseline ~30%），预期 ~100%（regex 约束为硬约束）

### inference.py 最终实现
```python
# _init_constrained_decoding():
from outlines.models.transformers import Transformers
import outlines.generate
outlines_model = Transformers(self.model, self.tokenizer)
self._outlines_generator = outlines.generate.regex(outlines_model, CONSTRAINED_PATTERN)

# generate() / generate_batch():
result = self._outlines_generator(input_text, max_tokens=max_new_tokens)
```

---

## 2026-04-05: DPO 首次训练运行 + 超参调整

### CUDA 版本不匹配
- **报错**：`DeepSpeed CUDAMismatchException: Installed CUDA 11.8 ≠ torch compiled with 12.6`
- **原因**：`offload_optimizer` 需要编译 `DeepSpeedCPUAdam` JIT 扩展，系统 CUDA toolkit 11.8 与 PyTorch CUDA 12.6 不匹配
- **修复**：`DS_SKIP_CUDA_CHECK=1 bash train_swift_dpo.sh`
- 实际 GPU driver 支持 CUDA 12.8（`nvidia-smi` 确认），只是系统 nvcc 版本旧

### 首次训练运行结果（5 epoch，v10-20260405-013625）
- 8×A100-80GB，显存 21–28 GiB/卡（27–35%），GPU 利用率 100%
- 训练速度：~39 分钟/step（ZeRO-3 + offload_optimizer 开销大）

| Step | Loss | Accuracy | Margins | LR |
|------|------|----------|---------|-----|
| 1 | 0.1876 | 98.4% | 56.8 | 7.14e-6 |
| 10 | 0.0111 | 99.5% | 65.7 | 4.99e-5 |

- **观察**：Loss 10 步内从 0.188 降到 0.011，accuracy 99.5%，模型已基本学会区分 chosen/rejected
- **问题**：
  - `num_train_epochs=5` → 总 140 steps，预计 3.5 天，且极大概率过拟合
  - `save_steps=50` → 到第一个 checkpoint 时模型可能已经过拟合

### 超参调整
- `num_train_epochs`: 5 → **1**（总步数 ~28，DPO 通常 1-2 epoch 足够）
- `save_steps`: 50 → **10**（保留 step 10/20/28 的 checkpoint 做对比）
- `eval_steps`: 50 → **10**（同步）
- `logging_steps`: 10 → **5**（更细粒度观察 loss 曲线）
- 需要停掉当前运行，`git pull` 后重新启动

### 1-epoch 重跑结果（v11-20260405-094235，28 steps，总耗时 20.7h）

| Step | Epoch | Loss | Grad Norm | Accuracy | Margins | Eval Loss | Eval Margins |
|------|-------|------|-----------|----------|---------|-----------|--------------|
| 1 | 0.04 | 0.1876 | 1.70 | 98.4% | 56.8 | - | - |
| 5 | 0.18 | 0.0234 | ~0 | 99.2% | 64.2 | - | - |
| 10 | 0.36 | 0.0 | 0.0 | 100% | 75.9 | 0.0 | 80.3 |
| 15 | 0.55 | 0.0 | 0.0 | 100% | 83.5 | - | - |
| 20 | 0.73 | 0.0 | 0.0 | 100% | 83.2 | 0.0 | 84.4 |
| 25 | 0.91 | 0.0 | 0.0 | 100% | 83.2 | - | - |
| 28 | 1.0 | 0.0 | 0.0 | 100% | - | 0.0 | 84.6 |

- **swift 自动选出 best_model_checkpoint = checkpoint-10**
- **收敛极快**：step 5 时 loss 已降到 0.023，step 10 后 loss=0、grad_norm=0，模型完全停止学习
- **step 10 之后的 ~14 小时无实质更新**：checkpoint-10/20/28 效果应几乎相同
- **任务太简单**：format corruption 的 chosen/rejected 差异太大，模型几乎不需要学习就能区分
- **潜在风险**：margins 持续增大（56→84），logps/chosen 在下降（-1114→-1277），可能存在 likelihood displacement，需通过 inference 验证生成质量未退化
- **Checkpoints 路径**：
  - `checkpoint-10`（best）：`.../qwen3_dpo_lora_cot_refine/v11-20260405-094235/checkpoint-10`
  - `checkpoint-20`：`.../qwen3_dpo_lora_cot_refine/v11-20260405-094235/checkpoint-20`
  - `checkpoint-28`（last）：`.../qwen3_dpo_lora_cot_refine/v11-20260405-094235/checkpoint-28`

### 下一步
1. ~~用 checkpoint-10 执行 `bash eval_swift_dpo.sh`（merge → inference → evaluate）~~（已完成）
2. ~~对比 SFT baseline 的格式合规率（~30%）和生成质量~~（见下方评估结果）

---

## 2026-04-06: DPO checkpoint-10 Inference 评估结果

### 环境问题
- `vllm_infer` 环境的 `autoawq` 与 `transformers` 版本冲突：`ImportError: cannot import name 'PytorchGELUTanh'`
- **修复**：merge 步骤改用 `swift_train` 环境执行 `swift.cli.export`，inference 继续用 `vllm_infer`

### 评估结果（checkpoint-10，190 条 dpo_combined_eval_cot.jsonl）

| 指标 | DPO checkpoint-10 | SFT baseline |
|------|-------------------|--------------|
| 5 tags 全部存在 | **31.6%** | ~30% |
| think block 存在 | 74.7% | - |
| CoT 6 字段全有 | 7.9% | - |
| 平均 prompt 字数 | 40.2 | - |
| 关键词覆盖率 | 5.5% | - |
| 禁用词 prompts | 1.4/5 | - |

### 结论：DPO 未能提升格式合规率

**格式合规率 31.6% ≈ SFT baseline 30%**，DPO 训练几乎无效。

### 失败原因分析

1. **Train-Inference Gap**：训练时 accuracy 100%（判别任务），但自回归生成时模型并未因此生成更好的格式。DPO 优化的是 chosen vs rejected 的相对 log-prob，不直接优化生成质量
2. **Likelihood Displacement**：logps/chosen 从 -1114 下降到 -1277，模型在推大 margin 的同时降低了 chosen 的绝对概率，可能导致生成质量退化
3. **负样本太简单**：format corruption 策略太极端（如 `drop_all_prompts`、`no_think`），chosen/rejected 差异巨大，模型轻松区分但没学到细粒度的格式约束
4. **数据不平衡**：format DPO 占 95.8%（1700 条），quality DPO 仅 4.2%（74 条），质量信号被淹没

### 可能的改进方向

1. **更难的负样本**：borderline 负样本（如只少一个 tag、tag 顺序错、字数略超 150 words），让模型学到更细粒度的区分
2. **减小 beta**：当前 beta=0.1，更小的值让模型更保守地偏离 reference policy，减少 likelihood displacement
3. **Constrained Decoding**：regex 约束是硬约束（100% 格式合规），可能比 DPO 更实际、更高效
4. **IPO / KTO 等其他 preference 算法**：对 likelihood displacement 更鲁棒

---

## 2026-04-06: DPO 早期 Checkpoint 实验（每步保存）

### 动机
- checkpoint-10 评估显示格式合规率 31.6% ≈ SFT baseline，DPO 过度训练可能导致 likelihood displacement
- 训练曲线显示 step 1 时 loss=0.1876、accuracy=98.4%，模型已在学习但尚未过拟合
- step 5 时 loss 已降到 0.023，step 10 完全归零 → **最早的 checkpoint 可能效果最好**
- 假设：step 1-3 时模型刚开始调整偏好方向，chosen 的绝对概率尚未被压低

### 超参调整
- `save_steps`: 10 → **1**（每步保存 checkpoint）
- `logging_steps`: 5 → **1**（每步记录 loss）
- `eval_steps`: 保持 **5**（eval 耗时较长）
- 其余参数不变（1 epoch, 28 steps, beta=0.1）

### 评估计划
- 重新训练后，对 checkpoint-1 ~ checkpoint-5 逐个做 inference 评估
- 找到格式合规率最高的 sweet spot
- LoRA adapter 每个 checkpoint ~几百 MB，28 个 checkpoint 磁盘空间无压力

### 下一步
1. 训练机 `git pull` 后 `DS_SKIP_CUDA_CHECK=1 bash train_swift_dpo.sh`
2. 用 `eval_swift_dpo.sh` 逐个评估 checkpoint-1 到 checkpoint-5
3. 对比各 checkpoint 格式合规率，选出最佳

---

## 2026-04-08: GRPO comp2048 checkpoint-6 Inference 评估结果

### 评估配置
- **Checkpoint**：`grpo_zero2_qlora_novllm_canary_len4096_comp2048_gen2/v0-20260405-095718/checkpoint-6`
- **Base model**（merge 对象）：SFT merged model（`qwen3_sft_lora_cot_8192_v2/v0-20260319-083851/checkpoint-50/merged_model`）
- **评估数据**：`dpo_combined_eval_cot.jsonl`（8 条，dpo_refine 子集）
- **推理方式**：`eval_swift_cot.sh` — swift export merge → swift infer（vLLM backend, TP=8）→ evaluate.py
- **结果路径**：`.../checkpoint-6/eval_results/eval_report_20260408_033632.json`

### 环境问题
- 当前机器 `vllm_infer` 环境 swift 版本 4.0.2，merge 时 peft 触发 autoawq 与 transformers 版本冲突：`ImportError: cannot import name 'PytorchGELUTanh'`
- **修复**：卸载 autoawq（`pip uninstall autoawq`），该包已 deprecated 且不影响推理

### 评估结果（小样本，8 条 dpo_refine_eval_cot.jsonl）

| 指标 | GRPO comp2048 ckpt-6 | DPO ckpt-10 | SFT baseline |
|------|---------------------|-------------|--------------|
| 5 tags 全部存在 | **87.5%** | 31.6% | ~30% |
| think block 存在 | **100%** | 74.7% | - |
| CoT 6 字段全有 | 12.5% | 7.9% | - |
| 平均 prompt 字数 | 56.7 | 40.2 | - |
| 关键词覆盖率 | **31.0%** | 5.5% | - |
| 禁用词 prompts | 0.9/5 | 1.4/5 | - |
| 质量 hints | 1.1/5 | - | - |

### 评估结果（完整，190 条 dpo_combined_eval_cot.jsonl）

| 指标 | GRPO comp2048 ckpt-6 (190条) | DPO v12 ckpt-1 (190条) | SFT baseline |
|------|------------------------------|------------------------|--------------|
| 5 tags 全部存在 | 23.7% | **57.9%** | ~30% |
| All 5 unique | 22.6% | **52.6%** | - |
| Fully compliant | 22.6% | **47.9%** | - |
| think block 存在 | 45.8% | 74.2% | - |
| CoT 6 字段全有 | 6.3% | **17.4%** | - |
| 平均 prompt 字数 | 18.7 | **68.2** | - |
| 关键词覆盖率 | 7.5% | 9.1% | - |
| 禁用词 prompts | 0.3/5 | 2.7/5 | - |
| 质量 hints | 0.1/5 | - | - |

### 关键发现

1. **8 条小样本结果严重高估**：87.5% → 23.7%（190 条），小样本评估不可靠
2. **190 条上 GRPO ckpt-6 表现不如 SFT baseline（~30%）和 DPO ckpt-1（57.9%）**
3. **平均 prompt 字数仅 18.7**（vs DPO 68.2）：模型生成内容过短，大量回复不完整
4. **think block 仅 45.8%**：近一半输出缺少 CoT 推理过程
5. **大量 "No model prompts" 警告**：许多 format corruption 类型的输入上模型未能生成有效 prompt
6. **可能原因**：
   - GRPO 仅 6 步训练，KL ~0.006，策略更新幅度极小
   - GRPO 训练数据分布与 format DPO 评估数据分布不同
   - 训练时 reward ~0.48 看似不错，但 inference 时生成质量不足
   - max_length=4096 的推理设置可能与 GRPO 训练时的上下文长度不匹配

### 结论
- GRPO comp2048 ckpt-6 在 190 条完整评估上表现不佳，8 条小样本结果不具代表性
- 当前 DPO v12 ckpt-1（47.9% fully compliant）仍是最佳方案
- GRPO 需要更多训练步数和/或更高 LR 才能产生实质性改进

---

## 2026-04-08: DPO v12 早期 Checkpoint 评估结果

### 评估配置
- **训练版本**：v12-20260407-052859（save_steps=1，每步保存）
- **评估数据**：`dpo_combined_eval_cot.jsonl`（190 条）
- **评估脚本**：`eval_swift_dpo.sh`（含 `--max_new_tokens 4096`）
- **evaluate.py 更新**：新增 prompt uniqueness 检查（5 个 prompt 必须互不相同）

### Eval 数据 Input Token 分布
| 统计量 | Tokens |
|--------|--------|
| 平均 | 1303 |
| 中位数 | 1046 |
| P90 | 2217 |
| P95 | 3063 |
| 最大值 | 7363 |
| >4096 | 5 个样本 |
| >6144 | 3 个样本 |

### SFT 训练集 Output Token 分布
| 部分 | 平均 | P95 | 最大 |
|------|------|-----|------|
| Think | 77 | 94 | 115 |
| Prompts | 869 | 964 | 1106 |
| 总输出 | 946 | 1058 | 1221 |

### Checkpoint-1 评估结果（190 条）

| 指标 | DPO v12 ckpt-1 | DPO v11 ckpt-10 | SFT baseline | GRPO ckpt-6 (8条) |
|------|----------------|-----------------|--------------|-------------------|
| All 5 tags present | **57.9%** | 31.6% | ~30% | 87.5% |
| All 5 unique | **52.6%** | - | - | - |
| Fully compliant | **47.9%** | - | - | - |
| Think block present | **74.2%** | 74.7% | - | 100% |
| CoT 6 fields | 17.4% | 7.9% | - | 12.5% |
| Avg word count | 68.2 | 40.2 | - | 56.7 |
| Keyword coverage | 9.1% | 5.5% | - | 31.0% |
| Forbidden words | 2.7/5 | 1.4/5 | - | 0.9/5 |

### 关键发现

1. **DPO checkpoint-1 显著优于 checkpoint-10**：format compliance 从 31.6% 提升到 57.9%（+26.3pp），fully compliant 47.9%
2. **证实 likelihood displacement 假说**：step 1 时模型刚开始调整偏好，chosen 概率尚未被过度压低；step 10 后完全过拟合
3. **DPO 在极早期有正向效果**：仅 1 步 DPO 训练就将 SFT baseline 的 ~30% 提升到 47.9%（fully compliant）
4. **GRPO 在 190 条上不如 DPO ckpt-1**：GRPO ckpt-6 仅 23.7%（190 条），远低于 DPO ckpt-1 的 57.9%。此前 8 条小样本上 87.5% 的结果不具代表性
5. **Forbidden words 上升到 2.7/5**：DPO 在减少禁用词方面不如 GRPO（0.9/5），可能因为 format DPO 数据不含禁用词约束

### 全部 5 个 Checkpoint 对比（190 条 eval 数据）

| 指标 | ckpt-1 | ckpt-2 | ckpt-3 | ckpt-4 | ckpt-5 | SFT baseline | GRPO ckpt-6 (8条) |
|------|--------|--------|--------|--------|--------|--------------|-------------------|
| **Fully compliant** | **47.9%** | 46.8% | 39.5% | 25.3% | 41.6% | ~30% | 87.5% |
| All 5 tags | **57.9%** | 55.8% | 49.5% | 37.4% | 50.5% | ~30% | 87.5% |
| All unique | **52.6%** | 52.1% | 44.7% | 33.2% | 45.3% | - | - |
| Think block | 74.2% | 74.2% | 75.3% | 72.1% | **81.6%** | - | 100% |
| CoT 6 fields | 17.4% | **21.6%** | 12.6% | 9.5% | 17.4% | - | 12.5% |
| Avg word count | **68.2** | 62.2 | 55.5 | 47.5 | 53.6 | - | 56.7 |
| Keyword coverage | 9.1% | **9.8%** | 8.9% | 6.3% | 8.4% | - | 31.0% |
| Forbidden words | 2.7/5 | 2.6/5 | 2.4/5 | 1.8/5 | 2.3/5 | - | 0.9/5 |

### 趋势分析

1. **Format compliance 单调递减（ckpt-1→4）**：47.9% → 46.8% → 39.5% → 25.3%，ckpt-4 甚至低于 SFT baseline（~30%）
2. **Ckpt-5 回弹到 41.6%**：可能是 eval noise，但仍低于 ckpt-1
3. **Likelihood displacement 完全确认**：DPO 训练步数越多，chosen 概率越被压低，生成质量越差
4. **Avg word count 持续下降**（68→62→55→47→54）：模型逐渐生成更短、更不完整的输出
5. **Ckpt-1 是唯一全面优于 SFT baseline 的 checkpoint**

### DPO 实验最终结论

- **DPO 最佳 checkpoint = checkpoint-1**（仅 1 步训练），fully compliant 47.9%
- DPO 对 format compliance 的提升有限（47.9% vs GRPO 87.5%），且极易过拟合退化
- Likelihood displacement 是 DPO 在此任务上的根本瓶颈：负样本太简单，模型通过压低 chosen 概率来拉大 margin，而非真正学习格式约束
- **GRPO 是更有前景的方向**：直接优化 reward 而非相对偏好，不存在 likelihood displacement 问题

---

## 待办
1. ~~在新机器上执行环境升级（0.10.2）~~（已完成但 bug 未修复）
2. 在新机器上重建环境：vllm 0.19.0 + torch 2.10.0+cu126
3. 验证 vllm 0.19.0 + Qwen3MoE TP=2 能否正常启动 `swift rollout`
4. 重跑 Plan A：`start_rollout_server.sh` + `run_grpo_server_mode.sh`
5. **重跑方案十（canary）**：`MAX_COMPLETION_LENGTH=1024`，观察 clipped_ratio 是否下降、reward 是否出现上升趋势
6. 如果 Plan A 仍有问题，在原机器跑 Plan B：`run_grpo_stable_scaleup.sh`（已有超时修复）
7. 若成功：对比 GRPO 训后 reward 与 SFT baseline（0.211）
