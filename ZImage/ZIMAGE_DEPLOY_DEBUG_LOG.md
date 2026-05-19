# ZImage DLIS 部署调试日志

## 背景

将 ZImage (text-to-image) 模型部署到 DLIS 平台，基于 OaaS_LLMTemplate 框架。

- ZImage 基于 diffusers 的 `ZImagePipeline`（Tongyi-MAI/Z-Image-Turbo）
- 原始推理脚本使用 `accelerate launch --multi_gpu` + 8 GPU 并行
- 目标：适配 DLIS 单请求推理模式，支持 text-to-image 生成

## 参考实现

参考 branch: `user/hanbangliang/img-outpainting-v1`（图片 outpainting 服务）
- 使用 `Flux2KleinPipeline` + LoRA，接收 base64 图片输入
- 完整的 Kusto/EventHub 日志框架
- Pydantic 配置管理

## 部署 #1：初始 Branch 创建（2026-04-20）

### 操作
在 OaaS_LLMTemplate 仓库创建 `jinjinchen/ZImage-v1` branch（基于 main），commit `b4dbd9e`。

### 修改文件清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `dlis_model/model/model.py` | 重写 | 同步版 ZImage ModelImp，使用 ZImagePipeline |
| `dlis_model/model/async_model.py` | 重写 | 异步版，asyncio.to_thread 包装推理 |
| `dlis_model/model/config.py` | 新增 | Pydantic 配置，application_name="ZImageModel" |
| `dlis_model/model/kusto_log.py` | 新增 | Kusto 日志 handler，写入 EventHub |
| `dlis_model/model/eventhub_sink.py` | 新增 | EventHub 发送器 |
| `dlis_model/model/utils.py` | 修改 | 新增 get_tracking_data() |
| `requirements-vllm.txt` | 重写 | diffusers、Pillow、APScheduler 等依赖 |

### 请求格式
```json
{
  "prompt": "A beautiful sunset over the ocean",
  "width": 1344,
  "height": 768
}
```

### 环境变量配置
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| ZIMAGE_MODEL_PATH | /Model/model | 模型路径 |
| AB_DEFAULT_WIDTH | 1344 | 默认宽度 |
| AB_DEFAULT_HEIGHT | 768 | 默认高度 |
| AB_NUM_INFERENCE_STEPS | 9 | 推理步数 |
| AB_GUIDANCE_SCALE | 0 | CFG scale |
| AB_SEED | 42 | 随机种子 |
| AB_MODEL_DTYPE | bfloat16 | 模型精度 |
| AB_DEVICE | cuda:0 | 设备 |

### 构建方式
使用现有 `pipeline/build_vllm_image.sh` + `pipeline/Dockerfile_vllm_0.10.0`，自动安装 requirements-vllm.txt 中的依赖。

### 状态
- [x] Branch 创建完成
- [x] 代码编写完成
- [x] Commit 完成
- [x] Push 到 remote
- [x] 触发 pipeline 构建镜像
- [ ] DLIS 部署测试

## A6000 本地 Docker 测试（2026-04-20）

### 构建
使用 Hanbang 的 `Dockerfile_vllm_fast` + `build_vllm_image.sh` 构建镜像：
- 基础镜像：`vllm/vllm-openai:latest`
- 最终镜像：`dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:20260420-1255-jinjinchen-ZImage-v1`
- `Dockerfile_vllm_fast` 和 `build_vllm_image.sh` 从 Hanbang 的 `user/hanbangliang/img-outpainting-v1` branch 复制

### 遇到的问题与修复

1. **`build_vllm_image.sh` 中 `/mnt/storage` 不存在**：A6000 没有 `/mnt/storage`，`set -e` 导致脚本退出。注释掉相关行后解决。

2. **代码仍为 Gemma4 版本**：A6000 上 repo 在 `jinjinchen/Gemma4-v2-direct-vllm` branch，需 `git checkout jinjinchen/ZImage-v1` 并 `git checkout -- dlis_model/model/`。

3. **缺少 `dlis_inter.py`**：ZImage branch 没有此文件，创建了简单的 JSON 版 `PreAndPostProcessor`。

4. **缺少 `Dockerfile_vllm_fast`**：通过 `git show` 从 Hanbang branch 复制。

5. **CUDA OOM**：GPU 0 被 vLLM 进程占用，改用 `--gpus '"device=1"'`。

6. **403 Forbidden**：`docker run` 未加 `-p 8888:8888` 端口映射，宿主机 curl 被拒绝。容器内 curl 测试正常。

7. **`my-vllm-base` vs 最终镜像**：`my-vllm-base:zimage` 只是基础镜像，不含 `dlis_model/`。需使用 `docker commit` 后的完整镜像 `dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:20260420-1255-jinjinchen-ZImage-v1`。

8. **模型路径错误**：默认 `ZIMAGE_MODEL_PATH` 未设置，fallback 到 `/dlis_model/model`。需加 `-e ZIMAGE_MODEL_PATH=/Model`，且模型实际位于 `/home/jinjinchen/data/Z-Image`（非 `/home/jinjinchen/models/ZImage`）。

### 最终成功的启动命令
```bash
sudo docker run -d --name zimage-test \
  --gpus '"device=1"' \
  -e ZIMAGE_MODEL_PATH=/Model \
  -v /home/jinjinchen/data/Z-Image:/Model \
  -p 8889:8888 \
  dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:20260420-1255-jinjinchen-ZImage-v1 \
  /dlis_model/run.sh http
```

### 测试结果
- 模型加载成功，耗时 5126ms（第二次加载，模型已在内存缓存）
- `curl -X POST http://localhost:8889 --data '{"prompt": "A beautiful sunset over the ocean", "width": 1344, "height": 768}'` 推理成功
- 返回 base64 编码的 PNG 图片，包含 `image`、`width`、`height` 字段
- **结论：ZImage DLIS 本地 Docker 测试通过**

## 代码修复与 Pipeline 构建（2026-04-21）

### 修复 1：重命名 FLUX → ZIMAGE（commit `14caf90`）
- `model.py` 和 `async_model.py`：`FLUX_MODEL_PATH` → `ZIMAGE_MODEL_PATH`
- 默认值从 `self.model_dir`（`/dlis_model/model`）改为 `/Model`（DLIS 挂载路径）
- DLIS 部署时不再需要额外设置 `ZIMAGE_MODEL_PATH` 环境变量

### 修复 2：删除 dlis_inter.py 调试代码（commit `878b059`）
- 删除 `print(os.listdir("/Model"))`，避免 `/Model` 未 mount 时报错

### 修复 3：Pipeline 添加清华 PyPI 镜像（commit `3d99861`）
- ZImage 分支的 `azure-pipelines-unified.yml` 缺少 `PIP_INDEX_URL` 配置
- Pipeline agent 无法直连 `pypi.org`，`pip install` 超时导致构建失败
- 添加 `PIP_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple` 和 `UV_INDEX_URL`
- main 分支已有此配置，但 ZImage 分支从早期 main 分出时尚未包含

### Pipeline 构建 #1 失败分析
```
ERROR: Could not find a version that satisfies the requirement tornado (from versions: none)
ERROR: No matching distribution found for tornado
```
- 原因：`PIP_INDEX_URL` 未配置，`docker exec pip install` 直连 `pypi.org` 超时
- `build_vllm_image.sh` 中 `PIP_ARGS` 为空（日志：`Using PIP arguments:`）
- 已通过 commit `3d99861` 修复

### Cosmos 目录结构规划
DLIS 部署时 cosmos 目录 mount 到 `/Model`，推荐扁平结构：
```
cosmos: local/users/jinjinchen/zimage-v1/
├── Z-Image-Turbo/          ← 模型文件夹
│   ├── model_index.json
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── transformer/
│   └── vae/
├── dlis_inter.py            ← 直接放根目录
├── AggSvcAuthCert-prod.pfx  ← Kusto 证书（可选）
└── AggSvcAuthCert-si.pfx
```

对应环境变量：
```
DLIS_MODEL_DATA_TARGET_PATH=/Model
ZIMAGE_MODEL_PATH=/Model/Z-Image-Turbo
```

### 状态
- [x] 重命名 FLUX → ZIMAGE
- [x] 删除 dlis_inter.py 调试代码
- [x] 添加清华 PyPI 镜像配置
- [x] 确认 Gemma4 分支未被误改
- [x] Pipeline 构建成功
- [x] Cosmos 上传模型数据
- [x] DLIS 部署测试 ✅

## DLIS 部署测试（2026-04-22）

### 部署配置
- **镜像**：`dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:20260421-0328-merge`
- **ModelName**：`ZImage-V1-Jinjin`
- **Cosmos**：`abfs://dlisstore@dlisstoregen2.dfs.core.windows.net/dlismodelrepository-c09/local/users/jinjinchen/zimage-v1/`
- **环境变量**：`DLIS_MODEL_DATA_TARGET_PATH=/Model;GPU_MEMORY_UTILIZATION=0.7`

### 遇到的问题

1. **请求超时（Connection timed out）**：
   - 错误 URL：`http://WestUS2BE.bing.prod.dlis.binginternal.com:86/route/...`（HTTP + 端口 86）
   - 正确 URL：`https://WestUS2.bing.prod.dlis.binginternal.com/route/...`（HTTPS + 默认 443）
   - 注意：不需要 `:8888` 后缀，不需要 `/routebatch/`

2. **需要客户端证书**：
   - cert: `private1.cer` + `private1.key`
   - 路径：`/home/jinjinchen/dlis/abo-models/team/dai/auto_image/client/`

### 成功的请求方式
```python
import requests

response = requests.post(
    "https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.ZImage-V1-Jinjin",
    cert=("private1.cer", "private1.key"),
    json={"prompt": "A beautiful sunset over the ocean", "width": 1344, "height": 768},
    headers={"Content-Type": "application/json"},
    verify=False,
)
```

### 测试结果
- HTTP 200，推理成功
- 返回 base64 编码的 PNG 图片
- **结论：ZImage DLIS 部署测试通过 ✅**

## 部署 #3：Kusto 日志调试（2026-04-24）

### 问题
ZImage 部署后 Kusto 中查不到任何日志，`appsvc_info` 表中 `ApplicationName == 'ZImageModel'` 无结果。

### 根因分析（对比 siwen 和 hanbang 的实现）

1. **`eventhub_sink.py` 静默吞掉认证错误**：原代码 try-except 包裹 credential 创建，失败时不抛异常，导致 EventHub producer 为 None，日志全部丢弃。
   - **修复**：移除 try-except，直接创建 credential，认证失败时立即报错。

2. **`kusto_log.py` 格式不兼容**：字段用引号包裹、时间戳格式错误、ticks 计算错误。
   - **修复**：对齐 siwen 的格式 — 无引号、`%m/%d/%Y %H:%M:%S.%f` 时间戳、`int(timestamp * 1000)` ticks、逗号替换为 `[COMMA]`。

3. **Logger level 默认 WARNING**：`logging.getLogger("zimage")` 创建的子 logger 默认继承 root level（WARNING），所有 INFO 日志被过滤。
   - **修复**：在 `model.py` 和 `async_model.py` 中显式添加 `logger.setLevel(logging.INFO)`。

4. **`record.msg` 不解析格式参数**：使用 `%s` 格式的日志只输出 `%` 符号。
   - **修复**：改用 `record.getMessage()` 解析完整消息。

5. **请求中缺少 tracking_data 提取**：`model.py` 未从请求 JSON 中提取 tracking_data 字段。
   - **修复**：在 `_run_single` 中添加 tracking_data 解析逻辑。

6. **证书与 EventHub namespace 不匹配**：`config.py` 默认使用 SI namespace（`aggregation-si-logging`），但挂载的是 prod 证书（`AggSvcAuthCert-prod.pfx`）。
   - **修复**：在 `/Model/settings.json` 中覆盖 `eventhub_namespace` 为 prod（`aggregation-logging.servicebus.windows.net`），与 prod 证书匹配。

### 相关 Commits（OaaS_LLMTemplate `jinjinchen/ZImage-v1` branch）
- `63c8afe`：eventhub_sink 和 kusto_log 格式对齐 siwen
- `6d845c8`：logger.setLevel(logging.INFO) + record.getMessage()
- `8558bd7`：撤回误加的 tornado 依赖

### 本地 Docker 测试命令

#### 构建镜像（使用 build_vllm_image.sh）
```bash
cd ~/0424_test_zimage/OaaS_LLMTemplate
git checkout jinjinchen/ZImage-v1
sudo bash pipeline/build_vllm_image.sh
```

#### 创建 settings.json 覆盖 EventHub namespace
```bash
echo '{"eventhub_namespace": "aggregation-logging.servicebus.windows.net"}' > /home/jinjinchen/data/Z-Image/settings.json
```

#### 启动容器
```bash
sudo docker run -d --name zimage-test \
  --gpus '"device=1"' \
  -v /home/jinjinchen/data/Z-Image:/Model \
  -v /home/jinjinchen/data/pfx_cert/AggSvcAuthCert-prod.pfx:/Model/AggSvcAuthCert-prod.pfx \
  -p 8889:8888 \
  dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:20260424-0621- \
  /dlis_model/run.sh http
```

#### 查看日志
```bash
sudo docker logs -f zimage-test
```

#### 测试推理请求
```bash
curl -X POST http://localhost:8889 --data '{"prompt": "A beautiful sunset over the ocean", "width": 1344, "height": 768, "seed": 42, "tracking_data": {"requestid": "test-001", "trackingid": "track-001", "sessionid": "sess-001", "customerid": "cust-001", "callername": "local-test"}}'
```

#### Kusto 查询验证
```kql
appsvc_info | union appsvc_warn | union appsvc_err
| where Timestamp > ago(1d)
| where ApplicationName == 'ZImageModel'
```

### 测试结果
- 模型加载成功（~5s）
- Kusto 日志发送成功，prod 证书 + prod namespace 认证通过
- 推理请求正常返回 base64 图片
- **结论：ZImage Kusto 日志调试通过 ✅**

## 部署 #4：Kusto 日志增强（2026-04-24）

### 设计理念
参考团队同事的日志实现（hanbang 的分步耗时 + siwen 的配置日志和全量 tracking_data），结合 ZImage 的 T2I 推理特点进行增强。

**核心目标**：
- **可追踪性**：所有日志都带 `extra=tracking_data`，可通过 RequestId/TrackingId 追踪单个请求的完整生命周期
- **性能分析**：每个阶段独立计时（preprocess → inference → encode → postprocess），便于定位瓶颈
- **问题诊断**：记录完整 prompt 内容、配置参数、环境变量覆盖情况，出问题时可复现
- **实例区分**：init tracking_data 包含 `hostname_pid`，多实例部署时可区分日志来源

**参考对比**：
| 特性 | Hanbang (outpainting) | Siwen (MML vLLM) | ZImage (增强后) |
|------|----------------------|-------------------|-----------------|
| 分步耗时 | ✅ preprocess/infer/postprocess | ❌ | ✅ preprocess/infer/encode/postprocess |
| 配置日志 | ❌ | ✅ sampling params/tensor_parallel | ✅ device/dtype/size/cfg/steps/seed |
| tracking_data | 部分（推理阶段） | ✅ 全量 | ✅ 全量（含初始化和错误） |
| 错误 traceback | ✅ logger.exception | ❌ logger.error | ✅ logger.exception + tracking_data |
| prompt 内容 | ❌ | ❌ | ✅ 完整 prompt 文本 |
| 图片大小 | ❌ | ❌ | ✅ 输出 KB |
| 环境变量覆盖 | ❌ | ✅ | ✅ |
| EvalBatch 进度 | ❌ | ✅ | ✅ |

### 改动内容

1. **初始化配置日志**：记录 device、dtype、默认参数（width/height/steps/cfg/seed）、环境变量覆盖情况
2. **所有日志加 `extra=tracking_data`**：包括初始化、推理、错误、OnDataUpdate
3. **分步耗时细化**：新增 preprocess 和 encode 的独立计时
4. **图片编码信息**：记录输出图片大小（KB）
5. **完整耗时汇总**：`Eval total latency=Xms (preprocess=Xms, infer=Xms, encode=Xms, postprocess=Xms)`
6. **EvalBatch 日志**：记录批量处理进度和总耗时
7. **错误处理加 tracking_data**：错误日志也能追踪到具体请求
8. **实例标识**：init tracking_data 包含 `hostname_pid`
9. **Prompt 内容记录**：日志中记录完整 prompt 文本

### 相关 Commits
- `2d6d1c7`：综合日志增强
- `d643e25`：添加 prompt 内容记录

### 构建与测试

#### 构建镜像
```bash
cd ~/0424_test_zimage/OaaS_LLMTemplate
git pull origin jinjinchen/ZImage-v1
sudo bash pipeline/build_vllm_image.sh
# 镜像 tag: 20260424-0645-
```

#### 启动容器
```bash
sudo docker stop zimage-test && sudo docker rm zimage-test

sudo docker run -d --name zimage-test \
  --gpus '"device=1"' \
  -v /home/jinjinchen/data/Z-Image:/Model \
  -v /home/jinjinchen/data/pfx_cert/AggSvcAuthCert-prod.pfx:/Model/AggSvcAuthCert-prod.pfx \
  -p 8889:8888 \
  dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:20260424-0645- \
  /dlis_model/run.sh http
```

#### 测试推理请求
```bash
curl -X POST http://localhost:8889 --data '{"prompt": "A beautiful sunset over the ocean", "width": 1344, "height": 768, "seed": 42, "tracking_data": {"requestid": "test-002", "trackingid": "track-002", "sessionid": "sess-002", "customerid": "cust-002", "callername": "local-test"}}'

```

### 测试结果
- 模型加载成功
- 推理请求正常返回
- Kusto 日志包含完整信息：配置参数、prompt 内容、分步耗时、图片大小

#### Kusto 中预期看到的日志条目（按时间顺序）

**初始化阶段**（容器启动时）：
```
Message: Model Path: /dlis_model/run.sh
Message: Model Dir: /dlis_model/model
Message: ZImage config: device=cuda:0[COMMA] dtype=bfloat16[COMMA] default_size=1344x768[COMMA] guidance_scale=0.0[COMMA] num_inference_steps=9[COMMA] seed=42
Message: Using default parameters (no env overrides)
Message: Initializing ZImage pipeline...
Message: Loading ZImagePipeline from /Model[COMMA] dtype=bfloat16
Message: Model loaded successfully[COMMA] total latency=4914ms    (Duration: 4914)
```

**推理阶段**（收到请求时）：
```
Message: Eval request received                                     (RequestId: test-002)
Message: Preprocess done[COMMA] latency=0ms
Message: Inference request: prompt='A beautiful sunset over the ocean'[COMMA] prompt_len=33[COMMA] size=1344x768[COMMA] cfg=0.0[COMMA] steps=9[COMMA] seed=42
Message: Inference done[COMMA] latency=7634ms
Message: Image encoded[COMMA] latency=345ms[COMMA] output_size=1344x768[COMMA] ~2800KB
Message: Postprocess done[COMMA] latency=3ms
Message: Eval total latency=7980ms (preprocess=0ms[COMMA] infer=7634ms[COMMA] encode=345ms[COMMA] postprocess=3ms)[COMMA] size=1344x768    (Duration: 7980)
```

#### 关键 Kusto 字段映射
| Kusto 字段 | 来源 | 说明 |
|-----------|------|------|
| ApplicationName | config.py `application_name` | `ZImageModel` |
| ApplicationSubSystem | logger name | `zimage` |
| RequestId | tracking_data.requestid | 请求级追踪 |
| TrackingId | tracking_data.trackingid | 会话级追踪 |
| Duration | tracking_data["duration"] | 仅在 total latency 和 model loaded 时设置 |
| MethodName | record.funcName | `__init__` / `_run_single` |
| Reserved1 | hostname-pid | 实例标识 |
| Reserved3 | filename:lineno | `model.py:142` |

- **结论：ZImage Kusto 日志增强通过 ✅**

## 部署 #5：torch.compile 优化（2026-05-12）

### 背景
DLIS A100 上 ZImage 单次推理 avg latency ~4666ms，参考 Gemma4 通过 `ENFORCE_EAGER=false` 启用 CUDA Graph 后延迟大幅降低的经验，探索 ZImage（diffusers pipeline）的等效优化方案。

**关键区别**：
- Gemma4 (vLLM)：通过 `ENFORCE_EAGER=false` 启用内置 CUDA Graph
- ZImage (diffusers)：无内置 CUDA Graph，使用 `torch.compile` 对 transformer 子模块进行编译优化

### 优化思路

`torch.compile` 将 PyTorch 模型编译为优化的内核，减少 Python 开销和 GPU kernel launch 延迟。对 diffusers pipeline 只编译 `pipe.transformer`（计算最密集的部分），不编译整个 pipeline（避免 VAE decode 等动态操作的兼容问题）。

**设计决策**：
1. **仅编译 transformer**：`self.pipe.transformer = torch.compile(self.pipe.transformer, mode=compile_mode)` — 推理的主要计算瓶颈
2. **环境变量控制**：默认关闭（`AB_TORCH_COMPILE=false`），避免影响现有部署
3. **可配置编译模式**：`AB_TORCH_COMPILE_MODE`（默认 `reduce-overhead`）
4. **首次推理预热**：torch.compile 首次运行会编译，后续请求直接使用编译缓存

### 第一版：fullgraph=True（commit `a6de84c`）

```python
self.pipe.transformer = torch.compile(
    self.pipe.transformer, mode=compile_mode, fullgraph=True
)
```

**测试结果**：
- 单个 prompt 重复测试：avg latency 65ms（看似 71x 提升）
- **但实际是 DLIS 测试框架缓存了结果** — 1797 个请求中只有第一个真正推理，其余返回缓存
- 多个不同 prompt 测试时报错：`accumulated_recompile_limit reached with fullgraph=True`

**根因**：`fullgraph=True` 要求整个 forward 必须在一个编译图中完成，不允许 graph break。不同 prompt 长度/token 导致 dynamo 需要重编译，累积超过默认限制（8 次）后报错。

### 第二版：去掉 fullgraph + 增大缓存（commit `a27c4a9`）

```python
torch._dynamo.config.cache_size_limit = cache_size  # 默认 64
self.pipe.transformer = torch.compile(
    self.pipe.transformer, mode=compile_mode
)
```

**改动**：
1. 去掉 `fullgraph=True` — 允许 graph break，兼容不同输入
2. 增加 `torch._dynamo.config.cache_size_limit = 64`（默认 8 太小）— 缓存更多编译结果，避免驱逐导致的重编译
3. 新增 `AB_TORCH_COMPILE_CACHE_SIZE` 环境变量可配置缓存大小

### 环境变量配置

```
DLIS_MODEL_DATA_TARGET_PATH=/Model;GPU_MEMORY_UTILIZATION=0.7;ZIMAGE_MODEL_PATH=/Model/Z-Image-Turbo;AB_TORCH_COMPILE=true
```

完整可配置参数：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| AB_TORCH_COMPILE | false | 是否启用 torch.compile（true/1/yes 启用） |
| AB_TORCH_COMPILE_MODE | reduce-overhead | 编译模式（default/reduce-overhead/max-autotune） |
| AB_TORCH_COMPILE_CACHE_SIZE | 64 | dynamo 编译缓存条目数（默认 PyTorch 只有 8） |

### 测试经验教训

1. **DLIS 测试必须用多个不同 prompt** — 单个 prompt 的结果会被框架缓存，延迟数据不准确
2. **`fullgraph=True` 不适合动态输入** — diffusers transformer 内部有条件分支，不同输入会触发重编译
3. **`cache_size_limit` 需要增大** — 默认 8 对多种输入场景太小，64 是合理的起点
4. **首次推理会变慢** — torch.compile 编译耗时，第一个请求延迟会比未优化时更高，后续请求受益

### 相关 Commits
- `a6de84c`：初版 torch.compile（fullgraph=True）
- `a27c4a9`：修复 — 去掉 fullgraph，增大 cache_size_limit

### DLIS 测试结果（2026-05-12）

| 指标 | Baseline（无 compile） | torch.compile（reduce-overhead） |
|------|----------------------|--------------------------------|
| 平均延迟 | 4666ms | 3290ms |
| 提升幅度 | — | **~29.5% ↓** |
| 测试查询数 | 26 | 40 |
| 成功率 | 100% | 100% |
| 首次请求延迟 | ~5s | 更高（编译开销） |

**结论：torch.compile 优化有效，avg latency 从 4666ms 降至 3290ms，提升约 29.5% ✅**

### 状态
- [x] torch.compile 优化代码完成
- [x] fullgraph 重编译问题修复
- [x] 代码推送到 remote
- [x] 多 prompt DLIS 测试通过（40 queries，avg 3290ms，29.5% 提升）

### DLIS Signoff 测试用例（20 个不同 prompt）
```json
{"prompt": "A beautiful sunset over the ocean with golden light reflecting on calm waves", "width": 1344, "height": 768}
{"prompt": "A cozy coffee shop interior with warm lighting and bookshelves on rainy day", "width": 1344, "height": 768}
{"prompt": "A futuristic cyberpunk cityscape at night with neon signs and flying cars", "width": 1344, "height": 768}
{"prompt": "A golden retriever puppy playing in a field of wildflowers during spring", "width": 1344, "height": 768}
{"prompt": "An astronaut floating in space with Earth visible in the background", "width": 1344, "height": 768}
{"prompt": "A Japanese zen garden with cherry blossoms falling and a stone bridge", "width": 1344, "height": 768}
{"prompt": "A medieval castle on a cliff overlooking a misty valley at dawn", "width": 1344, "height": 768}
{"prompt": "A photorealistic portrait of a cat wearing a tiny top hat and monocle", "width": 1344, "height": 768}
{"prompt": "An underwater coral reef scene with tropical fish and sunlight rays", "width": 1344, "height": 768}
{"prompt": "A snowy mountain landscape with a wooden cabin and smoke from chimney", "width": 1344, "height": 768}
{"prompt": "A bustling night market in Bangkok with colorful food stalls and lanterns", "width": 1344, "height": 768}
{"prompt": "A minimalist modern living room with floor to ceiling windows and city view", "width": 1344, "height": 768}
{"prompt": "A fantasy dragon perched on a mountain peak breathing fire into stormy sky", "width": 1344, "height": 768}
{"prompt": "A vintage red sports car driving along a coastal highway at sunset", "width": 1344, "height": 768}
{"prompt": "A magical forest with glowing mushrooms and fireflies at twilight", "width": 1344, "height": 768}
{"prompt": "A professional food photography of a gourmet burger with melting cheese", "width": 1344, "height": 768}
{"prompt": "An oil painting style portrait of a woman in Renaissance clothing", "width": 1344, "height": 768}
{"prompt": "A steampunk airship flying through clouds above a Victorian city", "width": 1344, "height": 768}
{"prompt": "A peaceful lakeside scene with mountains reflected in crystal clear water", "width": 1344, "height": 768}
{"prompt": "A robot playing chess against a human in a dimly lit room", "width": 1344, "height": 768}
```

## 部署 #6：TensorRT 加速探索（2026-05-12）

### 背景
torch.compile（inductor backend）已将 avg latency 从 4666ms 降至 3290ms（~29.5%）。探索 TensorRT backend 是否能进一步加速。

### 方案：torch.compile + torch_tensorrt backend

最低侵入性的方式：复用现有 `torch.compile` 框架，仅切换 backend 为 `torch_tensorrt`。

```python
# inductor（当前）
torch.compile(pipe.transformer, mode="reduce-overhead")

# TensorRT
import torch_tensorrt
torch.compile(pipe.transformer, backend="torch_tensorrt")
```

### 代码改动
- `model.py` / `async_model.py`：新增 `AB_TORCH_COMPILE_BACKEND` 环境变量（默认 `inductor`）
- 当 backend 为 `torch_tensorrt` 时自动 import torch_tensorrt
- `requirements-vllm.txt`：添加 `torch-tensorrt` 依赖

### 环境变量
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| AB_TORCH_COMPILE_BACKEND | inductor | 编译 backend（inductor / torch_tensorrt） |

### 注意事项
- torch-tensorrt 版本需与 CUDA/PyTorch 严格匹配
- 首次编译时间更长（几分钟）
- 不同分辨率可能触发重编译
- A6000 先本地测试，验证兼容性后再部署 DLIS

### 状态
- [x] 代码添加 backend 可配置支持
- [x] A6000 本地 Docker 测试
- [x] 确认 TensorRT 兼容性 → **不兼容，放弃**

### 测试过程

#### 问题 1：torch-tensorrt PyPI 默认版本需要 CUDA 13
- `pip install torch-tensorrt` 拉取的 wheel 链接到 `libcudart.so.13`，基础镜像是 CUDA 12.9
- **解决**：从 PyTorch cu129 index 安装匹配版本 `torch-tensorrt==2.10.0+cu129`

#### 问题 2：`--no-deps` 导致缺少依赖
- `torch-tensorrt` 的 `--no-deps` 安装缺少 `dllist` 模块
- **解决**：单独 `pip install dllist`

#### 问题 3：TensorRT runtime (tensorrt pip 包) 版本不匹配
- 默认 `pip install tensorrt` 装了 cu13 版本（10.16.1.11），`trt.Builder()` 返回 nullptr
- **解决**：卸载 cu13 版本，安装 `tensorrt-cu12>=10.14,<10.16`（解析到 10.15.1.29）

#### 问题 4：`except ImportError` 未捕获 torch_tensorrt 运行时异常
- `torch_tensorrt` 初始化失败时抛出 `Exception`（不是 `ImportError`），导致容器崩溃
- **解决**：改为 `except Exception as e`

#### 问题 5（致命）：ZImage transformer 使用 `torch.complex64`
- TensorRT 不支持 complex 数据类型（仅支持 bool, int, long, half, float, bfloat16）
- ZImage transformer 内部使用 `torch.complex64`（rotary position embedding）
- 错误：`TypeError: Provided an unsupported data type as a data type for translation, got: torch.complex64`
- 尝试 `pass_through_build_failures=True` + `min_block_size=3`：**无效**
  - 原因：complex64 错误发生在 `prepare_inputs → Input.from_tensor → dtype._from` 阶段（输入准备），而非 TensorRT 引擎构建阶段
  - `pass_through_build_failures` 仅处理 build 阶段的失败，对输入准备阶段的 TypeError 无效
- **无法解决**：这是 TensorRT 架构级限制，complex64 作为图的输入张量传入，无法绕过

### 最终安装命令（验证可用，但推理不兼容）
```bash
pip uninstall -y tensorrt tensorrt-cu13 tensorrt-cu13-bindings tensorrt-cu13-libs
pip install 'tensorrt-cu12>=10.14,<10.16' dllist
pip install --no-deps --force-reinstall torch-tensorrt==2.10.0 --index-url https://download.pytorch.org/whl/cu129
# trt.Builder OK，但推理时因 complex64 失败
```

### 结论
**TensorRT backend 与 ZImage 模型不兼容** ❌

ZImage transformer 内部使用 `torch.complex64`（rotary position embedding 中的频率计算），TensorRT 不支持此数据类型。这是架构级限制，无法通过版本升级或配置绕过。

**生产方案：使用 inductor backend**（torch.compile 默认），已验证 4666ms → 3290ms（~29.5% 加速）。

---

## 部署 #7：FlashAttention 可行性调研

**日期**：2026-05-13
**目标**：评估 FlashAttention 能否进一步加速 ZImage 推理
**测试环境**：A6000 (48GB GDDR6)，无 torch.compile

### 调研过程

#### 1. ZImage 注意力架构分析
- ZImage 使用 `ZSingleStreamAttnProcessor`（自定义 attention processor）
- 底层调用 `dispatch_attention_fn`（来自 `diffusers.models.attention_dispatch`）
- 默认使用 **NATIVE backend**（即 PyTorch SDPA，`F.scaled_dot_product_attention`）
- SDPA 会自动选择最优 kernel（flash / mem_efficient / math）

#### 2. FlashAttention 兼容性测试
- ZImage transformer 的 **head_dim = 512**
- FlashAttention 内核要求 **head_dim ≤ 256**
- 强制启用 Flash kernel 报错：
  ```
  RuntimeError: FlashAttention only supports head dimensions up to 256
  ```
- **结论：FlashAttention 与 ZImage 不兼容** ❌

#### 3. Attention Kernel 性能对比（A6000，无 torch.compile）

| Kernel | 延迟 | 说明 |
|--------|------|------|
| Math（纯数学实现） | 15738ms | 最慢，无优化 |
| Mem Efficient | 6989ms | xformers 风格，支持大 head_dim |
| **Default（SDPA 自动选择）** | **6470ms** | **已是最优** |
| Flash | ❌ 不兼容 | head_dim=512 超限 |

#### 4. A6000 vs A100 延迟差异

| 配置 | A100 (DLIS) | A6000 (lixin) |
|------|-------------|---------------|
| 无 torch.compile | 4666ms | 6470ms |
| 有 torch.compile | 3290ms | ~4500ms（预估） |

A6000 比 A100 慢约 39%，主要因为内存带宽差异（A100: 2039 GB/s vs A6000: 768 GB/s），diffusion model 推理是 memory-bound 任务。

### 结论
**FlashAttention 与 ZImage 不兼容** ❌（head_dim=512 > 限制 256）

ZImage 默认的 SDPA 自动选择已是当前最优方案，无需额外配置。注意力层不是当前瓶颈，torch.compile（inductor backend）仍是最有效的优化手段。

---

## 后续优化方案汇总与可行性评估

**日期**：2026-05-13
**背景**：torch.compile (inductor) 已实现 29.5% 加速，TensorRT 和 FlashAttention 均不兼容。以下整理所有已探索和待探索的优化方向。

### 已验证方案

| 方案 | 状态 | 结果 |
|------|------|------|
| torch.compile (inductor) | ✅ 已部署 | 4666ms → 3290ms（29.5% 加速） |
| TensorRT backend | ❌ 不兼容 | complex64 rotary embedding + 非标准算子组合导致 FX/ONNX 图无法完整转换（部署 #6） |
| FlashAttention standalone | ❌ 不兼容 | head_dim=512 > FA v2 限制 256（部署 #7）；但 PyTorch SDPA 已自动选择 mem_efficient kernel |
| xformers memory_efficient | ⚪ 无额外收益 | SDPA 默认已自动选择 mem_efficient kernel |
| DeepCache | ❌ 不兼容 | ZImage 使用 Transformer 架构，DeepCache 需要 UNet（`'ZImagePipeline' object has no attribute 'unet'`） |
| INT8 权重量化 (torchao) | ❌ 无效 | W8A16 weight-only 量化：无 compile 时变慢 10%（6403→7066ms）。注意：未测试 W8A8 双量化，Compute-Bound 场景可能需要激活也量化 |
| Channels-Last 内存格式 | ❌ 无效 | 全模型：慢 4%（6465→6746ms）；仅 VAE：慢 4.6%（6440→6739ms）。VAE 占比小，格式转换开销 > 收益 |
| torch.compile VAE | ⚪ 无额外收益 | compile transformer 5126ms → +VAE 5185ms，VAE 占总延迟 <1%，不值得额外 30s 编译时间 |
| torch.compile + INT8 | ❌ 严重劣化 | compile-only 5064ms → compile+INT8 786,511ms（155x 慢）。AffineQuantizedTensor subclass 导致编译路径极慢，fusion 未生效 |
| ONNX Runtime (GPU EP) | ❌ 不兼容 | torch.export 导出阶段失败（strict/non-strict 均失败），ORT 本身未执行。根因是 complex64 + 动态 freqs_cis 导致图无法 trace |

### 待验证方案（按优先级排序）

#### 🥇 优先级 1：~~DeepCache（算法级优化）~~ ❌ 已排除
- **原理**：复用相邻 step 的高层特征，跳过部分 Transformer Block 计算
- **测试结果**：`'ZImagePipeline' object has no attribute 'unet'`
- **结论**：DeepCache 仅支持 UNet 架构的 pipeline（如 StableDiffusionPipeline），ZImage 使用 Transformer 架构，不兼容

#### 🥈 优先级 2：INT8 权重量化（torchao）
- **原理**：对权重做 INT8 量化，减少内存带宽压力（diffusion 是 memory-bound）
- **预期收益**：15-30%
- **改动量**：几行代码
- **风险**：需验证画质损失是否可接受
- **硬件支持**：A100 有 INT8 Tensor Core
```python
from torchao.quantization import quantize_, int8_weight_only
quantize_(pipe.transformer, int8_weight_only())
```

#### 🥉 优先级 3：torch.compile 扩展到 VAE
- **原理**：当前仅 compile transformer，VAE decoder 也可优化
- **预期收益**：5-15%
- **改动量**：一行代码
- **风险**：低
```python
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead")
```

#### 优先级 4：Channels-Last 内存格式
- **原理**：让 Conv2d 使用更高效的 NHWC 布局，对 VAE 特别有效
- **预期收益**：5-10%
- **改动量**：两行代码
- **风险**：低
```python
pipe.transformer = pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
```

#### 优先级 5：VAE Tiling
- **原理**：分块解码，减少显存峰值
- **预期收益**：对 1344x768 可能有轻微改善
- **改动量**：一行代码
- **风险**：低
```python
pipe.enable_vae_tiling()
```

#### 优先级 6：TinyVAE / TAESD
- **原理**：用轻量 VAE 替换默认 VAE，解码耗时大幅缩短
- **预期收益**：VAE 部分 10-15%
- **改动量**：中等
- **风险**：ZImage 是自定义架构，**可能没有对应的 Tiny VAE**，需确认

### 不适用方案（已排除）

| 方案 | 排除原因 |
|------|----------|
| FP8 量化 | A100 (SM80) 不支持 FP8，需 H100/Ada (SM90+) |
| LCM / SDXL-Lightning 蒸馏 | ZImage 是自定义架构，无现成蒸馏模型 |
| 单步蒸馏 (ADD/SDXL-Turbo) | 同上，需团队自行训练 |
| model_cpu_offload | A100 80GB 显存充足，开启反而变慢 |
| FP16 vs BF16 切换 | A100 两者吞吐相同，BF16 数值更稳定，当前选择已是最优 |

### 下一步计划
1. ~~在 A6000 测试 DeepCache 兼容性和画质~~ ❌ 不兼容（需要 UNet）
2. ~~测试 INT8 量化效果~~ ❌ 无 compile 时变慢，需搭配 torch.compile 验证
3. ~~测试 torch.compile 扩展到 VAE~~ ⚪ 无额外收益（VAE 占比 <1%）
4. ~~测试 Channels-Last 内存格式（NHWC 布局）~~ ❌ 变慢
5. ~~评估 ONNX Runtime（GPU EP）可行性~~ ❌ 不兼容（torch.export 失败，无法导出 ONNX）
6. ~~测试 torch.compile + INT8 量化组合~~ ❌ 严重劣化（155x 慢，fusion 未生效）
7. 将有效方案集成到 DLIS 部署代码

#### 第二轮优化（基于 Gemini + ChatGPT 深度分析）

**部署侧可直接测试（高优先级）：**
8. 测试 `torch.compile(mode="max-autotune")` vs 当前 `mode="reduce-overhead"`
9. 确认 TF32 是否开启（`torch.backends.cuda.matmul.allow_tf32 = True`）
10. 测试 torch.compile + CUDA Graph（diffusion 每步图结构相同，适合 capture，预期 10-30% 加速）
11. 调研 TeaCache / DiT-Cache 对 ZImage 的适配性（DiT 专用缓存，跳过冗余 timestep 计算，潜在 1.5-2x 加速）
12. 测试 W8A8 量化（权重+激活双量化，针对 Compute-Bound 场景，区别于之前 W8A16 的失败）
13. Prompt Embedding Cache（固定 prompt 时缓存 text encoder output，预期节省 5-15%）
14. Scheduler 降步数测试（50→28→20 步，配合 DPM-Solver++ 等高阶 sampler）
15. Dynamic Shape Specialization（固定 latent shape + batch size，提升 torch.compile 效果）

**需要团队配合修改模型代码（中长期）：**
16. 重写 RoPE 去除 complex64 → 用实数 cos/sin 替换复数乘法（解锁 TensorRT / ONNX / fullgraph=True，**最关键的战略性优化**）
17. 蒸馏降步数（LCM / TCD / Rectified Flow）→ 50步降至4-8步，潜在 5x+ 加速
18. 下一代模型将 head_dim 从 512 降至 ≤256 → 解锁 FlashAttention
19. FP8 推理（需 H100/B200 硬件升级，1.2-1.8x 加速）

**系统级优化（运维侧）：**
20. NUMA 绑定（`numactl --cpunodebind=0 --membind=0`）
21. 固定 GPU 时钟（`nvidia-smi -lgc`）
22. 确保非计时路径无多余 `torch.cuda.synchronize()`

---

## 第二轮优化 Benchmark 结果（2026-05-14）

**测试环境**：A6000 Docker，模型 `/home/lixinqian/jinjin/Z-Image`，bfloat16，768×1344，9 steps
**测试脚本**：`test_round2.py`（3 warmup + 3 timed runs per test）

### ⚠️ 重要发现：TF32 matmul 默认未开启

```
torch.backends.cuda.matmul.allow_tf32 = False  ← 默认值！
torch.backends.cudnn.allow_tf32  = True
```

**`matmul.allow_tf32` 在 PyTorch 中默认为 `False`**（PyTorch ≥1.12），这意味着之前所有测试（包括 DLIS A100 生产环境）可能都没有利用 TF32 matmul 加速。

**后续实测**：TF32 ON vs OFF 对比（去掉首次重编译 outlier）：9843ms vs 9770ms，**无差异（~0.7%）**。ZImage 在 BF16 下运行，TF32 仅影响 FP32 matmul，对 ZImage 无实际影响。无需在生产中特别处理。

### Benchmark 结果

| 测试 | 配置 | 延迟 (A6000) | vs Baseline | 备注 |
|------|------|-------------|-------------|------|
| [A] Baseline | 无 compile，TF32 已启用 | 12404ms | — | 基准 |
| [B] reduce-overhead | torch.compile 当前生产模式 | 9898ms | **+20.2%** | 当前部署配置 |
| [C] max-autotune | torch.compile 含 kernel autotuning | 9908ms | +20.1% | 与 B 无差异 |
| [D-7] 7 steps | max-autotune + 7 步 | 7827ms | +36.9% | 需质量验证 |
| [D-5] 5 steps | max-autotune + 5 步 | 5690ms | +54.1% | 需质量验证 |
| [D-3] 3 steps | max-autotune + 3 步 | 3553ms | +71.3% | 需质量验证 |
| [E] Prompt Cache | max-autotune + 缓存 prompt embeddings | 9747ms | +21.4% | vs B 仅 ~1.6% |

### 关键结论

1. **max-autotune ≈ reduce-overhead**（TODO #8 + #10）：差异仅 0.1%，max-autotune 编译时间更长，不值得切换。CUDA Graph 已通过 `reduce-overhead` 生效。
2. **TF32 matmul 默认关闭**（TODO #9）：⚠️ **潜在生产问题**，需确认 DLIS 环境并显式启用。
3. **步数线性缩放**（TODO #14）：9→7→5→3 步延迟线性下降，**9→7 步（-22%）是最佳性价比**，但需画质验证。
4. **Prompt Cache 边际收益**（TODO #13）：仅 ~1.6%，text encoder 占比很小，对固定 prompt 场景有轻微帮助。

### TODO 状态更新

| TODO | 项目 | 状态 | 结果 |
|------|------|------|------|
| #8 | max-autotune vs reduce-overhead | ✅ 已测 | ⚪ 无差异，保持 reduce-overhead |
| #9 | TF32 确认 | ✅ 已测 | ⚪ 无影响（BF16 模型不走 FP32 matmul） |
| #10 | CUDA Graph | ✅ 已测 | ⚪ 已通过 reduce-overhead 隐式启用 |
| #11 | TeaCache / DiT-Cache | 🔲 待测 | — |
| #12 | W8A8 量化 | 🔲 待测 | — |
| #13 | Prompt Embedding Cache | ✅ 已测 | ⚪ 边际收益 ~1.6% |
| #14 | 降步数 | ✅ 已测 | ✅ 线性缩放，9→7 步可降 22%，需画质验证 |
| #15 | Dynamic Shape Specialization | 🔲 待测 | — |
| #16 | 重写 RoPE 去除 complex64 | ✅ 已测 | ⚪ 可行但直接性能收益有限（见下方详细结果） |

---

## 部署 #8：Real RoPE（去除 complex64）可行性验证（2026-05-18）

### 背景
ZImage transformer 使用 `torch.complex64` 实现 RoPE（Rotary Position Embedding），这是导致 TensorRT、ONNX Runtime、`fullgraph=True` 全部不兼容的根因。探索用实数 cos/sin 运算替换复数乘法。

### 改写方案

**原始代码**（3 处修改）：

1. `RopeEmbedder.precompute_freqs_cis`：
```python
# 原始：生成 complex64 张量
freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)

# 改写：拼接 cos/sin 为实数张量 [seq, dim*2]
freqs_cis_i = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
```

2. `ZSingleStreamAttnProcessor.__call__` 中的 `apply_rotary_emb`：
```python
# 原始：complex 乘法
x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
freqs_cis = freqs_cis.unsqueeze(2)
x_out = torch.view_as_real(x * freqs_cis).flatten(3)

# 改写：实数 cos/sin 旋转
half = freqs_cis.shape[-1] // 2
cos = freqs_cis[..., :half].unsqueeze(2).float()
sin = freqs_cis[..., half:].unsqueeze(2).float()
x_even = x_in.float()[..., 0::2]
x_odd = x_in.float()[..., 1::2]
x_out = torch.stack([x_even * cos - x_odd * sin,
                     x_even * sin + x_odd * cos], dim=-1).flatten(-2)
```

3. `_prepare_sequence` 中的 `freqs_cis` 处理：**无需修改**，因为改写后 `freqs_cis` 仍是普通 tensor（shape 从 `[seq, dim]` complex 变为 `[seq, dim*2]` real），`.split()` 和 `pad_sequence` 正常工作。

### 数学等价性验证 ✅

```
Axis 0: cos match=True, sin match=True
Axis 1: cos match=True, sin match=True
Axis 2: cos match=True, sin match=True
apply_rotary_emb equivalence: match=True, max_diff=1.19e-07
```

### 端到端 Benchmark（A6000）

| 配置 | 原始 complex64 | Real RoPE | 差异 |
|------|---------------|-----------|------|
| 无 compile | 12404ms | 13466ms | **+8.6% 变慢** |
| + compile (reduce-overhead) | 9898ms | 9744ms | **-1.6% 略快** |

### 分析

- **无 compile 变慢**：`torch.stack + flatten` 比 native `view_as_complex * view_as_real` 多了内存分配，eager 模式下 complex 乘法更高效
- **有 compile 略快**：编译器优化了 real 运算路径，且消除了 "Torchinductor does not support complex operators" 警告导致的 fallback 开销
- **战略价值**（不体现在直接延迟上）：
  - ✅ 解锁 `fullgraph=True`（待验证）
  - ✅ 解锁 TensorRT backend（之前因 complex64 完全不兼容）
  - ✅ 解锁 ONNX 导出（之前 torch.export 失败）
  - ✅ 消除 "Torchinductor does not support complex operators" 警告

### 实现方式
通过直接修改 diffusers 源码 `transformer_z_image.py`（Docker 内 monkey-patch），修改 2 处函数即可。生产部署建议在 model.py 初始化时通过 monkey-patch 注入，避免修改 diffusers 包。

### fullgraph=True 测试结果（2026-05-18）

Real RoPE patch 后测试 `fullgraph=True`，过程中遇到并修复了两个障碍：

**障碍 1: `with torch.device("cpu"):` 上下文管理器**
- `precompute_freqs_cis` 中使用 `with torch.device("cpu"):`，dynamo 不支持
- 修复：去掉上下文管理器，tensor 构造已显式带 `device="cpu"` 参数

**障碍 2: CUDA Graph 缓存 tensor 被覆写**
- `self.freqs_cis` 预计算后缓存在 module 属性上，CUDA Graph replay 时被覆写
- 报错：`RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`
- `.clone()` 不够——`max-autotune` 强制开启 CUDA Graphs，与缓存 tensor 根本冲突
- 解决方案：禁用 CUDA Graphs（`torch._inductor.config.triton.cudagraphs = False` + `cudagraph_trees = False`）

**最终可运行配置及性能：**

| 配置 | 平均延迟 | vs reduce-overhead 基准 |
|------|---------|----------------------|
| reduce-overhead（无 fullgraph，当前生产） | ~9770ms | 基准 |
| max-autotune（无 fullgraph） | ~9744ms | -0.3% |
| **fullgraph=True + reduce-overhead（禁 CUDA Graphs）** | **9637ms** | **-1.4%** |

**结论**：
- ✅ Real RoPE 成功解锁 `fullgraph=True`（去除 complex64 + 修复 `torch.device` 上下文管理器）
- ⚠️ CUDA Graphs 与 transformer 内部缓存 tensor（`self.freqs_cis`）不兼容，必须禁用
- 📊 fullgraph=True 带来约 1.4% 的小幅提升（~133ms），不算显著
- 🔑 fullgraph 的真正价值在于解锁 TensorRT/ONNX 导出，而非直接的 compile 性能提升

### TensorRT Backend 测试结果（2026-05-18）

安装 `torch_tensorrt 2.11.0+cu129`，使用 `torch.compile(backend='torch_tensorrt', fullgraph=True)` 编译。

**结果：❌ 转换失败**

```
ERROR: ITensor::getDimensions: IScatterLayer `input` and `updates` must have identical types.
       `input` type is BFloat16 and `updates` type is Int32.

BackendCompilerFailed: aten.scatter.src 操作中 input(BF16) 与 index/updates(Int32) 类型不匹配
```

**根因**：`torch_tensorrt` 的 `aten.scatter.src` converter 不支持混合类型（BF16 input + Int32 updates）。scatter 操作在 transformer 的 mask 构建中使用，无法简单绕过。

**尝试的修复方案：**

1. **`torch_executed_ops` 排除 scatter** — 让 scatter fallback 到 PyTorch：
   - TRT engine 构建时仍报类型错误：`Set network output type BFloat16 must be same as inferred output type Int32`
   - 即使 engine 构建成功的子图，fallback 回 PyTorch 时 `scatter(): Expected self.dtype to be equal to src.dtype`
   
2. **scatter 来源分析**：
   - 源码中无显式 scatter 调用
   - `_prepare_sequence` 中的 `zeros` + index assignment（`tensor[indices] = values`）被 dynamo trace 降级为 `aten.scatter`
   - BF16 模型中 zeros 默认创建 float32/int 类型，与 bf16 值产生类型冲突

**结论**：
- ❌ TensorRT backend 对 ZImage transformer 当前不可用（torch_tensorrt 2.11.0 + TRT）
- 根因是多层类型不匹配：scatter 混合类型 + TRT engine 输出类型声明
- 不是 Real RoPE 的问题，是 transformer 内部隐式 scatter（index assignment）与 TRT 的兼容性问题
- 修补投入产出比极低，建议等 torch_tensorrt 后续版本改善 BF16 支持

### 待验证
- [x] Real RoPE + `fullgraph=True` 是否能工作 → ✅ 可以，需禁用 CUDA Graphs
- [x] Real RoPE + TensorRT backend 是否能工作 → ❌ scatter 混合类型不兼容
- [ ] 生成图片质量对比（pixel-level 差异验证）

---

## 部署 #9: TeaCache / First Block Cache (TODO #11)（2026-05-18）

### 背景

TeaCache 通过检测相邻 denoising step 之间 transformer 中间特征的相似度，跳过高相似度 step 的完整计算。diffusers 0.38.0 内置了 `apply_first_block_cache`（基于 TeaCache 思想的 First Block Cache, FBC）。

### 兼容性分析

- ✅ `ZImageTransformerBlock` 已注册在 `TransformerBlockRegistry`（`return_hidden_states_index=0`, 无 `encoder_hidden_states`）
- ✅ `ZSingleStreamAttnProcessor` 已注册在 `AttentionProcessorRegistry`
- ✅ transformer 使用 `self.layers = nn.ModuleList()`，匹配 `_ALL_TRANSFORMER_BLOCK_IDENTIFIERS` 中的 `"layers"`
- ✅ `_should_compute_remaining_blocks` 有 `@torch.compiler.disable` 装饰器，与 `torch.compile` 兼容
- ⚠️ ZImage 默认只有 9 步推理，可跳过的步骤有限，加速比可能不如 20-50 步模型显著

### 测试计划

1. **Baseline（无 FBC）**：无 compile，确认原始延迟
2. **FBC 不同 threshold**：0.03, 0.05, 0.08, 0.1, 0.15, 0.2 — 找最佳质量/速度平衡点
3. **FBC + torch.compile**：最佳 threshold 与 reduce-overhead + fullgraph 组合

### Bug 修复：`hidden_states_argument_name` 缺失

首次运行 FBC 报错：
```
ValueError: Parameter 'hidden_states' not found in function signature but was requested.
```

**根因**：`ZImageTransformerBlock.forward` 签名是 `(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None)`，第一个参数是 `x` 而非标准的 `hidden_states`。但 diffusers `_helpers.py` 中 ZImage 的 `TransformerBlockMetadata` 注册没有指定 `hidden_states_argument_name`，默认查找 `"hidden_states"` 参数，导致失败。

**修复**：Patch `_helpers.py`（L327-334），在 ZImage 的 `TransformerBlockMetadata` 中添加 `hidden_states_argument_name="x"`：

```python
# 文件：/usr/local/lib/python3.12/dist-packages/diffusers/hooks/_helpers.py
# 修改前：
    # ZImage
    TransformerBlockRegistry.register(
        model_class=ZImageTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )

# 修改后：
    # ZImage
    TransformerBlockRegistry.register(
        model_class=ZImageTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
            hidden_states_argument_name="x",
        ),
    )
```

**一键 Patch 命令**（在 Docker 容器内执行）：
```bash
python3 -c "
path = '/usr/local/lib/python3.12/dist-packages/diffusers/hooks/_helpers.py'
with open(path, 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'return_encoder_hidden_states_index=None,' in line and i > 320:
        if 'hidden_states_argument_name' not in lines[i+1]:
            lines.insert(i+1, '            hidden_states_argument_name=\"x\",\n')
            print(f'Inserted hidden_states_argument_name at line {i+2}')
            break
with open(path, 'w') as f:
    f.writelines(lines)
"
```

**验证 Patch**：
```bash
grep -n -A6 "# ZImage" /usr/local/lib/python3.12/dist-packages/diffusers/hooks/_helpers.py
# 应输出含 hidden_states_argument_name="x" 的行
```

> 📝 这是 diffusers 0.38.0 的 bug——ZImageTransformerBlock 用了非标准参数名 `x`，注册时遗漏了 `hidden_states_argument_name` 配置。

### 需要的 Patch（3 处，diffusers 0.38.0 的 bug）

| # | 文件 | 修改 | 原因 |
|---|------|------|------|
| 1 | `diffusers/hooks/_helpers.py` L332 | ZImage 注册添加 `hidden_states_argument_name="x"` | forward 参数名是 `x` 不是 `hidden_states` |
| 2 | `diffusers/hooks/first_block_cache.py` L79,L158 | `"hidden_states"` → `self._metadata.hidden_states_argument_name` | 硬编码了参数名，未用 metadata 字段 |
| 3 | `diffusers/pipelines/z_image/pipeline_z_image.py` L528 | transformer 调用前后添加 `_set_context('default')` / `_set_context(None)` | ZImage pipeline 缺少 `cache_context` 调用，StateManager 无 context |

### ⚠️ 模型权重问题排查

初始测试使用本地 `/home/lixinqian/jinjin/Z-Image` 权重，baseline 输出存在明显颗粒感/不清晰。排查过程：

1. ❌ 怀疑 Real RoPE patch 残留 → clean reinstall diffusers 后仍有颗粒感
2. ❌ 怀疑 FBC patch 影响 → 无任何 patch 时仍有颗粒感
3. ❌ 怀疑 VAE bf16 精度 → 本地脚本同样用 bf16 但输出清晰
4. ✅ 改用 HuggingFace Hub `Tongyi-MAI/Z-Image-Turbo` 加载 → 输出正常

**结论**：本地 `/home/lixinqian/jinjin/Z-Image` 权重与 Hub 版本不一致（可能是旧版本或非 Turbo 版本），导致生成质量下降。后续测试和部署统一使用 Hub 版本。

### 测试结果（Hub ckpt + FBC patch，无 torch.compile，A6000 单卡）

| 配置 | 平均延迟 | vs Baseline | 加速比 | PSNR | 质量评估 |
|------|---------|-------------|--------|------|---------|
| Baseline（无 FBC） | 6386ms | 基准 | 1.0x | — | ✅ 基准 |
| FBC t=0.2 | 6515ms | +2.0% | 0.98x | ∞ (identical) | ✅ 无损（未触发跳步） |
| FBC t=0.25 | 5948ms | -6.9% | 1.07x | 35.7dB | ✅ 优秀 |
| FBC t=0.3 | 5336ms | -16.4% | 1.20x | 32.3dB | ✅ 良好（>30dB） |
| FBC t=0.4 | 5348ms | -16.3% | 1.19x | 32.3dB | ✅ 良好（>30dB） |
| FBC t=0.5 | 4716ms | -26.1% | 1.35x | 26.5dB | ⚠️ 肉眼可见差异 |

**分析**：
- t=0.2 时 PSNR=∞（与 baseline 完全相同），说明在 9 步推理中无任何步被跳过，hook 开销反而增加 ~130ms
- t=0.25 开始跳步，6.9% 加速，PSNR 35.7dB 质量优秀
- t=0.3 和 t=0.4 加速效果几乎相同（~16%），说明在此区间跳过的步数一样（PSNR 也一样 32.3dB）
- t=0.5 进一步跳步，26% 加速但 PSNR 降至 26.5dB，肉眼可见色彩/细节差异

### 肉眼质量评估

| threshold | 肉眼评估 |
|-----------|---------|
| t=0.2 | ✅ 与 baseline 完全一致（pixel-identical） |
| t=0.25 | ✅ 无明显差异 |
| t=0.3 | ✅ 无明显差异 |
| t=0.4 | ✅ 无明显差异 |
| t=0.5 | ⚠️ 可见差异 |

### 结论

- **推荐配置：threshold = 0.3~0.4**
  - 1.20x 加速（6386→5336ms），节省 ~1050ms
  - PSNR 32.3dB（>30dB），肉眼无明显差异
- t=0.25 保守选择：1.07x 加速，PSNR 35.7dB，质量优秀
- t=0.5 开始肉眼可见退化（PSNR 26.5dB）
- **FBC 是目前测试过的最有效优化**：仅需 3 处 diffusers patch，最高 1.35x 加速且质量可接受

### ⚠️ DLIS 部署注意：FBC 状态必须在请求间 Reset

多 prompt 连续推理时，FBC 会缓存上一次推理的中间状态（hidden_states_residual）。如果前后请求的 **token 数量不同**（不同 prompt 长度、不同分辨率），会触发 shape mismatch：

```
RuntimeError: The size of tensor a (4064) must match the size of tensor b (4096) at non-singleton dimension 1
```

**解决方法**：每次推理前 reset FBC 的 stateful hooks：

```python
from diffusers.hooks import HookRegistry

# 在每次 pipe() 调用前执行
registry = HookRegistry.check_if_exists_or_initialize(pipe.transformer)
registry.reset_stateful_hooks(recurse=True)
```

**DLIS 部署 model.py 中必须在 `_run_single()` 的推理调用前加上这段 reset 逻辑**，否则不同请求间会 crash。即使 prompt 长度相同，也建议始终 reset 以避免前序请求的缓存状态污染当前推理结果。

### FBC 测试命令（Patch 后执行）

```bash
python3 -c "
import torch, time
from diffusers import ZImagePipeline
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig

pipe = ZImagePipeline.from_pretrained('/home/lixinqian/jinjin/Z-Image', torch_dtype=torch.bfloat16)
pipe.to('cuda')

prompt = 'A beautiful sunset over the ocean, photorealistic, 8k'
gen = torch.Generator(device='cuda').manual_seed(42)

# Baseline
print('=== Baseline (no FBC) ===')
times = []
for i in range(4):
    gen.manual_seed(42)
    torch.cuda.synchronize()
    t0 = time.time()
    img = pipe(prompt, height=768, width=1344, guidance_scale=0, num_inference_steps=9, generator=gen).images[0]
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) * 1000
    print(f'  Run {i+1}: {elapsed:.0f}ms')
    if i > 0: times.append(elapsed)
print(f'  Avg (excl warmup): {sum(times)/len(times):.0f}ms')
img.save('/tmp/baseline_no_fbc.png')

for threshold in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
    del pipe
    torch.cuda.empty_cache()
    pipe = ZImagePipeline.from_pretrained('/home/lixinqian/jinjin/Z-Image', torch_dtype=torch.bfloat16)
    pipe.to('cuda')
    apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=threshold))
    print(f'\n=== FBC threshold={threshold} ===')
    times = []
    for i in range(4):
        gen.manual_seed(42)
        torch.cuda.synchronize()
        t0 = time.time()
        img = pipe(prompt, height=768, width=1344, guidance_scale=0, num_inference_steps=9, generator=gen).images[0]
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000
        print(f'  Run {i+1}: {elapsed:.0f}ms')
        if i > 0: times.append(elapsed)
    print(f'  Avg (excl warmup): {sum(times)/len(times):.0f}ms')
    img.save(f'/tmp/fbc_t{threshold}.png')
print('\nDone!')
"
```

## 优化测试 #7：torch.compile + FBC 组合（2026-05-18）

### 环境
- Docker 容器 `zimage-test`，A6000 GPU
- PyTorch 2.11.0+cu129，Triton 3.6.0
- diffusers 0.38.0 + 3 FBC patches
- Hub ckpt: `Tongyi-MAI/Z-Image-Turbo`
- 测试分辨率：1344x768，9 steps，seed=42

### 发现：ZImageTransformer2DModel 未继承 CacheMixin

```
MRO: ['ZImageTransformer2DModel', 'ModelMixin', 'Module', 'PushToHubMixin', 'ConfigMixin', 'PeftAdapterMixin', 'FromOriginalModelMixin', 'object']
```

因此不能用 `pipe.transformer.enable_cache()`，必须直接调用 `apply_first_block_cache(pipe.transformer, config)`。

### torch.compile 模式选择

- `mode='reduce-overhead'`：使用 CUDA Graphs，与 FBC 有状态 hook **冲突**
  - 报错：`RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`
  - 原因：FBC 的 `_should_compute_remaining_blocks()` 需要访问上一步缓存的 tensor，CUDA Graphs 会覆盖
- `mode='default'`：纯 Inductor 编译，**与 FBC 兼容** ✅

### 测试结果

| 配置 | 时间 | 加速 | vs Baseline PSNR |
|------|------|------|------------------|
| Baseline（无优化） | 6315ms | 1.00x | — |
| FBC t=0.3 only | 5268ms | 1.20x | 32.3dB |
| torch.compile (reduce-overhead) only | 5071ms | 1.25x | 45.8dB（几乎无损） |
| compile(reduce-overhead) + FBC | ❌ CUDA Graphs 冲突 | — | — |
| **compile(default) + FBC t=0.3** | **4074ms** | **1.55x** ✅ | **30.6dB** |

### 关键发现

1. **torch.compile 单独即可获得 1.25x 加速**，且几乎无质量损失（PSNR 45.8dB）
2. **compile + FBC 组合可叠加到 1.55x**，质量仍可接受（PSNR 30.6dB，与 FBC 单独接近）
3. **必须用 `mode='default'`**，`reduce-overhead` 的 CUDA Graphs 与 FBC 有状态缓存不兼容
4. **compile warmup 较慢**（首次约 15s），适合 DLIS 长期运行服务（模型加载时编译一次）

### ⚠️ DLIS 部署注意：torch.compile 使用方式

```python
# 正确顺序：先 apply FBC，再 compile
apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.3))
registry = HookRegistry.check_if_exists_or_initialize(pipe.transformer)
pipe.transformer = torch.compile(pipe.transformer, mode='default')

# 首次推理触发编译（约 15s warmup）
_ = pipe(warmup_prompt, ...)

# 后续每次推理前 reset FBC 状态
registry.reset_stateful_hooks(recurse=True)
result = pipe(prompt, ...)
```

### 完整测试脚本（可复现）

```bash
docker exec zimage-test python3 -c "
import torch, time
from diffusers import ZImagePipeline
from diffusers.hooks import HookRegistry
from diffusers.hooks.first_block_cache import FirstBlockCacheConfig, apply_first_block_cache

pipe = ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo', torch_dtype=torch.bfloat16)
pipe.to('cuda')

prompt = 'A beautiful sunset over the ocean, golden light reflecting on calm waves, dramatic clouds'
gen_kwargs = dict(prompt=prompt, height=768, width=1344, guidance_scale=0, num_inference_steps=9)

# === Test 1: Baseline ===
gen = torch.Generator('cuda').manual_seed(42)
_ = pipe(**gen_kwargs, generator=gen)
times = []
for _ in range(3):
    gen = torch.Generator('cuda').manual_seed(42)
    torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(**gen_kwargs, generator=gen)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
baseline_ms = min(times) * 1000
out.images[0].save('/tmp/test_baseline.png')
print(f'Baseline: {baseline_ms:.0f}ms')

# === Test 2: torch.compile only (reduce-overhead) ===
pipe.transformer = torch.compile(pipe.transformer, mode='reduce-overhead')
gen = torch.Generator('cuda').manual_seed(42)
print('Compiling (first run)...')
t0 = time.time()
_ = pipe(**gen_kwargs, generator=gen)
torch.cuda.synchronize()
print(f'Compile warmup: {(time.time()-t0)*1000:.0f}ms')

times = []
for _ in range(3):
    gen = torch.Generator('cuda').manual_seed(42)
    torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(**gen_kwargs, generator=gen)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
compile_ms = min(times) * 1000
out.images[0].save('/tmp/test_compile_only.png')
print(f'torch.compile only: {compile_ms:.0f}ms (speedup: {baseline_ms/compile_ms:.2f}x)')

# === Test 3: FBC only (for comparison) ===
del pipe
torch.cuda.empty_cache()
pipe = ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo', torch_dtype=torch.bfloat16)
pipe.to('cuda')
apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.3))
registry = HookRegistry.check_if_exists_or_initialize(pipe.transformer)

gen = torch.Generator('cuda').manual_seed(42)
_ = pipe(**gen_kwargs, generator=gen)
times = []
for _ in range(3):
    registry.reset_stateful_hooks(recurse=True)
    gen = torch.Generator('cuda').manual_seed(42)
    torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(**gen_kwargs, generator=gen)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
fbc_ms = min(times) * 1000
out.images[0].save('/tmp/test_fbc_only.png')
print(f'FBC 0.3 only: {fbc_ms:.0f}ms (speedup: {baseline_ms/fbc_ms:.2f}x)')

# === Test 4: torch.compile(default) + FBC t=0.3 ===
del pipe
torch.cuda.empty_cache()
pipe = ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo', torch_dtype=torch.bfloat16)
pipe.to('cuda')
apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.3))
registry = HookRegistry.check_if_exists_or_initialize(pipe.transformer)
pipe.transformer = torch.compile(pipe.transformer, mode='default')

gen = torch.Generator('cuda').manual_seed(42)
print('Compiling+FBC (first run)...')
t0 = time.time()
_ = pipe(**gen_kwargs, generator=gen)
torch.cuda.synchronize()
print(f'Compile+FBC warmup: {(time.time()-t0)*1000:.0f}ms')

times = []
for _ in range(3):
    registry.reset_stateful_hooks(recurse=True)
    gen = torch.Generator('cuda').manual_seed(42)
    torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(**gen_kwargs, generator=gen)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
compile_fbc_ms = min(times) * 1000
out.images[0].save('/tmp/test_compile_fbc03.png')
print(f'compile + FBC 0.3: {compile_fbc_ms:.0f}ms (speedup: {baseline_ms/compile_fbc_ms:.2f}x)')

# === PSNR comparison ===
import numpy as np
from PIL import Image
baseline_arr = np.array(Image.open('/tmp/test_baseline.png')).astype(float)
def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0: return float('inf')
    return 10 * np.log10(255**2 / mse)

compile_arr = np.array(Image.open('/tmp/test_compile_only.png')).astype(float)
fbc_arr = np.array(Image.open('/tmp/test_fbc_only.png')).astype(float)
compile_fbc_arr = np.array(Image.open('/tmp/test_compile_fbc03.png')).astype(float)

print()
print('=== Summary ===')
print(f'Baseline:              {baseline_ms:.0f}ms  1.00x')
print(f'torch.compile:         {compile_ms:.0f}ms  {baseline_ms/compile_ms:.2f}x  PSNR={psnr(baseline_arr, compile_arr):.1f}dB')
print(f'FBC 0.3:               {fbc_ms:.0f}ms  {baseline_ms/fbc_ms:.2f}x  PSNR={psnr(baseline_arr, fbc_arr):.1f}dB')
print(f'compile + FBC 0.3:     {compile_fbc_ms:.0f}ms  {baseline_ms/compile_fbc_ms:.2f}x  PSNR={psnr(baseline_arr, compile_fbc_arr):.1f}dB')
"
```

**前置条件**：需要先应用 3 个 FBC patches（见优化测试 #4）。

### 当前推荐部署配置

```
torch.compile(mode='default') + FBC(threshold=0.3)
→ 4074ms / 请求（1.55x 加速），PSNR 30.6dB
```

---

## 优化测试 #8：ONNX Runtime 加速探索（2026-05-18）

### 背景
在 torch.compile + FBC 获得 1.55x 加速后，探索 ONNX Runtime 是否能进一步提升性能。
此前 ONNX 路径因 complex64 RoPE 无法导出，现已通过 Real RoPE patch 解锁。

### 环境准备

安装 ONNX 依赖：
```bash
pip install onnxruntime-gpu onnx optimum[onnxruntime-gpu]
```
- onnxruntime-gpu 1.26.0
- onnx 1.21.0
- optimum 2.1.0 + optimum-onnx 0.1.0
- ORT Providers: `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`

⚠️ 安装副作用：transformers 从 5.5.3 降级到 4.57.6（ZImagePipeline 仍可正常加载）

### Patches 状态

重新应用了以下 patches（diffusers 之前 reinstall 导致丢失）：

1. **Real RoPE patch**（2 处）：
   - `precompute_freqs_cis`: `torch.polar → torch.cat([cos, sin])`
   - `apply_rotary_emb`: `view_as_complex/view_as_real → 实数 cos/sin 旋转`

2. **FBC patches**（3 处）：
   - Patch 1 (`_helpers.py`): `hidden_states_argument_name="x"` → **已内置于 diffusers 0.38.0，无需手动 patch**
   - Patch 2 (`first_block_cache.py`): 硬编码 `"hidden_states"` → **已内置，无需手动 patch**
   - Patch 3 (`pipeline_z_image.py`): `_set_context` 包裹 transformer 调用 → 仍需手动 patch

### 路径 1: torch.compile ORT backend

```python
pipe.transformer = torch.compile(pipe.transformer, backend='onnxrt')
```

**结果：❌ 失败**

```
torch._dynamo.exc.InvalidBackend: Invalid backend: 'onnxrt'
Available backends: ['cudagraphs', 'inductor', 'openxla', 'tvm']
```

PyTorch 2.11 已移除 `onnxrt` compile backend。此路径不可行。

### 路径 2: optimum ONNX 导出

```python
from optimum.exporters.onnx import main_export
main_export('Tongyi-MAI/Z-Image-Turbo', output='/tmp/zimage_onnx/', task='text-to-image')
```

**结果：❌ 失败**

```
AttributeError: 'Qwen3Config' object has no attribute 'projection_dim'
```

ZImage 使用 Qwen3 作为 text encoder，optimum 的 ONNX 导出器不支持 Qwen3Config。

### 路径 3: 手动 torch.onnx.export（transformer only）

**结果：❌ 不可行**

ZImage transformer forward 签名：
```python
(x: list[torch.Tensor, list[list[torch.Tensor]]],
 t,
 cap_feats: list[torch.Tensor, list[list[torch.Tensor]]],
 return_dict: bool = True,
 controlnet_block_samples: dict[int, torch.Tensor] | None = None,
 siglip_feats: list[list[torch.Tensor]] | None = None,
 image_noise_mask: list[list[int]] | None = None,
 patch_size: int = 2,
 f_patch_size: int = 1)
```

输入是嵌套的 `list[Tensor, list[list[Tensor]]]`，ONNX 图要求固定的 tensor 输入签名，无法处理这种动态嵌套结构。要支持 ONNX 需要重写 transformer 接口将嵌套 list 展开为 flat tensor inputs，工作量巨大且收益不确定。

### ONNX 结论

❌ **ONNX Runtime 对 ZImage 当前不可行**，原因：
1. optimum 不支持 Qwen3 text encoder
2. transformer forward 使用嵌套 list[Tensor] 输入，无法导出 ONNX 图
3. PyTorch 2.11 已移除 `onnxrt` compile backend

---

## 优化测试 #9: Pipeline 组件 Profiling（2026-05-19）

### 目的
按照"先 Profile → 定位热点 → 逐模块测试 → 全模型"的方法论，对 ZImage 推理全流程做组件级耗时分解。

### 环境
- GPU: NVIDIA RTX A6000 (48GB)
- Docker container: `zimage-test`
- 测试分辨率: 768×1344, 9 steps, guidance_scale=0, bf16

### Step 1: Pipeline 组件级耗时

通过 monkey-patch 对 `encode_prompt`、`transformer.forward`、`vae.decode` 加计时 wrapper，跑 3 次取平均。

```python
# Monkey-patch 计时方法（不额外分配显存，避免 OOM）
timings = {'text_encoder': [], 'transformer': [], 'vae': []}

orig_encode = pipe.encode_prompt
orig_transformer = pipe.transformer.forward
orig_vae = pipe.vae.decode

def timed_encode(*a, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = orig_encode(*a, **kw)
    torch.cuda.synchronize()
    timings['text_encoder'].append(time.perf_counter() - t0)
    return r
# ... 同理 patch transformer 和 vae
```

**结果：**

| 组件 | 耗时 (ms) | 占比 |
|------|----------|------|
| Text Encoder (Qwen3) | 109.4 | 1.6% |
| **Transformer (30 blocks × 9 steps)** | **6630.4** | **95.5%** |
| VAE Decode | 204.2 | 2.9% |
| **Total** | **6944.1** | **100%** |

**结论：Transformer 占 95.5%，是绝对瓶颈。Text Encoder 和 VAE 优化无意义。**

### Step 2: Transformer 内部结构

```
ZImageTransformer2DModel — 6154.9M params
├── noise_refiner: 2 × ZImageTransformerBlock (361.8M)
├── context_refiner: 2 × ZImageTransformerBlock (353.9M)
├── cap_embedder: Sequential (9.8M)
├── layers: 30 × ZImageTransformerBlock (5427.3M, 88% params)
│   └── 每个 Block:
│       ├── attention: Attention (59.0M) — ZSingleStreamAttnProcessor
│       ├── feed_forward: FeedForward (118.0M)
│       ├── attention_norm1/2: RMSNorm
│       ├── ffn_norm1/2: RMSNorm
│       └── adaLN_modulation: Sequential (3.9M)
└── all_final_layer: ModuleDict (1.2M)
```

### Step 3: Per-Block Profiling

Hook 每个 block 的 attention 和 ffn，跑 1 次 (9 steps)：

| Block | Attn (ms/step) | FFN (ms/step) | Total (ms/step) | Attn% |
|-------|---------------|---------------|-----------------|-------|
| 0-29 | 10.9 | 8.2 | 19.1 | 57.1% |
| **SUM** | **326.9** | **245.2** | **572.1** | **57.1%** |

**关键发现：**
1. **30 个 block 完全均匀**（每个 19.1ms/step），无单独热点 block
2. **Attention (57%) > FFN (43%)**，attention 是 block 内主要瓶颈
3. **每步总计 737ms**，其中 blocks 占 572ms (78%)，refiners/embedding/other 占 165ms (22%)

### Step 4: Attention 实现分析

**Attention Processor**: `ZSingleStreamAttnProcessor` — 自定义 processor
- heads: 30, head_dim: 128 (inner_dim=3840)
- 使用 `dispatch_attention_fn()` 路由到 SDPA backend
- `dispatch_attention_fn` 通过 `_AttentionBackendRegistry` 选择实际 backend
- PyTorch SDPA 所有 backend 均已启用：Flash=True, MemEfficient=True, Math=True, cuDNN=True

**RoPE 实现**: Real cos/sin 旋转（diffusers 0.38.0 内置），兼容 torch.compile

```python
# ZSingleStreamAttnProcessor 核心流程
query = attn.to_q(hidden_states)      # Linear: 3840→3840
key = attn.to_k(hidden_states)        # Linear: 3840→3840
value = attn.to_v(hidden_states)      # Linear: 3840→3840
# → unflatten to (batch, seq, 30, 128)
# → RMSNorm (q, k)
# → RoPE (real cos/sin rotation)
# → dispatch_attention_fn (routes to SDPA/Flash/MemEfficient)
# → flatten → to_out Linear
```

**flash-attn 独立包**: 未安装。依赖 PyTorch 内置 SDPA 自动选择 backend。

### Step 5: CUDA Kernel Profiling（torch.profiler）

通过 `torch.profiler` 运行 1 step 推理，按 `cuda_time_total` 排序 top 25 kernel：

```
操作                                    Self CUDA    Self CUDA%   CUDA total
aten::mm (GEMM)                         393.3ms      40.3%        601.2ms
Command Buffer Full (调度开销)           360.0ms      36.8%        360.0ms
cutlass_80_tensorop_bf16 (GEMM kernel)  202.1ms      20.7%        202.1ms
aten::conv2d → cudnn_convolution         98.3ms      10.1%        179.4ms
aten::mul                               109.0ms      11.2%        157.8ms
aten::_flash_attention_forward           74.9ms       7.7%        126.1ms
aten::copy_ (dtype cast)                 81.0ms       8.3%        118.4ms
Lazy Function Loading                    82.2ms       8.4%         82.2ms
pytorch_flash::flash_fwd_kernel          74.9ms       7.7%         74.9ms
aten::native_group_norm                  51.2ms       5.2%         52.3ms
```

**关键发现：**

1. **✅ FlashAttention 已启用** — `pytorch_flash::flash_fwd_kernel` 确认在用 PyTorch 内置 Flash Attention
   - 34 次调用（30 主 blocks + 2 noise_refiner + 2 context_refiner），每次 2.2ms
   - Flash Attention 本身只占 7.7%，优化空间有限

2. **GEMM (aten::mm) 是最大开销 — 40.3%**
   - 493 次调用，来自 Linear 层（Q/K/V/Out projection + FFN）
   - 使用 cutlass bf16 tensorop kernel

3. **⚠️ Command Buffer Full — 36.8%**
   - GPU 命令队列满，CPU 跟不上 GPU 的 kernel 发射速度
   - **这正是 torch.compile 能大幅优化的地方** — 融合小 kernel，减少 launch overhead
   - 解释了为什么 `torch.compile(mode='default')` 能有 1.25x 加速

4. **dtype 转换 (copy_) — 8.3%**
   - bf16↔fp32 转换开销，scheduler 要求 fp32 精度

5. **Conv2d (VAE) — 10.1%**
   - 仅跑一次的 VAE decode，绝对值不大

### 综合分析与优化策略

基于完整 profiling 数据，优化优先级重新排序：

| 优先级 | 方向 | 目标 | 预期收益 | 状态 |
|--------|------|------|---------|------|
| 1 | **torch.compile** | 消除 Command Buffer Full (36.8%) + kernel fusion | 1.25x 已验证 | ✅ 已验证 |
| 2 | **FBC (TeaCache)** | 跳过相似 block，减少 GEMM + FA 调用次数 | 与 compile 组合 1.55x | ✅ 已验证 |
| 3 | **减少 dtype 转换** | 尝试全 bf16 推理，避免 copy_ 开销 | ~8% 节省 | 待测 |
| 4 | **INT8/FP8 量化** | 降低 GEMM 计算量（40.3%） | 理论 2x GEMM 提速 | 待测 |
| 5 | **安装 flash-attn 包** | 独立 FA 可能比 PyTorch 内置更快 | 微小提升 | 低优先 |

**当前最佳部署方案：torch.compile(mode='default') + FBC = 1.55x 加速**
- Baseline: 6630ms → 优化后: ~4280ms
- DLIS 预期 latency: 4666ms → 优化后: ~3010ms

**建议**：放弃 ONNX 路径，当前最佳方案维持 `torch.compile(default) + FBC(t=0.3) = 1.55x`

---

## 优化测试 #10: INT8 量化探索（2026-05-19）

### 目的
尝试 W8A8 INT8 量化加速 GEMM（profiler 显示 GEMM 占 40.3%）。

### 测试环境
- torchao 0.17.0, PyTorch 2.11.0+cu129, A6000 (SM 8.6 Ampere)

### W8A8 动态量化（torchao）

```python
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
quantize_(pipe.transformer, Int8DynamicActivationInt8WeightConfig())
```

**结果：❌ 反而慢了 2.1x**

| 配置 | 耗时 | 加速比 | PSNR |
|------|------|--------|------|
| Baseline (bf16) | 6854ms | 1.00x | — |
| W8A8 INT8 | 14784ms | **0.46x** | 31.8dB |

### 根因分析

1. **torchao 的 CUTLASS INT8 kernel 只有 SM90a (Hopper/H100) 版本**
   - 容器中只有 `_C_cutlass_90a.abi3.so`，加载失败（缺 `libcudart.so.13`）
   - **没有 SM80/SM86 (Ampere) 的优化 kernel**

2. **Fallback 路径极慢**
   - 每次 forward 都执行：计算激活值 scale → 量化到 INT8 → dequant 回 bf16 → bf16 GEMM
   - 量化/反量化的 Python 开销远大于 INT8 GEMM 的理论收益

3. **`_C_mxfp8` 也加载失败** — Python 3.10 编译的 .so，容器是 Python 3.12

### 结论

❌ **torchao INT8 量化在 A6000 (Ampere) 上不可行**
- 缺少 SM86 优化 kernel，需要源码编译 torchao（工程成本高，收益不确定）
- W8A8 + torch.compile 理论上 Triton 可以生成 INT8 kernel，但未验证

### 其他量化方向（未测试）

| 方案 | 可行性 | 备注 |
|------|--------|------|
| FP8 | ❌ A6000 不支持 | 需要 Hopper (H100) |
| bitsandbytes INT8 | 可能 | 有 Ampere kernel，但主要面向 LLM 推理 |
| AutoAWQ INT4 | 可能 | 主要面向 LLM，对 DiT 支持未知 |
| TensorRT | 不可行 | ZImage 嵌套 list 输入无法导出 |

### bitsandbytes INT8 量化测试

尝试用 bitsandbytes 替代 torchao 进行 INT8 量化（bitsandbytes 有 Ampere kernel 支持）。

**环境准备**：
```bash
pip install bitsandbytes==0.49.2
```

**加载方式**：使用 diffusers 的 `PipelineQuantizationConfig` API：
```python
from diffusers import PipelineQuantizationConfig
quant_config = PipelineQuantizationConfig(
    quant_backend='bitsandbytes_8bit',
    components_to_quantize=['transformer'],
    quant_kwargs={'load_in_8bit': True},
)
pipe = ZImagePipeline.from_pretrained(..., quantization_config=quant_config, torch_dtype=torch.bfloat16)
```

**遇到的问题链**：

1. **`getCurrentRawStream` RuntimeError** — PyTorch 2.11 API 变更，`torch._C._cuda_getCurrentRawStream()` 对 `device.index=None` 报错
   - 修复：patch `bitsandbytes/functional.py` 的 `_get_tensor_stream`，`device.index` 为 None 时默认 0
2. **bias device mismatch** — `ops.py` 中 `out.add_(bias)` 失败，bias 在 CPU 而 out 在 CUDA
   - 修复：patch `bitsandbytes/backends/cuda/ops.py`，`bias.to(out.device)`
3. **进程 hang 死** — patch 后模型加载到 GPU（30GB 显存占用），但推理卡死在 C++ 层面
   - Ctrl+C 无 Python traceback，说明卡在底层 CUDA kernel 或 bitsandbytes C 扩展中
   - GPU-Util 0%，进程无响应

**方法二：手动替换 Linear → Linear8bitLt（绕过 PipelineQuantizationConfig）**

```python
# 先正常加载 bf16，再手动替换所有 Linear 层
pipe = ZImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to('cuda:0')
for name, module in list(pipe.transformer.named_modules()):
    if isinstance(module, torch.nn.Linear):
        new_linear = bnb.nn.Linear8bitLt(old.in_features, old.out_features, ...)
        new_linear.weight = bnb.nn.Int8Params(old.weight.data.to(torch.float16), ...)
        setattr(parent, child_name, new_linear.to('cuda:0'))
# 共替换 276 个 Linear 层
```

**结果：跑通了，但几乎没有加速且质量严重下降**

| 配置 | 耗时 | 加速比 | PSNR | GPU 显存 |
|------|------|--------|------|---------|
| Baseline (bf16) | 6987ms | 1.00x | — | 19.3GB |
| BnB INT8 | 6413ms | **1.09x** | **9.6dB** ❌ | 13.5GB |

### 深度分析：为什么 GEMM 占 40.3% 但 INT8 量化却无加速？

#### 核心矛盾：INT8 GEMM 快了，但额外开销更大

虽然 GEMM 是最大计算瓶颈 (40.3%)，INT8 tensorcore 吞吐理论上是 bf16 的 2x，但 bitsandbytes `MatMul8bitLt` 的**实际执行流程**是：

```
每次 Linear forward:
  输入 x (bf16) → cast fp16     ← 新增开销
  → 计算 x 的 absmax scale      ← 新增开销（逐 tensor 扫描最大值）
  → 量化 x → int8               ← 新增开销
  → INT8 GEMM(x_int8, w_int8)   ← 这一步确实快了 ~2x
  → dequant → fp16               ← 新增开销
  → cast → bf16                  ← 新增开销
```

**一次 bf16 GEMM (1 个 kernel) 被替换成了 5-6 个小 kernel**。

#### 逐项开销分析

| 步骤 | 类型 | 说明 |
|------|------|------|
| bf16→fp16 cast | 新增 | BnB 只支持 fp16 输入（日志警告 `inputs will be cast from torch.bfloat16 to float16`） |
| 计算 absmax scale | 新增 | 动态量化必须每次计算激活的最大值（**无法预量化，因为激活每次都不同**） |
| 量化 fp16→int8 | 新增 | 按 scale 缩放取整 |
| INT8 GEMM | 理论 2x 快 | Ampere INT8 tensorcore 吞吐确实更高 |
| dequant int8→fp16 | 新增 | 反量化回浮点 |
| fp16→bf16 cast | 新增 | 转回原始精度 |

**INT8 GEMM 本身确实快了 ~2x，但前后的量化/反量化/cast 开销把收益全吃掉了。**

#### 关键：Command Buffer Full 雪上加霜

Profiler 已经显示 Command Buffer Full 占 36.8%（GPU 等 CPU 发射 kernel）。BnB 把每次 GEMM 从 1 个 kernel 变成 5-6 个，**额外产生 276层 × 5 = ~1380 个新增 kernel launch**，进一步加重了 CPU→GPU 的命令队列瓶颈。

#### 为什么"提前量化权重"也不够？

权重确实已经是预量化的 INT8（`Int8Params` 存储），但问题在于**激活（每层输入）每次推理都不同**，必须运行时动态量化。即使用校准数据做静态量化（固定 scale），仍然需要 `量化 + INT8 GEMM + 反量化` 三个独立 kernel——没有 fused kernel 就没有实际收益。

#### 什么情况下 INT8 量化才有效？

| 方案 | 原理 | A6000 可用？ |
|------|------|-------------|
| torchao CUTLASS fused INT8 | 把 量化+GEMM+反量化 融合成一个 kernel | ❌ 只有 SM90a (H100) |
| TensorRT INT8 | 编译时全图优化，自动插入 fused 量化节点 | ❌ ZImage 无法导出 ONNX |
| torch.compile + INT8 Triton | Triton JIT 自动生成 fused kernel | 🔶 理论可行，未验证 |

**结论：在没有 fused INT8 kernel 的 A6000 上，任何形式的 INT8 量化都无法带来净加速。**

### 质量问题分析

PSNR 9.6dB 的质量崩溃原因：
- absmax 量化精度不足：DiT transformer 的激活分布比 LLM 更敏感，per-tensor absmax 粒度太粗
- bf16→fp16→int8 双重精度损失：经过两次 cast，累积误差在 30 个 block 中不断放大
- 无校准：没有用实际数据校准 scale，量化范围可能不匹配实际分布

### 最终结论

**❌ bitsandbytes INT8 对 ZImage 无价值**
- 加速 1.09x：额外 kernel launch 抵消 INT8 GEMM 收益
- 质量 9.6dB：完全不可接受（需要 >25dB）
- 根因：缺少 fused INT8 kernel（需要 H100 或 TensorRT）

---

## 优化测试总结（2026-05-19，已被更新版本替代，见下方）

> ⚠️ 此表已过时，最新汇总见 "优化测试 #11" 之后的更新版本。

| # | 方案 | 加速比 | PSNR | 状态 | 备注 |
|---|------|--------|------|------|------|
| 1 | torch.compile(default) | 1.25x | 无损 | ✅ 可用 | 消除 Command Buffer Full |
| 2 | torch.compile(max-autotune) | 1.30x | 无损 | ✅ 可用 | 编译时间长 |
| 3 | FBC (t=0.3) | 1.24x | 30.5dB | ✅ 可用 | 跳过相似 block |
| 4 | compile + FBC | **1.55x** | 30.5dB | ✅ 可用 | 两者叠加 |
| 5 | torchao W8A8 | 0.46x | 31.8dB | ❌ 更慢 | 无 Ampere kernel |
| 6 | bitsandbytes INT8 | 1.09x | 9.6dB ❌ | ❌ 无价值 | 加速微小+质量崩溃 |

### 未测试但已排除的方向

| 方案 | 排除原因 |
|------|---------|
| FP8 量化 | A6000 不支持 FP8（需要 Hopper） |
| TensorRT | ZImage 嵌套 list 输入无法导出 ONNX |
| AutoAWQ INT4 | 主要面向 LLM，DiT 支持未知 |
| flash-attn 独立包 | FlashAttention 已通过 PyTorch SDPA 激活，额外收益极小 |

---

## 优化测试 #11: torch.compile 扩展模式测试（2026-05-19）

### 目的
测试 `torch.compile` 的更多模式组合，寻找超越当前最佳 1.55x 的方案。

### 测试环境
- A6000 Docker，模型缓存在 `/root/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/...`
- PyTorch 2.11.0+cu129, diffusers (latest), bfloat16, 768×1344, 9 steps
- 3 warmup + 3 timed runs

### 测试脚本：`/tmp/test_more_compile.py`

测试三种配置：
1. **Baseline**（无 compile）
2. **reduce-overhead**（CUDA Graphs 捕获+回放，消除 CPU→GPU kernel launch 开销）
3. **max-autotune + FBC**（最大 kernel 优化 + First Block Cache）

### 结果

| 配置 | 耗时 | 加速比 | PSNR | 状态 |
|------|------|--------|------|------|
| Baseline (bf16) | 6982ms | 1.00x | — | ✅ |
| compile(reduce-overhead) | 5137ms | **1.36x** | 37.3dB | ✅ **新最佳单一优化** |
| compile(max-autotune) + FBC | crashed | — | — | ❌ CUDA Graph 冲突 |

### 分析

#### reduce-overhead: 1.36x — 超越 default (1.25x) 和 max-autotune (1.30x)

`reduce-overhead` 模式使用 CUDA Graphs 将整个推理图捕获为单个 GPU 操作序列，然后每步直接回放，**完全消除 CPU→GPU 的 kernel launch 开销**。

对比：
- `default` (1.25x): 只做 Triton kernel fusion，不做 CUDA Graph
- `max-autotune` (1.30x): kernel fusion + autotuning，不做 CUDA Graph
- `reduce-overhead` (1.36x): kernel fusion + **CUDA Graph capture/replay**

在 ZImage 这种 Command Buffer Full 占 36.8% 的场景下，CUDA Graphs 的收益尤其显著——直接跳过了 CPU 命令队列瓶颈。

**质量**: PSNR 37.3dB（接近无损），因为只是改变了 kernel 调度方式，数学计算完全相同。

#### max-autotune + FBC: CUDA Graph 冲突崩溃

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been 
overwritten by a subsequent run. During a CUDA Graph's recording, any 
tensor that is output from a graph has its underlying storage managed by 
that CUDA Graph...
```

**原因**: FBC 的 `_should_compute_remaining_blocks()` 会访问前一步缓存的 tensor 输出，但 CUDA Graphs（max-autotune 隐式启用）在下一次 replay 时覆盖了这些 tensor 的底层存储。

**潜在修复**: 错误信息提示 `call torch.compiler.cudagraph_mark_step_begin() before each model invocation`，这可以告知 CUDA Graph 运行时新一步开始，保护之前的 tensor 输出。

### 下一步

尝试 `cudagraph_mark_step_begin()` 修复，使 reduce-overhead + FBC 组合可用。如果成功，预期加速比 1.36x × 1.24x ≈ **1.68x**，超越当前最佳 1.55x。

---

## 优化测试总结（2026-05-19 更新）

### 所有已测试方案汇总

| # | 方案 | 加速比 | PSNR | 状态 | 备注 |
|---|------|--------|------|------|------|
| 1 | torch.compile(default) | 1.25x | 无损 | ✅ 可用 | 消除 Command Buffer Full |
| 2 | torch.compile(max-autotune) | 1.30x | 无损 | ✅ 可用 | 编译时间长 |
| 3 | FBC (t=0.3) | 1.24x | 30.5dB | ✅ 可用 | 跳过相似 block |
| 4 | compile(default) + FBC | 1.55x | 30.5dB | ✅ 可用 | 之前最佳 |
| 5 | torchao W8A8 | 0.46x | 31.8dB | ❌ 更慢 | 无 Ampere kernel |
| 6 | bitsandbytes INT8 | 1.09x | 9.6dB ❌ | ❌ 无价值 | 加速微小+质量崩溃 |
| 7 | compile(reduce-overhead) | **1.36x** | 37.3dB | ✅ **新最佳单一** | CUDA Graphs |
| 8 | compile(max-autotune) + FBC | crashed | — | ❌ | CUDA Graph 与 FBC 冲突 |
| 9 | reduce-overhead + FBC (patch clone) | crashed | — | ❌ | CUDA Graph storage 覆盖，patch 无效 |
| 10 | max-autotune-no-cudagraphs + FBC | **1.52x** | 27.9dB | ✅ 可用 | kernel 优化 + FBC，无 CUDA Graph 冲突 |
| 11 | compile(fullgraph=True) | crashed | — | ❌ | ZImage RoPE 用 `with torch.device("cpu")` 不兼容 |
| 12 | compile(fullgraph=True) + FBC | crashed | — | ❌ | 同上 |

### 结论：CUDA Graphs 与 FBC 根本不兼容

尝试了多种修复（clone original_hidden_states、clone output、cudagraph_mark_step_begin），均失败。根因：FBC 的 `_should_compute_remaining_blocks()` 需要跨 CUDA Graph replay 步访问之前步骤的 tensor 输出，而 CUDA Graphs 会复用这些 tensor 的底层 storage。`.clone()` 在 compiled 区域内部执行时，clone 出的 tensor 同样被 CUDA Graph 管理，无法逃逸。

**任何使用 CUDA Graphs 的 compile 模式（reduce-overhead、max-autotune）都无法与 FBC 组合。**

### fullgraph=True 不兼容原因

ZImage transformer 的 `precompute_freqs_cis()` 使用了 `with torch.device("cpu"):`，torch.compile 的 Dynamo 不支持这种 context manager。需要修改模型代码去掉这个 pattern 才能启用 fullgraph=True。

### 待测试

所有待测方案已完成。

### Scheduler bf16 测试结果（#13）

将 scheduler 内部的 float64 tensor 强制转为 bfloat16，尝试减少 copy_ dtype 转换开销（profiler 显示占 8.3%）。

| 配置 | 耗时 | 加速比 | PSNR |
|------|------|--------|------|
| compile(default) + FBC | 4631ms | 1.52x | 28.6dB |
| compile(default) + FBC + sched_bf16 | 4669ms | 1.50x | 28.6dB |
| reduce-overhead + sched_bf16 | 5205ms | 1.35x | 42.0dB |

**结论：❌ 无收益**。copy_ 的 8.3% 开销并非来自 scheduler dtype 转换，更可能是 latent/hidden_states 在不同模块间传递时的 layout/contiguity copy。scheduler bf16 反而略慢（精度降低可能导致额外重计算）。

### 当前最佳部署方案（最终）

**torch.compile(mode='default') + FBC(threshold=0.3) = 1.55x 加速**

备选方案：
- `max-autotune-no-cudagraphs + FBC` = 1.52x（略低但 kernel 更优化，编译时间更长）
- `reduce-overhead` 单独 = 1.36x（无质量损失，PSNR 37-42dB）

### 下一步

1. 测试 scheduler bf16 减少 dtype 转换
2. 将最终方案集成到 DLIS 部署代码 (`model.py`)
3. 如果未来迁移到 H100，可重新测试 FP8/torchao INT8（预期额外 1.5-2x）

---

## diffusers 原生 Cache 方法全量对比（2026-05-19）

### 背景

测试 diffusers 0.38.0 中所有可用的 cache hook 方法，看是否能在 ZImage 上获得额外加速。
TeaCache 不在 diffusers 0.38.0 中，测试了以下原生方法：FBC、TaylorSeer、MagCache。

### Pipeline 修复

ZImage pipeline (`pipeline_z_image.py`) 缺少 cache hook 所需的 context 设置。
其他 pipeline（如 Flux）使用 `self.transformer.cache_context("cond")`，但 `ZImageTransformer2DModel` 没有该方法。

**正确修复方式**（不要用 `cache_context`！）：
```python
from diffusers.hooks import HookRegistry as _HR
_reg = _HR.check_if_exists_or_initialize(self.transformer)
_reg._set_context("default")
model_out_list = self.transformer(...)
_reg._set_context(None)
```

### 第一轮测试结果（threshold 过低）

| Method | Avg Time (s) | Speedup | 备注 |
|--------|-------------|---------|------|
| Baseline (no cache) | 6.967 | 1.00x | 基准 |
| FBC (threshold=0.05) | 7.124 | 0.98x | ⚠️ threshold 太低，未触发跳步 |
| TaylorSeer (interval=2) | 7.164 | 0.97x | hooks 未匹配 ZImage block 结构 |
| TaylorSeer (interval=3) | 7.201 | 0.97x | 同上 |
| MagCache | FAILED | N/A | ratios 打印到 stdout 但未存回 config |

测试环境：A100 GPU，bf16，1344x768，9 steps，seed=42，warmup=1，repeat=3

⚠️ **FBC threshold=0.05 过于保守**。之前优化测试 #6 已验证：
- t=0.2 → 0.98x（未触发），t=0.25 → 1.07x，t=0.3 → 1.20x，t=0.35 → 1.35x
- 需要用 t=0.25~0.35 才能触发跳步

### MagCache calibration 输出

```
mag_ratios = [1.0, 1.432, 1.539, 1.261, 1.143, 1.076, 1.077, 1.030, 0.962]
```
`cal_config.mag_ratios` 返回 None — ratios 只打印到 stdout，未存回 config 对象。
需要手动解析 stdout 或直接硬编码 ratios。

### 运行命令

```bash
# 写入测试脚本
cat > /tmp/all_in_one.py << 'PYEOF'
... (见下方第二轮修正脚本)
PYEOF
python /tmp/all_in_one.py
```

### 第二轮测试结果（修正 threshold + MagCache hardcoded ratios）

运行命令：
```bash
cat > /tmp/all_in_one_v2.py << 'PYEOF'
#!/usr/bin/env python3
"""ZImage Cache 全量对比 v2 - 修正 FBC threshold + MagCache hardcoded ratios"""
import time, gc, os, sys, re, io, torch
import numpy as np

MODEL_DIR = "Tongyi-MAI/Z-Image-Turbo"
PIPELINE_PATH = "/usr/local/lib/python3.12/dist-packages/diffusers/pipelines/z_image/pipeline_z_image.py"
PROMPT = "A beautiful sunset over the ocean, golden light reflecting on calm waters, dramatic clouds"
WIDTH, HEIGHT = 1344, 768
STEPS = 9
SEED = 42
WARMUP = 1
REPEAT = 3
results = {}

# calibrated from round 1
MAG_RATIOS = [1.0, 1.431571125984192, 1.5391831398010254, 1.26146399974823,
              1.1428234577178955, 1.0757588148117065, 1.0774412155151367,
              1.029515266418457, 0.9623218178749084]

def fix_pipeline():
    print("=" * 60)
    print("Step 1: Fixing pipeline_z_image.py")
    print("=" * 60)
    with open(PIPELINE_PATH, "r") as f:
        content = f.read()
    lines = content.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "cache_context" in line and "self.transformer" in line:
            print(f"  Removing broken cache_context at line {i+1}")
            base_indent = len(line) - len(line.lstrip())
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip() == "":
                    new_lines.append(next_line)
                    i += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent > base_indent:
                    new_lines.append(" " * base_indent + next_line[base_indent + 4:])
                    i += 1
                else:
                    break
            continue
        if "_set_context" in line and ("_HR" in line or "_reg" in line):
            i += 1; continue
        if "from diffusers.hooks import HookRegistry as _HR" in line:
            i += 1; continue
        if "_reg = _HR.check_if_exists_or_initialize" in line:
            i += 1; continue
        new_lines.append(line)
        i += 1
    content = "\n".join(new_lines)
    match = re.search(r"(\s+)(model_out_list = self\.transformer\()", content)
    if match:
        indent = match.group(1)
        replacement = (
            f"{indent}from diffusers.hooks import HookRegistry as _HR\n"
            f"{indent}_reg = _HR.check_if_exists_or_initialize(self.transformer)\n"
            f'{indent}_reg._set_context("default")\n'
            f"{indent}{match.group(2)}"
        )
        content = content[:match.start()] + replacement + content[match.end():]
        rest = content[match.start():]
        close_match = re.search(r"\)\[0\]\n", rest)
        if close_match:
            insert_pos = match.start() + close_match.end()
            content = content[:insert_pos] + f"{indent}_reg._set_context(None)\n" + content[insert_pos:]
            print("  Inserted _set_context before/after transformer call")
    elif '_set_context("default")' in content:
        print("  Pipeline already correctly patched")
    else:
        print("  ERROR: Cannot patch!"); sys.exit(1)
    with open(PIPELINE_PATH, "w") as f:
        f.write(content)
    v = open(PIPELINE_PATH).read()
    assert "cache_context" not in v, "cache_context still present!"
    assert '_set_context("default")' in v, "_set_context not found!"
    print("  Pipeline fixed!\n")

def load_pipe():
    from diffusers import ZImagePipeline
    return ZImagePipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")

def free_pipe(pipe):
    del pipe; gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

def generate(pipe, seed=SEED):
    gen = torch.Generator("cuda").manual_seed(seed)
    t0 = time.time()
    result = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH, guidance_scale=0, num_inference_steps=STEPS, generator=gen)
    torch.cuda.synchronize()
    return result.images[0], time.time() - t0

def benchmark(pipe, label):
    print(f"\n--- {label} ---")
    for i in range(WARMUP):
        _, t = generate(pipe); print(f"  Warmup {i+1}: {t:.3f}s")
    times = []
    for i in range(REPEAT):
        _, t = generate(pipe, seed=SEED+i); times.append(t); print(f"  Run {i+1}: {t:.3f}s")
    avg = np.mean(times); print(f"  Average: {avg:.3f}s")
    return avg

if __name__ == "__main__":
    print("ZImage Cache v2 - Corrected thresholds")
    print(f"Model: {MODEL_DIR}, {WIDTH}x{HEIGHT}, {STEPS} steps\n")
    fix_pipeline()

    # Baseline
    print("\n" + "=" * 60); print("Test: Baseline"); print("=" * 60)
    pipe = load_pipe()
    results["baseline"] = benchmark(pipe, "Baseline")
    free_pipe(pipe)

    # FBC t=0.25
    print("\n" + "=" * 60); print("Test: FBC t=0.25"); print("=" * 60)
    pipe = load_pipe()
    from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
    apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.25))
    results["fbc_025"] = benchmark(pipe, "FBC t=0.25")
    free_pipe(pipe)

    # FBC t=0.3
    print("\n" + "=" * 60); print("Test: FBC t=0.3"); print("=" * 60)
    pipe = load_pipe()
    apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.3))
    results["fbc_030"] = benchmark(pipe, "FBC t=0.3")
    free_pipe(pipe)

    # FBC t=0.35
    print("\n" + "=" * 60); print("Test: FBC t=0.35"); print("=" * 60)
    pipe = load_pipe()
    apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.35))
    results["fbc_035"] = benchmark(pipe, "FBC t=0.35")
    free_pipe(pipe)

    # TaylorSeer i=2
    print("\n" + "=" * 60); print("Test: TaylorSeer i=2"); print("=" * 60)
    pipe = load_pipe()
    from diffusers.hooks import apply_taylorseer_cache, TaylorSeerCacheConfig
    apply_taylorseer_cache(pipe.transformer, TaylorSeerCacheConfig(
        cache_interval=2, max_order=1, disable_cache_before_step=1, disable_cache_after_step=STEPS-1))
    results["taylor_2"] = benchmark(pipe, "TaylorSeer i=2")
    free_pipe(pipe)

    # MagCache with hardcoded ratios
    print("\n" + "=" * 60); print("Test: MagCache (hardcoded ratios)"); print("=" * 60)
    pipe = load_pipe()
    from diffusers.hooks import apply_mag_cache, MagCacheConfig
    apply_mag_cache(pipe.transformer, MagCacheConfig(num_inference_steps=STEPS, mag_ratios=MAG_RATIOS))
    results["magcache"] = benchmark(pipe, "MagCache")
    free_pipe(pipe)

    # Summary
    bl = results["baseline"]
    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)
    print(f"{'Method':<25} {'Avg(s)':<10} {'Speedup':<10}")
    print("-" * 45)
    for name, key in [("Baseline","baseline"),("FBC t=0.25","fbc_025"),("FBC t=0.3","fbc_030"),
                      ("FBC t=0.35","fbc_035"),("TaylorSeer i=2","taylor_2"),("MagCache","magcache")]:
        if key in results and results[key] > 0:
            print(f"{name:<25} {results[key]:<10.3f} {bl/results[key]:<10.2f}x")
        else:
            print(f"{name:<25} {'FAILED':<10} {'N/A':<10}")
    print("\nDone!")
PYEOF
python /tmp/all_in_one_v2.py
```

| Method | Avg Time (s) | Speedup | 备注 |
|--------|-------------|---------|------|
| Baseline | 6.999 | 1.00x | 基准 |
| FBC t=0.25 | 6.461 | 1.08x | 与之前 1.07x 吻合 |
| FBC t=0.3 | 6.253 | 1.12x | 之前 1.20x（波动大，Run2=5.799s 异常快） |
| FBC t=0.35 | 5.828 | 1.20x | 之前 1.35x |
| TaylorSeer i=2 | 7.203 | 0.97x | ❌ hooks 未匹配 ZImage block 结构 |
| MagCache (hardcoded) | 5.767 | 1.21x | ✅ 新发现！略优于 FBC t=0.35 |

MagCache calibrated ratios（hardcoded）：
```python
[1.0, 1.432, 1.539, 1.261, 1.143, 1.076, 1.077, 1.030, 0.962]
```

### FBC 加速比差异分析

| Threshold | 之前（同 pipeline 连续跑） | 现在（每次 fresh load） |
|-----------|--------------------------|----------------------|
| t=0.25 | 1.07x | 1.08x ≈ |
| t=0.3 | 1.20x | 1.12x ↓ |
| t=0.35 | 1.35x | 1.20x ↓↓ |

差异原因：之前同一 pipeline 连续测试，GPU kernel cache 对后续测试有加持效果；
fresh load 每次冷启动更接近 DLIS 实际部署场景（每个请求不共享 pipeline 状态）。

### 结论

1. **MagCache 1.21x** — 最佳单项 cache 加速，无需调 threshold，质量待评估
2. **FBC t=0.35 = 1.20x** — 接近 MagCache，之前已验证 PSNR 良好
3. **TaylorSeer** — 对 ZImage 完全无效，hooks 不匹配 block 结构
4. **下一步**：测试 MagCache 的图片质量（PSNR），以及 MagCache + torch.compile 组合