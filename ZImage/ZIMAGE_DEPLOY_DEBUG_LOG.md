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

import requests

response = requests.post(
    "https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.ZImage-v2",
    cert=("private1.cer", "private1.key"),
    json={"prompt": "A beautiful sunset over the ocean", "width": 1344, "height": 768},
    headers={"Content-Type": "application/json"},
    verify=False,
)