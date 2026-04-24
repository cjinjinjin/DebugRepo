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
