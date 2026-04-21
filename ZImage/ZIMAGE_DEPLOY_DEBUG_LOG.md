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
- [ ] Pipeline 构建成功
- [ ] Cosmos 上传模型数据
- [ ] DLIS 部署测试
