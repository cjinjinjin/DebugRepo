# DLIS 模型部署指南（基于 OaaS_LLMTemplate）

> 版本 v5 — 2026-04-22 更新，整合 Gemma4、ZImage 部署实战经验及 ChangXu v2 文档内容

---

## 目录

1. [简介](#1-简介)
2. [部署流程概览](#2-部署流程概览)
3. [Step 1：本地开发与测试](#3-step-1本地开发与测试)
4. [Step 2：上传 Checkpoint 到 Gen1](#4-step-2上传-checkpoint-到-gen1)
5. [Step 3：Gen1 → Gen2 数据迁移](#5-step-3gen1--gen2-数据迁移)
6. [Step 4：OaaS 模板定制与 Docker 镜像构建](#6-step-4oaas-模板定制与-docker-镜像构建)
7. [Step 5：DLIS 部署与验证](#7-step-5dlis-部署与验证)
8. [Step 6：Polaris Job 优化（可选）](#8-step-6polaris-job-优化可选)
9. [Base Docker 镜像选择与更新](#9-base-docker-镜像选择与更新)
10. [清华镜像源配置](#10-清华镜像源配置)
11. [Model 代码编写指南](#11-model-代码编写指南)
12. [外部 settings.json 配置覆盖（证书/环境切换）](#12-外部-settingsjson-配置覆盖证书环境切换)
13. [常见问题与解决方案](#13-常见问题与解决方案)
14. [附录](#14-附录)

---

## 1. 简介

### 1.1 本文档用途

本文档为部署 LLM / 多模态 / Diffusion 等模型到 DLIS（Deep Learning Inference Service）平台提供完整指导，适用于需要：

- 将优化后的模型部署到生产环境
- 利用 DLIS 基础设施进行模型推理服务
- 使用 OaaS 模板实现标准化部署流程

### 1.2 什么是 OaaS？

OaaS（Optimization as a Service）是 DLIS 上部署 LLM 的综合框架，提供：

- **自动化模型优化**：量化、编译、优化的标准化工作流
- **多后端支持**：vLLM 和 TensorRT-LLM 推理引擎
- **生产就绪模板**：预配置模板，快速部署
- **批处理支持**：高吞吐量场景下的高效批量推理

> 📘 **官方文档**：[Optimizing a LLM model with DLIS LLM engine](https://dev.azure.com/msasg/ContentServices/_wiki/wikis/DLIS/...) — 包含模型优化技术、性能调优策略和故障排除指导

---

## 2. 部署流程概览

```
                                        ┌→ Gen1 → Gen2 迁移 ─┐
本地开发 & Docker 测试 → 上传 ckpt 到 Gen1 ─┤                      ├→ DLIS 部署
                                        └→ 构建镜像 ──────────┘
         ↑                               （可同步进行）           ↑
     先验证功能正确性                                      DLIS 从 Gen2 读取模型数据
```

**关键原则**：
- **先本地验证，再上线**。本地 Docker 测试通过后再上传数据和构建镜像
- **DLIS 读取的模型数据来自 Gen2**（`dlisstoregen2.dfs.core.windows.net`），不是 Gen1
- 镜像通过 OaaS_LLMTemplate 仓库的 CI pipeline 自动构建

---

## 3. Step 1：本地开发与测试

### 3.1 代码准备

在 `OaaS_LLMTemplate` 仓库创建个人分支（如 `jinjinchen/ZImage-v1`），修改以下文件：

| 文件 | 说明 |
|------|------|
| `dlis_model/model/model.py` | 模型初始化 + 推理逻辑（**核心文件**） |
| `dlis_model/model/dlis_inter.py` | 预处理/后处理，实现 `PreAndPostProcessor` 类 |
| `dlis_model/http_server.py` | HTTP 服务（如需支持 multipart/form-data 等自定义格式） |
| `requirements-vllm.txt` | Python 依赖 |

**`dlis_inter.py` 接口说明**：

```python
class PreAndPostProcessor:
    def preprocess(self, data):
        """将输入数据转换为模型兼容格式"""
        ...
    
    def postprocess(self, output, metadata):
        """将模型输出转换为期望的响应格式"""
        ...
```

> 详见 [Model 代码编写指南](#11-model-代码编写指南)

### 3.2 本地 Docker 测试

```bash
# 1. 构建镜像
cd /path/to/OaaS_LLMTemplate
export SOURCE_BRANCH="test"
sudo bash pipeline/build_vllm_image.sh

# 2. 启动容器
IMAGE_TAG="<build_tag>"   # 替换为实际 tag
sudo docker run -d --name model-test \
  --gpus all \
  -v /path/to/model_weights:/Model/model_name \
  -p <host_port>:8888 \
  <image_name>:$IMAGE_TAG \
  /dlis_model/run.sh http

# 3. 测试请求
curl -X POST http://localhost:<host_port> \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test input"}'
```

**注意事项**：
- 如果某个 GPU 被占用，用 `--gpus '"device=N"'` 指定可用 GPU
- 必须加 `-p <host_port>:8888` 端口映射，否则宿主机 curl 会 403
- `vllm/vllm-openai` 基础镜像的 entrypoint 是 `vllm serve`，如需交互式 bash 必须 `--entrypoint bash`
- 如果 volume mount 了 Python 文件，注意 `__pycache__` 可能导致旧代码被加载，需要 `rm -rf __pycache__` 或重启容器

### 3.3 离线测试（不启动 HTTP server）

```bash
sudo docker run --rm -it --gpus all \
  -v /path/to/model:/Model/model_name \
  <image>:<tag> \
  bash -c 'cd /dlis_model && ./run.sh offline /tmp/input.json /tmp/output.json'
```

---

## 4. Step 2：上传 Checkpoint 到 Gen1

模型权重文件上传到 Gen1 Cosmos 存储。

### 4.1 上传地址

```
上传目标：https://cosmos09.osdinfra.net:443/cosmos/DLISModelRepository/local/<your-alias>/
```

使用 **Visual Studio Scope Extension** 进行认证和上传。

### 4.2 目录结构（扁平化，推荐）

```
<model-dir>/
├── model_name/               ← 模型权重文件夹
│   ├── config.json
│   ├── model-00001-of-N.safetensors
│   ├── tokenizer.json
│   └── ...
├── dlis_inter.py             ← 直接放根目录（不要嵌套子文件夹！）
├── AggSvcAuthCert-prod.pfx   ← Kusto 证书（可选）
└── AggSvcAuthCert-si.pfx
```

**关键注意**：
- `dlis_inter.py` 必须放在 cosmos 根目录下，不要放在子文件夹中。代码中 `sys.path.append('/Model')` + `from dlis_inter import ...` 只能找到根目录的文件
- **不要在 cosmos 上放 `model.py`**，用镜像自带的版本。cosmos 上的旧 model.py 会覆盖镜像里的新版本
- 不要上传不需要的大文件（如 `.tar` 包），浪费同步时间

---

## 5. Step 3：Gen1 → Gen2 数据迁移

> **注意：Step 3（数据迁移）和 Step 4（镜像构建）可以同步进行，互不依赖。**

DLIS 部署时从 Gen2 读取模型数据，因此 Gen1 的数据需要迁移到 Gen2。

参考 Wiki：[Data Transfer Tools](https://dev.azure.com/msasg/ContentServices/_wiki/wikis/DLIS/...)
详细步骤参考：How_to_Build_Your_Own_DLIS_Model.docx（Step 6.2）

```
Gen2 路径格式：
abfs://dlisstore@dlisstoregen2.dfs.core.windows.net/dlismodelrepository-c09/local/users/<username>/<model-dir>/
```

Gen2 上的目录内容会被 DLIS 挂载到容器的 `/Model` 路径下（由 `DLIS_MODEL_DATA_TARGET_PATH=/Model` 控制）。

**DLIS 的 ModelDataPath 机制**：
- `ModelDataPath` 指向 Gen2 上的某个文件（如 `.rar` 或 `complete.txt`），DLIS 实际会把该文件所在的**整个父目录**挂载到 `/Model`
- `/Model` 以**只读方式挂载**（来自 Cosmos），无法在其中创建文件
- 如果需要在运行时创建配置文件（如 `_opt` 目录），必须用 writable mirror 方案（见常见问题 #3）

---

## 6. Step 4：OaaS 模板定制与 Docker 镜像构建

### 6.1 是否需要定制模板？

> **注意：模板定制是可选步骤。**

在定制之前，先检查 [OaaS LLM Template 仓库](https://dev.azure.com/msasg/ContentServices/_git/OaaS_LLMTemplate) 的原始模板是否已满足需求。对于纯文本 LLM 任务，现有模板通常已足够，可以跳过定制直接构建镜像。

常见需要定制的场景：

| 定制需求 | 说明 | 修改文件 |
|---------|------|---------|
| 多模态 vLLM 支持 | 原始模板的 vLLM 后端仅支持文本输入 | `dlis_model/model/model.py` |
| 图片传输格式 | 原始模板使用 Base64，改为 multipart/form-data 更高效 | `dlis_model/http_server.py` |
| 额外依赖包 | 如 diffusers、Pillow 等 | `requirements-vllm.txt` |
| 非 LLM 模型 | 如 Diffusion 模型（ZImage） | `dlis_model/model/model.py` |

### 6.2 分支方式构建镜像（推荐）

在 `OaaS_LLMTemplate` 仓库创建个人分支，push 代码后 CI pipeline 自动构建镜像。提交 PR 后会自动触发 Docker 镜像构建。

- 非 main 分支镜像 tag 格式：`YYYYMMDD-HHMM-<branch_name>`
- 镜像推送到：`dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:<tag>`
- 构建结果可在 Azure Pipelines 中查看

### 6.3 两种 Dockerfile 选择

| Dockerfile | 基础镜像 | 构建时间 | 适用场景 |
|------------|---------|---------|---------|
| `Dockerfile_vllm_0.10.0` | `nvidia/cuda:12.8.1-devel-ubuntu22.04` | **几十分钟**（编译 FlashInfer + DeepGEMM） | 需要特定 vLLM/torch 版本组合 |
| `Dockerfile_vllm_fast` | `vllm/vllm-openai:latest` | **< 1 秒** | 快速迭代，版本兼容即可 |

> 详见 [Base Docker 镜像选择与更新](#9-base-docker-镜像选择与更新)

### 6.4 本地手动构建测试

```bash
cd OaaS_LLMTemplate
IMAGE_TAG="local-test"

# Block 1: 构建基础镜像
sudo docker build -t my-vllm-base:$IMAGE_TAG \
    --file pipeline/Dockerfile_vllm_fast pipeline/

# Block 2: 安装 OaaS 代码和依赖
sudo docker run -d --name temp my-vllm-base:$IMAGE_TAG sleep infinity
sudo docker cp . temp:/
sudo docker exec temp chmod +x /dlis_model/run.sh /dlis_model/async_run.sh /LLMModelOptimization.sh
sudo docker exec temp python3 -m pip install -r /requirements-common.txt
sudo docker exec temp python3 -m pip install -r /requirements-vllm.txt
sudo docker exec temp python3 -m pip install -e /
sudo docker commit temp my-final:$IMAGE_TAG
sudo docker rm -f temp
```

---

## 7. Step 5：DLIS 部署与验证

### 7.1 Polaris Job 配置

| 字段 | 示例值 | 说明 |
|------|--------|------|
| ModelPath | `docker-repo://dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:<tag>` | 镜像地址 |
| ModelDataPath | `abfs://dlisstore@dlisstoregen2.dfs.core.windows.net/.../complete.txt` | Gen2 路径，指向目录内任一文件 |
| 环境变量 | `DLIS_MODEL_DATA_TARGET_PATH=/Model;GPU_MEMORY_UTILIZATION=0.7` | |
| WaitingModelReadyInMin | 30 | 模型加载超时时间 |

### 7.2 构建 DLIS Service

参考：How_to_Build_Your_Own_DLIS_Model.docx（Step 8）创建 DLIS Service。

### 7.3 测试请求

**注意 URL 格式**：
- 正确：`https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.<ModelName>`
- 错误：`http://WestUS2BE.bing.prod.dlis.binginternal.com:86/route/...`（HTTP + 端口 86 会超时）
- 不需要 `:8888` 后缀，不需要 `/routebatch/`

```python
import requests

response = requests.post(
    "https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.<ModelName>",
    cert=("private1.cer", "private1.key"),
    json={"prompt": "test input"},
    headers={"Content-Type": "application/json"},
    verify=False,
)
```

需要客户端证书（`.cer` + `.key` 文件）进行认证。

---

## 8. Step 6：Polaris Job 优化（可选）

> **注意：此步骤为可选。** 如果跳过此步骤，需要联系 Fang Zhang 提交 bypass job。

### 8.1 量化

**A. 自动优化**：通过 Polaris Job 内置的量化流程自动执行。

**B. 离线量化（推荐）**：部署前使用以下工具进行量化：
- **AutoGPTQ** — 适合 GPTQ 格式量化
- **AutoAWQ** — 适合 AWQ 格式量化
- **llm-compressor** — 通用压缩工具

> 离线量化后上传量化模型到 Cosmos，vLLM 会从模型的 `config.json` 自动检测量化格式，无需额外配置。

### 8.2 其他优化

| 优化方式 | 说明 |
|---------|------|
| **Async API Call** | 异步推理调用，提升吞吐量 |
| **Continuous Batching** | 连续批处理，适用于兼容模型，显著提升吞吐量 |

---

## 9. Base Docker 镜像选择与更新

### 9.1 快速构建方案（推荐）：`Dockerfile_vllm_fast`

```dockerfile
FROM vllm/vllm-openai:latest
# 已包含 vllm、torch、transformers，无需编译
# 如需特定 transformers 版本（如支持 Gemma4）：
RUN python3 -m pip install transformers==5.5.3
```

**优点**：构建 < 1 秒，包含预编译的 vllm + torch
**缺点**：`latest` 版本不可控，可能被上游更新

### 9.2 完整构建方案：`Dockerfile_vllm_0.10.0`

需要手动管理 torch/vllm/torchvision 版本兼容性：

```dockerfile
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# 关键：先装正确版本的 torch，再装 vllm
# 否则 vllm 的依赖可能降级 torch 导致 ABI 不匹配
RUN pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
RUN pip install vllm==0.19.0
```

### 9.3 版本兼容性经验

| 问题 | 根因 | 解决 |
|------|------|------|
| `vllm/_C.abi3.so: undefined symbol` | torch ABI 不匹配（torchvision 降级了 torch） | Dockerfile 中先装 torch，`requirements-vllm.txt` 不要加 torch/torchvision |
| `torchvision::nms operator does not exist` | torchvision 和 torch 版本不匹配 | 确保 torchvision 版本与 torch 对应（如 torch 2.10.0 + torchvision 0.25.0） |
| `num_scheduler_steps` 不被接受 | vllm 0.19.0 移除了该参数 | 从 `vllm_runner.py` 中删除相关代码 |
| `huggingface-hub>=0.34.0,<1.0 is required` | pin 了 huggingface_hub==1.6.0 | 移除 pin，使用基础镜像自带版本 |
| `Gemma4VideoProcessor requires Torchvision` | 卸载 torchvision 后 vllm 多模态处理器报错 | 装回正确版本的 torchvision |

**原则**：`requirements-vllm.txt` 不要加 torch、torchvision、transformers，这些由 Dockerfile 统一管理。

---

## 10. 清华镜像源配置

DLIS CI pipeline agent 和 Docker build 可能无法直连 `pypi.org`，导致 `pip install` 超时。

### 10.1 CI Pipeline（`azure-pipelines-unified.yml`）

```yaml
variables:
  PIP_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple
  UV_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple
```

### 10.2 Docker Build（`build_vllm_image.sh`）

```bash
# 构建阶段：通过 --build-arg 传入
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
  --build-arg UV_EXTRA_INDEX_URL=https://pypi.org/simple \
  ...

# docker exec 安装阶段：通过 PIP_ARGS 变量
PIP_ARGS="-i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple"
docker exec container pip install $PIP_ARGS -r requirements.txt
```

### 10.3 Dockerfile 接收参数

```dockerfile
ARG PIP_INDEX_URL
ARG UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG UV_EXTRA_INDEX_URL
```

**注意**：不能用 `docker exec -e` 环境变量方式传清华源参数，必须用 `--build-arg`（构建阶段）或命令行参数（exec 阶段）。

---

## 11. Model 代码编写指南

### 11.1 `model.py` 核心结构

DLIS 框架要求 `ModelImp` 类实现以下接口：

```python
class ModelImp:
    def __init__(self):
        # 模型初始化：加载引擎、配置参数
        pass
    
    def Eval(self, data):
        # 单条推理
        pass
    
    def EvalBatch(self, data_list):
        # 批量推理
        pass
```

### 11.2 方案 A：使用 OaasWrapper（适合简单 LLM 推理）

```python
from llm_opt.oaas_wrapper_v2 import OaasWrapper

class ModelImp:
    def __init__(self):
        self.oaas_wrapper = OaasWrapper("model_dir_name", is_llm_model=True)
    
    def Eval(self, data):
        prompts = preprocess(data)
        outputs = self.oaas_wrapper.run(prompts)
        return postprocess(outputs)
```

**需要的配置文件**（放在模型目录的 `<model_name>_opt/` 子目录下）：

- `opt_type.txt`：内容为 `llm`（用 `printf "llm"` 写入，不要用 `echo`，避免末尾换行）
- `best_setting.json`：
  ```json
  {
      "llm_type": "vllm",
      "model": "model-dir-name",
      "max_output_len": 256,
      "temperature": 0.8,
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.9,
      "trust_remote_code": true,
      "dtype": "auto",
      "max_model_len": 8192,
      "stop": ["</end_token>"]
  }
  ```

**注意**：`best_setting.json` 中**不要设 `quantization` 字段**。`QUANTIZATION_MAP` 没有 `"awq"` 映射，设了会导致模型路径拼接错误。vLLM 会从模型的 `config.json` 自动检测量化格式。

### 11.3 方案 B：直接使用 vLLM（推荐，更可控）

```python
from vllm import LLM, SamplingParams

class ModelImp:
    def __init__(self):
        # CUDA_VISIBLE_DEVICES UUID 修复
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_env and not cuda_env.replace(",", "").isdigit():
            gpu_count = len(cuda_env.split(","))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
        
        self.llm = LLM(
            model="/Model/model-name",
            tensor_parallel_size=1,
            gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")),
            trust_remote_code=True,
            dtype="auto",
            max_model_len=8192,
            enable_prefix_caching=True,   # batch 推理时复用 KV cache
        )
        
        self.sampling_params = SamplingParams(
            temperature=1.0, top_p=0.95, top_k=64,
            max_tokens=128, stop=["</end_token>"],
        )
    
    def Eval(self, data):
        prompts = preprocess(data)
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = [o.outputs[0].text for o in outputs]
        return postprocess(results)
```

**方案 B 的优势**：
- 去掉 OaasWrapper 中间层，初始化失败直接报错，不会静默 fallback 到 CPU 加载
- 不需要 `_opt` 目录和 `best_setting.json`
- 推理参数直接写在代码中，透明可控

### 11.4 非 LLM 模型（如 ZImage diffusion）

```python
class ModelImp:
    def __init__(self):
        from diffusers import ZImagePipeline
        model_path = os.environ.get("ZIMAGE_MODEL_PATH", "/Model/model_name")
        self.pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.pipe.to("cuda:0")
    
    def Eval(self, data):
        result = self.pipe(prompt=data["prompt"], width=data["width"], height=data["height"])
        # 返回 base64 编码图片
        ...
```

### 11.5 多步推理（如 Gemma4 Two-Step）

如果需要多次调用推理引擎：

```python
def Eval(self, data):
    # Step 1：生成 scene concepts
    step1_prompts = self.processor.preprocess(data)
    step1_outputs = self._run_vllm(step1_prompts, self.step1_params)
    
    # Step 2：展开为详细 prompts
    step2_prompts = self.processor.build_step2_prompts(step1_outputs)
    step2_outputs = self._run_vllm(step2_prompts, self.step2_params)
    
    return self.processor.postprocess(step2_outputs)
```

**关键**：Step 1 和 Step 2 应使用**独立的 SamplingParams**（不同的 max_tokens 和 stop tokens），共用会导致输出异常。

### 11.6 Tokenizer Thinking Mode 问题

某些量化模型（如 Gemma4 AWQ）的 tokenizer `chat_template` 内置了 thinking prefix（`<|channel>thought\n<channel|>`），导致模型进入 thinking mode。

**修复**：在 `__init__` 中 patch tokenizer：

```python
tokenizer = self.llm.get_tokenizer()
if hasattr(tokenizer, 'chat_template') and '<|channel>thought' in (tokenizer.chat_template or ''):
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "<|channel>thought\n<channel|>", ""
    )
```

---

## 12. 外部 settings.json 配置覆盖（证书/环境切换）

### 12.1 背景

`config.py` 中 `eventhub_namespace`、`certificate_path` 等配置如果硬编码在代码里，切换 si/prod 环境需要改代码重新构建镜像。参考 Hanbang（`user/hanbangliang/img-outpainting-v1` 分支）的实现，改为通过 Cosmos 上的外部 JSON 文件覆盖配置。

### 12.2 实现方式

在 `config.py` 中使用 `pydantic-settings` + 自定义 `JsonFileSettingsSource`：

```python
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_SETTINGS_JSON_PATH = os.environ.get("SETTINGS_JSON_PATH", "/Model/settings.json")

class JsonFileSettingsSource(PydanticBaseSettingsSource):
    """从 JSON 文件加载配置覆盖，文件不存在时静默忽略"""
    def __init__(self, settings_cls, json_file_path: str):
        super().__init__(settings_cls)
        self.json_file_path = json_file_path

    def get_field_value(self, field, field_name):
        # 从 JSON 文件读取对应字段值
        ...

class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")  # 防止 JSON 中多余字段导致 ValidationError
    eventhub_namespace: str = "aggregation-logging.servicebus.windows.net"
    certificate_path: str = os.path.join("/Model", "AggSvcAuthCert-prod.pfx")
    ...

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        json_source = JsonFileSettingsSource(settings_cls, DEFAULT_SETTINGS_JSON_PATH)
        return (init_settings, env_settings, dotenv_settings, json_source, file_secret_settings)
```

### 12.3 配置优先级

1. **init kwargs**（代码显式传参）
2. **环境变量**（如 `EVENTHUB_NAMESPACE=...`）
3. **JSON 文件**（`/Model/settings.json`）
4. **代码默认值**

### 12.4 使用方式

在 Cosmos 根目录放置 `settings.json`，只写需要覆盖的字段：

```json
{
    "eventhub_namespace": "aggregation-si-logging.servicebus.windows.net",
    "certificate_path": "/Model/AggSvcAuthCert-si.pfx"
}
```

不放 `settings.json` 则使用代码默认值（prod 环境）。

### 12.5 Cosmos 目录结构

证书文件放 cosmos 根目录，不要嵌套在模型子目录里：

```
cosmos root → /Model/
├── model_name/               ← 模型权重
├── dlis_inter.py
├── settings.json             ← 配置覆盖（可选）
├── AggSvcAuthCert-prod.pfx   ← prod 证书
└── AggSvcAuthCert-si.pfx     ← si 证书
```

### 12.6 EventHub Credential 容错

证书加载失败不应阻塞容器启动，需在 `eventhub_sink.py` 中做容错处理：

```python
def _try_get_credential(tenant_id: str):
    try:
        return CertificateCredential(...)
    except Exception as e:
        logger.info("EventHub credential unavailable (%s), kusto logs will be local-only", e)
        return None

_credential = _try_get_credential(settings.corp_tenant_id)

class EventHubSink:
    def __init__(self, ...):
        self.producer = None
        if _credential is not None:
            self.producer = EventHubProducerClient(...)

    def send_messages_in_background(self, msgs):
        if self.producer is None:
            return  # 静默跳过，不阻塞推理
        ...
```

### 12.7 优点

- 切换 si/prod 环境不需要改代码重新构建镜像，只需修改 cosmos 上的 `settings.json`
- 证书文件放 cosmos 根目录 `/Model/`，不依赖模型子目录名
- credential 创建失败不阻塞容器启动，便于在无证书环境下调试
- 与 Hanbang 的 img-outpainting 分支保持架构一致

### 12.8 本地测试 Kusto Log 输出

部署完成后，可以通过本地发送带 `tracking_data` 的请求来验证 Kusto 日志是否正常写入。

#### 测试客户端

使用 AAD Bearer token（MSAL + PFX 证书）认证的测试脚本 `test_pfx.py`：

```python
"""
Client for PicassoAdsCreative.ZImage-V1-Jinjin DLIS service.

Request format: JSON
  - prompt: str (text description for image generation)
  - width: int (image width, default 1344)
  - height: int (image height, default 768)

Response format:
  {"image": "<base64 PNG>", "width": 1344, "height": 768, "seed": 42}

Auth: AAD Bearer token via MSAL + PFX certificate
"""

import argparse, base64, json, os, time, uuid
import msal, requests, urllib3
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates
from cryptography.hazmat.primitives.hashes import SHA1

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = "https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.<ModelName>"
TENANT_ID = os.environ.get("TENANT_ID", "<your-tenant-id>")
CLIENT_ID = os.environ.get("CLIENT_ID", "<your-client-id>")
CERT_PATH = os.environ.get("CERT_PATH", "/path/to/AggSvcAuthCert-si.pfx")
DLIS_SCOPE = os.environ.get("DLIS_SCOPE", "<your-scope>/.default")

def _load_pfx(pfx_path, password=None):
    with open(pfx_path, "rb") as f:
        pfx_data = f.read()
    private_key, certificate, additional_certs = load_key_and_certificates(pfx_data, password, default_backend())
    key_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    cert_pem = certificate.public_bytes(Encoding.PEM)
    chain = [cert_pem] + ([c.public_bytes(Encoding.PEM) for c in additional_certs] if additional_certs else [])
    return {
        "private_key": key_pem,
        "thumbprint": certificate.fingerprint(SHA1()).hex().upper(),
        "public_certificate": b"".join(chain).decode(),
    }

def _get_bearer_token():
    cert = _load_pfx(CERT_PATH)
    app = msal.ConfidentialClientApplication(
        client_id=CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{TENANT_ID}",
        client_credential={
            "private_key": cert["private_key"].decode(),
            "thumbprint": cert["thumbprint"],
            "public_certificate": cert["public_certificate"],
        },
    )
    result = app.acquire_token_for_client(scopes=[DLIS_SCOPE])
    if "access_token" not in result:
        raise RuntimeError(f"Failed to get token: {result.get('error_description', result)}")
    return f"Bearer {result['access_token']}"

def call_model(prompt, width=1344, height=768):
    payload = {
        "prompt": prompt, "width": width, "height": height,
        "tracking_data": {
            "requestid": f"test-{uuid.uuid4().hex[:8]}",
            "trackingid": f"test-{uuid.uuid4().hex[:8]}",
            "callername": "local_test",
        },
    }
    resp = requests.post(API_URL, json=payload,
                         headers={"Authorization": _get_bearer_token(), "Content-Type": "application/json"},
                         verify=False, timeout=120)
    print(f"Status: {resp.status_code}")
    return resp.json() if resp.status_code == 200 else {}
```

#### 发送测试请求

```bash
python3 test_pfx.py --prompt "A cat and a dog on a cloud"
```

#### 查看 Kusto 日志

在 [Azure Data Explorer](https://dataexplorer.azure.com/clusters/bingads/databases/BingAdsTracing) 上查询：

```kql
appsvc_info | union appsvc_warn | union appsvc_err
| where Timestamp > ago(30min)
| where ApplicationName == 'ImgLPRelevanceModel'
```

**注意**：`ApplicationName` 的值来自代码 `config.py` 中 `application_name` 字段的配置。

---

## 13. 常见问题与解决方案

### 问题 1：容器 OOM Killed（CPU 内存，非 GPU）

**现象**：容器反复 crash，GPU 显存仅 9MB，CPU 内存接近 20GB 限制。

**根因**：OaasWrapper 找不到 `_opt` 目录 → fallback 到 BaseLLM → `AutoModelForCausalLM.from_pretrained().to("cuda")` 先在 CPU 加载全部权重 → 超过 CPU 内存限制。

**解决**：
- 确保 `_opt` 目录结构正确（方案 A）
- 或直接使用 `vllm.LLM()` 初始化（方案 B，推荐）

### 问题 2：CUDA_VISIBLE_DEVICES UUID 格式

**现象**：`ValueError: invalid literal for int() with base 10: 'GPU-405664ab-...'`

**根因**：DLIS 设置 `CUDA_VISIBLE_DEVICES` 为 GPU UUID 格式，vLLM 内部 `int()` 转换失败。

**解决**（两种方式，都需要）：
1. `run.sh` 中确保 `unset CUDA_VISIBLE_DEVICES`（不要注释掉！）
2. `model.py` 中做 UUID → 整数索引转换（见 9.3 示例）

### 问题 3：/Model 是只读文件系统

**现象**：`OSError: [Errno 30] Read-only file system: '/Model/...'`

**根因**：DLIS 将 Cosmos 数据以只读方式挂载到 `/Model`。

**解决**：Writable Mirror 方案 — 在 `/tmp` 创建镜像目录，用 symlink 指向只读模型文件：

```python
writable_model = os.path.join("/tmp", model_dir_name)
os.makedirs(writable_model, exist_ok=True)
for item in os.listdir(src_model):
    os.symlink(os.path.join(os.path.realpath(src_model), item),
               os.path.join(writable_model, item))
# 在 writable_model 中创建 _opt 目录...
self.oaas_wrapper = OaasWrapper(writable_model, is_llm_model=True)
```

### 问题 4：Unable to find exposed port 8888

**现象**：DLIS 日志 `Unable to find exposed port 8888 for container ...`

**根因**：通常不是端口配置问题，而是模型加载阶段 crash（OOM 等），HTTP server 从未启动。

**解决**：先解决模型加载问题（参见问题 1、2）。同时确保 Dockerfile 有 `EXPOSE 8888`。

### 问题 5：Pipeline 构建 pip install 超时

**现象**：`ERROR: Could not find a version that satisfies the requirement tornado`

**根因**：CI agent 无法直连 `pypi.org`。

**解决**：添加清华 PyPI 镜像（见 [清华镜像源配置](#8-清华镜像源配置)）。

### 问题 6：OaasWrapper `_create_runner()` 静默吞异常

**现象**：`RuntimeError: Could not get any available runner`，但看不到真正的错误。

**根因**：`_create_runner()` 有 `try/except` 捕获所有异常并 `print()`（不进 DLIS 日志），然后 `return None`。

**解决**：
- 修改 `oaas_wrapper_v2.py`，把 `print` 改为 `logger.error(exc_info=True)`
- 或直接使用方案 B（直接 vLLM），绕过 OaasWrapper

### 问题 7：`opt_type.txt` 换行符

**现象**：`Unknown opt type llm\n` 或 `Unknown opt type vllm`

**根因**：`echo "llm"` 写入时带了末尾换行符。

**解决**：用 `printf "llm"` 代替 `echo "llm"`。

### 问题 8：best_setting.json 字段缺失或错误

**常见错误**：
- `Failed to create runner: 'llm_type'` → 缺少 `llm_type` 字段
- 模型路径错误 → 不要设 `"quantization": "awq"`，让 vLLM 自动检测

### 问题 9：Cosmos 数据同步

**现象**：手动上传到 Cosmos 的文件（如 `_opt` 子目录）在容器内不可见。

**根因**：DLIS 挂载 Cosmos 目录时可能使用缓存快照，而非实时拉取。等待多小时也不会刷新。

**解决**：不要依赖运行时在 `/Model` 下创建文件。用代码在 `/tmp` 下创建配置（writable mirror），或直接用方案 B 绕过。

### 问题 10：DLIS 请求 URL 格式

| 错误格式 | 正确格式 |
|---------|---------|
| `http://WestUS2BE.bing.prod.dlis.binginternal.com:86/route/...` | `https://WestUS2.bing.prod.dlis.binginternal.com/route/...` |
| 带 `:8888` 后缀 | 不需要端口后缀 |
| `/routebatch/` | `/route/` |

---

## 14. 附录

### 附录 A：Central Log 调试工具

使用 Central Log 调试 Polaris Job，查看容器输出日志：

> 文档：[How to Use Central Log](https://dev.azure.com/msasg/ContentServices/_wiki/wikis/DLIS/...)

**示例查询**：

```sql
SELECT machine_name, log_level, log_time, description
FROM dlissensitivelog
WHERE file_name LIKE 'DLMSUserLog_ContainerOutput%.log'
  AND log_time BETWEEN TIMESTAMP '2026-02-27 00:00:00'
                   AND TIMESTAMP '2026-02-28 18:00:00'
  AND machine_name = '<your_machine_name>'
LIMIT 10000;
```

### 附录 B：参考链接

| 资源 | 链接 |
|------|------|
| DLIS LLM Engine 官方文档 | [Optimizing a LLM model with DLIS LLM engine](https://dev.azure.com/msasg/ContentServices/_wiki/wikis/DLIS/...) |
| OaaS Template 仓库 | [OaaS_LLMTemplate](https://dev.azure.com/msasg/ContentServices/_git/OaaS_LLMTemplate) |
| Gen1→Gen2 数据迁移工具 | [Data Transfer Tools](https://dev.azure.com/msasg/ContentServices/_wiki/wikis/DLIS/...) |
| Central Log 查询指南 | [How to Use Central Log](https://dev.azure.com/msasg/ContentServices/_wiki/wikis/DLIS/...) |
| DLIS 模型构建指南 | How_to_Build_Your_Own_DLIS_Model.docx |
| Kusto 日志查询 | [Azure Data Explorer - BingAdsTracing](https://dataexplorer.azure.com/clusters/bingads/databases/BingAdsTracing) |

### 附录 C：Gemma4 DLIS 部署经历总结（8 次迭代）

| 部署 # | 问题 | 修复 |
|--------|------|------|
| #1-#3 | `_opt` 目录不可见 → BaseLLM CPU 加载 → OOM | 确认 Cosmos 同步无法解决 |
| #4 | `/Model` 只读无法创建 `_opt` | Writable mirror 方案 |
| #5 | vLLM 初始化失败被静默吞掉 | `_create_runner()` 改为抛异常 |
| #6 | 改为 raise 后变成 crash loop | 改回 return None + logger.error |
| #7 | CUDA_VISIBLE_DEVICES UUID 格式 | UUID → 整数索引转换 |
| #8 | Cosmos 目录结构不对 + 旧 model.py 残留 | 新建扁平 cosmos 目录 + 去掉 OaasWrapper |
| 最终 | **直接 vLLM 方案成功** | 方案 B：`from vllm import LLM` |

**核心教训**：OaasWrapper 中间层带来的复杂度远大于收益。对于新模型部署，推荐直接使用 `vllm.LLM()`（方案 B），保留 `ModelImp` 接口契约即可。

---

## 修订历史

| 版本 | 日期 | 作者 | 说明 |
|------|------|------|------|
| v2 | 2026-02-26 | ChangXu | 初始文档，完整部署指南 |
| v4 | 2026-04-22 | Jinjin Chen | 整合 Gemma4/ZImage 实战经验，新增模板定制、常见问题、代码编写指南 |
| v5 | 2026-04-22 | Jinjin Chen | 整合 ChangXu v2 内容（OaaS 介绍、Polaris 优化、Central Log、参考链接），新增 settings.json 证书配置、Kusto 日志测试 |
