DLIS Model Deployment Guide (OaaS_LLMTemplate)

Version v5.5 — 2026-04-28 Consolidating Gemma4, ZImage deployment experience, ChangXu v2 doc, Hao’s docx, and team insights from Teams chats

# Table of Contents

Part 1: Deployment Main Flow

- 1. Introduction
- 2. Deployment Flow Overview
- 3. Step 1: Local Development & Testing
- 4. Step 2: Upload Checkpoint to Gen1
- 5. Step 3: Gen1 to Gen2 Data Migration
- 6. Step 4: PR Submission & CI Auto Image Build
- 7. Step 5: Polaris Testing
- 8. Step 6: DLIS Production Deployment
- 9. Step 7: Post-Deployment Verification
- 10. Step 8: Polaris Job Optimization (Optional)
Part 2: Environment & Operations

- 11. SI/Prod Environments & Certificate Management
- 12. Kusto Log Viewing & Debugging
- 13. Common Issues & Solutions
Appendices

- A. Base Docker Image Selection & Updates
- B. Tsinghua Mirror Configuration
- C. Model Code Writing Guide
- D. External settings.json Configuration Override
- E. Central Log Debugging Tool
- F. Reference Links
- G. Gemma4 Deployment Iteration Summary
- H. Revision History
Part 1: Deployment Main Flow

# 1. Introduction

## 1.1 Purpose of This Document

This document provides a complete guide for deploying LLM / multimodal / Diffusion models to the DLIS (Deep Learning Inference Service) platform.

## 1.2 What is OaaS?

OaaS (Optimization as a Service) is a comprehensive framework for deploying LLMs on DLIS, providing:

- Automated model optimization: standardized workflows for quantization, compilation, and optimization
- Multi-backend support: vLLM and TensorRT-LLM inference engines
- Production-ready templates: pre-configured templates for rapid deployment
- Batch processing support: efficient batch inference for high-throughput scenarios
Official docs: Optimizing a LLM model with DLIS LLM engine (DLIS Wiki)

# 2. Deployment Flow Overview

┌───────────────────────────────────┐ │ Step 1: Local Dev & Docker Test │ ← Verify correctness first └────────────────┬──────────────────┘ │ ┌────────────┴────────────┐ ▼ ▼ (can run in parallel) ┌─────────────────────┐ ┌─────────────┐ │Step 2: Upload ckpt │ │Step 4: │ │ to Gen1 │ │PR & CI │ │ ↓ │ │Build Image │ │Step 3: Gen1→Gen2 │ │ │ │ Migration │ │ │ └──────────┬──────────┘ └──────┬──────┘ └────────────┬───────┘ ▼ ┌───────────────────────────────────┐ │ Step 5: Polaris Test │ ← Check output & latency; DLIS reads model from Gen2 └────────────────┬──────────────────┘ ▼ ┌───────────────────────────────────┐ │ Step 6: DLIS Production Deploy │ └────────────────┬──────────────────┘ ▼ ┌───────────────────────────────────┐ │ Step 7: Verification & Monitoring │ └───────────────────────────────────┘

Key Principles:

- Test locally first. Only upload data and build images after local Docker tests pass
- Integrate Kusto logging from the local testing phase. Do not wait until online deployment — configure certificates and EventHub during local Docker testing so you can view logs in Kusto in real-time and catch log format, auth, and connection issues early
- DLIS reads model data from Gen2 (dlisstoregen2.dfs.core.windows.net), not Gen1
- Images are automatically built via the OaaS_LLMTemplate repo CI pipeline
- Step 2+3 (upload ckpt + data migration) and Step 4 (image build) can run in parallel, independently
# 3. Step 1: Local Development & Testing

## 3.1 Development Flow Overview

Create a personal branch in the OaaS_LLMTemplate repo (e.g., jinjinchen/ZImage-v1). Complete the following before pushing code to trigger CI build:

- 1. Modify model code (model.py, dlis_inter.py, etc.)
- 2. Evaluate whether OaaS template customization is needed (multimodal support, custom formats, etc.)
- 3. Choose the appropriate Dockerfile (fast iteration vs full build)
- 4. Build and test locally with Docker, confirm correct functionality
## 3.2 Code File Reference

In the OaaS_LLMTemplate repo, create a personal branch (e.g., jinjinchen/ZImage-v1) and modify the following files:

| File | Description |
| --- | --- |
| dlis_model/model/model.py | Model initialization + inference logic (core file) |
| dlis_model/model/dlis_inter.py | Pre/post-processing, implements PreAndPostProcessor class |
| dlis_model/http_server.py | HTTP server (if custom format needed) |
| requirements-vllm.txt | Python dependencies |

-> See "Appendix C: Model Code Writing Guide" for model.py details

## 3.3 OaaS Template Customization (Optional)

Template customization is optional. First check whether the original OaaS LLM Template repo template meets your needs. For pure text LLM inference, customization is usually unnecessary.

| Customization Need | Description | File to Modify |
| --- | --- | --- |
| Multimodal vLLM support | Original template only supports text input | model.py |
| Image transfer format | Use multipart/form-data for efficiency | http_server.py |
| Additional dependencies | e.g., diffusers, Pillow, etc. | requirements-vllm.txt |
| Non-LLM models | e.g., Diffusion models (ZImage) | model.py |

## 3.4 Two Dockerfile Options

| Dockerfile | Base Image | Build Time | Use Case |
| --- | --- | --- | --- |
| Dockerfile_vllm_0.10.0 | nvidia/cuda:12.8.1-devel-ubuntu22.04 | Tens of minutes | Need specific vLLM/torch version |
| Dockerfile_vllm_fast | vllm/vllm-openai:latest | < 1 second | Fast iteration |

-> See "Appendix A: Base Docker Image Selection & Updates" for version compatibility details

Common Docker Build Network Issues:

- apt-get install fails: CI build agent cannot connect to archive.ubuntu.com
- pip install timeout: pypi.org unreachable
-> See "Appendix B: Tsinghua Mirror Configuration" for solutions

## 3.5 Local Docker Testing

# 1. Build image cd /path/to/OaaS_LLMTemplate export SOURCE_BRANCH="test" sudo bash pipeline/build_vllm_image.sh # 2. Start container IMAGE_TAG="<build_tag>" sudo docker run -d --name model-test \ --gpus all \ -v /path/to/model_weights:/Model/model_name \ -p <host_port>:8888 \ <image_name>:$IMAGE_TAG \ /dlis_model/run.sh http # 3. Test request curl -X POST http://localhost:<host_port> \ -H "Content-Type: application/json" \ -d '{"prompt": "test input"}'

Important Notes:

- If a GPU is occupied, use --gpus '"device=N"' to specify an available GPU
- Must add -p <host_port>:8888 port mapping, otherwise host curl will get 403
- vllm/vllm-openai base image entrypoint is vllm serve; use --entrypoint bash for interactive shell
- If Python files are volume-mounted, __pycache__ may cause stale code to be loaded
Verify Kusto Logging (recommended during local testing):

After starting the local Docker container, verify Kusto log delivery alongside inference testing. This catches certificate and EventHub configuration issues before deployment.

- 1. Ensure the correct environment PFX certificate (e.g., AggSvcAuthCert-si.pfx) is in your Cosmos directory and volume-mounted to /Model in the container
- 2. Ensure settings.json has the correct EventHub namespace and kusto_log parameters (cert path, topic, etc.)
- 3. After sending a test request, check container logs (docker logs) for EventHub send success/failure output
- 4. Query the corresponding Kusto environment table in Kusto Explorer to confirm logs arrived (typically 1-2 minute delay)
# SI environment Kusto query example // Kusto cluster: https://bingadsppe.kusto.windows.net/ // Database: appsvc appsvc_info | where TIMESTAMP > ago(10m) | where ModelName == "<your_model_name>" | order by TIMESTAMP desc | take 20

If no results: ① check cert environment matches namespace (SI cert + SI namespace); ② verify logger level is set to INFO; ③ check if EventHub send errors are silently swallowed. See Section 12 for details.

## 3.6 Offline Testing (No HTTP Server)

sudo docker run --rm -it --gpus all \ -v /path/to/model:/Model/model_name \ <image>:<tag> \ bash -c 'cd /dlis_model && ./run.sh offline /tmp/input.json /tmp/output.json'

# 4. Step 2: Upload Checkpoint to Gen1

Upload model weight files to Gen1 Cosmos storage.

## 4.1 Upload Destination

Target: https://cosmos09.osdinfra.net:443/cosmos/DLISModelRepository/local/<your-alias>/

Use Visual Studio Scope Extension for authentication and upload.

Fig: Gen1 Cosmos upload directory example (Source: ChangXu doc)

[Attachment] Source doc: DLIS_Model_DeploymentWith_OaaS_v2.docx (ChangXu)

## 4.2 Directory Structure (Flat Layout, Recommended)

<model-dir>/ |-- model_name/ <- Model weight files folder | |-- config.json | |-- model-00001-of-N.safetensors | |-- tokenizer.json | +-- ... |-- dlis_inter.py <- Place in root directory |-- settings.json <- Config override (optional) |-- AggSvcAuthCert-prod.pfx <- Kusto certificate (optional) +-- AggSvcAuthCert-si.pfx

Critical Notes:

- dlis_inter.py must be in the Cosmos root directory; sys.path.append('/Model') only finds root-level files
- Do NOT put model.py in Cosmos; use the image-bundled version. Old model.py on Cosmos will override the new version in the image
- Do NOT upload unnecessary large files (e.g., .tar packages) — wastes sync time
- Place certificates in the Cosmos root directory, not in subdirectories (DLIS only mounts the top-level directory)
-> See "Appendix D: External settings.json Configuration Override" for settings.json details

# 5. Step 3: Gen1 to Gen2 Data Migration

Note: Step 2+3 (upload ckpt + data migration) and Step 4 (image build) can run in parallel, independently.

DLIS reads model data from Gen2 at deployment time, so Gen1 data must be migrated to Gen2.

Reference Wiki: Data Transfer Tools (DLIS Wiki)

Gen1 to Gen2 Migration Steps (see How_to_Build_Your_Own_DLIS_Model.docx Step 6.2):

- 1. Create a branch in Repos
- 2. Open DLIS copy pipeline, select View/Edit
Fig: Create branch and open pipeline (Source: Hao Zhang doc)

- 3. Select the newly created branch and update parameters
Fig: Select branch and set pipeline variables (Source: Hao Zhang doc)

Fig: Configure migration path parameters (Source: Hao Zhang doc)

- 4. Click Validate and Save to save parameters
Fig: Save and run pipeline (Source: Hao Zhang doc)

[Attachment] Source doc: How_to_Build_Your_Own_DLIS_Model.docx (Hao Zhang)

Gen2 path format: abfs://dlisstore@dlisstoregen2.dfs.core.windows.net/dlismodelrepository-c09/local/users/<username>/<model-dir>/

Gen2 Verification (Important):

- After migration, use SAW (Secure Admin Workstation) to verify file completeness on Gen2 (Desheng's advice)
- Files existing on Gen2 does not guarantee correctness — verify file sizes and integrity
- ADL data migration tools have pitfalls; carefully inspect after migration
ModelDataPath Mechanism:

- ModelDataPath points to a file on Gen2 (e.g., complete.txt); DLIS actually mounts the entire parent directory to /Model
- /Model is mounted read-only — cannot create files in it
- If runtime config files are needed, use the writable mirror approach (see Common Issues)
# 6. Step 4: PR Submission & CI Auto Image Build

Note: Step 2+3 (upload ckpt + data migration) and Step 4 (image build) can run in parallel, independently.

After local development and testing, push code to your personal branch in the OaaS_LLMTemplate repo. The CI pipeline will automatically build the Docker image.

## 6.1 Create Branch and Push Code

# Create a personal branch in OaaS_LLMTemplate repo git checkout -b <your-alias>/<model-name> # e.g., jinjinchen/ZImage-v1 # Commit local changes git add -A git commit -m "Add <model-name> model support" git push origin <your-alias>/<model-name>

## 6.2 CI Pipeline Auto Build

After pushing to the branch, the CI pipeline automatically triggers an image build. No manual action required.

| Item | Description |
| --- | --- |
| Trigger | Push to any branch triggers automatically (including non-main branches) |
| Image Tag Format | YYYYMMDD-HHMM-<branch_name> (non-main branches) |
| Image Registry | dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:<tag> |
| First Build Time | ~30 minutes |
| Incremental Build Time | ~8 minutes (Siwen's experience) |

Checking Build Status:

- View build progress and logs on the ADO Pipelines page
- After a successful build, the Pipeline log outputs the final image tag
- Use this image tag for the ModelPath configuration in subsequent Polaris Jobs
## 6.3 PR Build (Optional)

If you need to merge into the main branch (e.g., for general feature improvements), submitting a PR also triggers a build. After PR merge, the main branch builds an official version.

For project-specific model code, building on a personal branch is usually sufficient — no need to merge into main.

# 7. Step 5: Polaris Testing

After image build and data migration are complete, use a Polaris Job for testing to verify model output and latency meet expectations before DLIS production deployment.

## 7.1 Polaris Job Configuration

| Field | Example Value | Description |
| --- | --- | --- |
| ModelPath | docker-repo://dlisproddockerrepo.azurecr.io/dlis/llm_framework_vllm:<tag> | Image address |
| ModelDataPath | abfs://dlisstore@dlisstoregen2.dfs.core.windows.net/.../complete.txt | Gen2 path |
| Environment Variables | DLIS_MODEL_DATA_TARGET_PATH=/Model;GPU_MEMORY_UTILIZATION=0.7 |  |
| WaitingModelReadyInMin | 30 | Model load timeout |

## 7.2 Polaris Job Status

Fig: Polaris Job submission page (Source: Hao Zhang doc)

Fig: Polaris Job configuration parameters (1) (Source: Hao Zhang doc)

Fig: Polaris Job configuration parameters (2) (Source: Hao Zhang doc)

- Instance Loading: 100% + Success means deployment is successful (no need to wait for Instance Activate)
- Usually completes within 30 minutes; you can do other work after submission (Siwen's experience)
## 7.3 Testing Verification Points

During Polaris testing, verify the following:

- Output correctness: send test requests and check if model responses match expectations (format, content quality)
- Latency: record end-to-end response time, confirm it meets business SLA requirements
- Resource usage: observe GPU memory usage and CPU utilization are within reasonable bounds
- Stability: send multiple requests to confirm the service does not crash or return abnormal results
If testing reveals issues, go back to Step 1/4 to modify code or config, rebuild the image, and resubmit Polaris testing.

Fig: Polaris Job latency and QPS statistics after completion (Source: Hao Zhang doc)

## 7.4 Test Request Example

import requests response = requests.post( "https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.<ModelName>", cert=("private1.cer", "private1.key"), json={"prompt": "test input"}, headers={"Content-Type": "application/json"}, verify=False, ) print(f"Status: {response.status_code}") print(f"Latency: {response.elapsed.total_seconds():.2f}s") print(f"Response: {response.text[:500]}")

# 8. Step 6: DLIS Production Deployment

After Polaris testing passes, proceed with DLIS production deployment.

## 8.1 Hardware Allocation

Choose appropriate hardware based on model requirements (Image Model Service group experience):

| Model Type | Recommended Hardware | Notes |
| --- | --- | --- |
| Relevance Model | A100 / A100 Train | Large model inference requires high GPU memory |
| Diversity Model | T4 / MIG7 | Smaller models can use lower-spec GPUs |

- A100 machines may be resource-constrained; consider A100 Train as an alternative
- Check that instances have sufficient CPU resources (not just GPU)
- Open DLModelV2 - One Inference Portal
View available machines and quotas: DLIS Portal -> Quota V2, select Namespace and expand:

Fig: DLIS Portal Quota V2 page (Source: Hao Zhang doc)

Fig: Expand Namespace to view available machine list (Source: Hao Zhang doc)

Fig: Machine quota details (Source: Hao Zhang doc)

[Attachment] Source doc: How_to_Build_Your_Own_DLIS_Model.docx (Hao Zhang)

## 8.2 Create DLIS Service

Reference: How_to_Build_Your_Own_DLIS_Model.docx (Step 8) to create a DLIS Service.

Steps:

- 1. Open DLModelV2 - One Inference Portal, click New Model
Fig: One Inference Portal - click New Model (Source: Hao Zhang doc)

- 2. Paste Polaris Job Id, configure each page:
Fig: Configure Key, Hardware, General, ACL pages (Source: Hao Zhang doc)

- Key page: update Environment and Namespace
- Hardware page: select deployment target machine
- General page: fill in DRI contact, set min/max instance count, set Model Priority to Test (use Production for prod)
- ACL page: add access control ACL
## 8.3 ACL Configuration

The DLIS Service ACL string contains multiple certificate thumbprints and AAD application IDs, controlling who can call the service:

*:Certificate://Thumbprint/02AAAAA5AD...,*:AAD://appid/dda2a640-..., *:Certificate://Microsoft/dlis.si.advisoraggregator.trafficmanager.net,...

Incorrect ACL configuration will cause callers to receive 403 Forbidden. If requests are rejected after deployment, check ACL configuration first.

- 3. Click VALIDATION to validate, then click SUBMIT to submit
- Prod deployment requires a bypass process (manual operation by someone), which can be a bottleneck (Siwen's experience)
# 9. Step 7: Post-Deployment Verification

After successful DLIS production deployment, send requests to verify the service is working correctly.

## 9.1 Endpoint Naming Conventions

- Names should be descriptive and stable (e.g., PicassoAdsCreative.ZImage-V1)
- Avoid using personal names as endpoint names
- SI and Prod should not share the same endpoint (this is a pilot blocker)
## 9.2 Request URL Format

| Incorrect Format | Correct Format |
| --- | --- |
| http://WestUS2BE.bing.prod.dlis.binginternal.com:86/route/... | https://WestUS2.bing.prod.dlis.binginternal.com/route/... |
| With :8888 suffix | No port suffix needed |
| /routebatch/ | /route/ |

## 9.3 Test Request Example

import requests response = requests.post( "https://WestUS2.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.<ModelName>", cert=("private1.cer", "private1.key"), json={"prompt": "test input"}, headers={"Content-Type": "application/json"}, verify=False, )

Client certificates (.cer + .key files) are required for authentication; plain curl will not work.

-> See "Section 11: SI/Prod Environments & Certificate Management" for certificate details

-> See "Section 12: Kusto Log Viewing & Debugging" to verify Kusto logs are written correctly

# 10. Step 8: Polaris Job Optimization (Optional)

This step is optional. If skipped, contact Fang Zhang to submit a bypass job.

## 10.1 Quantization

- Automatic optimization: via Polaris Job built-in quantization workflow
- Offline quantization (recommended): use AutoGPTQ, AutoAWQ, or llm-compressor
- After offline quantization, upload the quantized model to Cosmos; vLLM auto-detects quantization format from config.json
## 10.2 Other Optimizations

| Optimization | Description |
| --- | --- |
| Async API Call | Asynchronous inference calls for improved throughput |
| Continuous Batching | Continuous batching for significantly improved throughput |

Part 2: Environment & Operations

# 11. SI/Prod Environments & Certificate Management

## 11.1 SI and Prod Environment Separation

- SI environment is for testing and validation; Prod environment is for production services
- Sharing the same endpoint between SI and Prod is a pilot blocker (confirmed by Image Model Service group); production launch requires separation
- Prod deployment requires a bypass process (manual operation), which can be a time bottleneck
## 11.2 Certificate Types and Usage

| Certificate Type | Format | Purpose | Location |
| --- | --- | --- | --- |
| Client authentication cert | .cer + .key | Call DLIS endpoint | Local machine or server |
| Kusto log certificate | .pfx | EventHub writes to Kusto logs | Cosmos /Model/ directory |
| SSL Keys | .cert + .key | Server-side SSL | Server-specific path |

Certificate file location reference (provided by Siwen):

- SSL keys: 10.224.120.197 /home/siwen/relevance/deploy
- New certificates on Cosmos: cosmos09.osdinfra.net/.../ImgLPRelevance6/
## 11.3 Certificate Expiration Management

- Certificates expire at the end of April 2026; new certificates use .pfx format
- Expired certificates will cause authentication failures; update in advance
- Recommend setting expiration reminders; update 2 weeks before expiry
# 12. Kusto Log Viewing & Debugging

## 12.1 Kusto Log Environment Routing

Key finding (confirmed by Siwen): the certificate determines which environment's logs you can see.

- Using SI certificate -> can only see SI environment logs
- Using Prod certificate -> can only see Prod environment logs
- If namespace is Prod but certificate is SI, logs will be written to the wrong database
Previously discovered SI Kusto logs erroneously written to Prod DB (Image Model Service group); ensure configuration consistency.

## 12.2 Kusto Queries

- SI environment: bingadsppe.AdInsightMT
- Prod environment: bingads.BingAdsTracing
| Environment | Link |
| --- | --- |
| PROD Kusto | https://bingads.kusto.windows.net/ |
| SI Kusto | https://bingadsppe.kusto.windows.net/ |
| DLIS Jarvis Dashboard (Prod) | DLIS Model Metrics \| Jarvis |
| DLIS Jarvis Dashboard (SI) | DLIS Model Metrics \| Jarvis (SI) |

[Attachment] Auto Image Service DLIS Documentation (ChunChen) — includes Kusto log analysis examples

appsvc_info | union appsvc_warn | union appsvc_err | where Timestamp > ago(30min) | where ApplicationName == 'ImgLPRelevanceModel'

The ApplicationName value comes from the application_name field in the config file (confirmed by Siwen).

## 12.3 Polaris Log

- Polaris log shows service startup logs but not container print logs (confirmed by Siwen)
- For more detailed logs, use Kusto or Central Log
-> See "Appendix E" for Central Log queries

## 12.4 Local Kusto Log Testing

Key principle: Integrate Kusto log output during the local development phase. Do not wait until online deployment to start using Kusto logging — you should be able to see Kusto logs during local Docker testing. This lets you catch log format, auth config, and EventHub connection issues early.

Local testing only requires: ① the correct PFX certificate file; ② the matching EventHub namespace config; ③ network access to the EventHub endpoint. With these three prerequisites met, logs from your local Docker container will be sent to Kusto in real-time and can be queried directly in Kusto Explorer.

Test script using AAD Bearer token (MSAL + PFX certificate) authentication:

import msal, requests from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates # 1. Load PFX certificate -> get private_key, thumbprint, public_certificate # 2. Get AAD Bearer token app = msal.ConfidentialClientApplication( client_id=CLIENT_ID, authority=f"https://login.microsoftonline.com/{TENANT_ID}", client_credential={"private_key": ..., "thumbprint": ..., "public_certificate": ...}, ) result = app.acquire_token_for_client(scopes=[DLIS_SCOPE]) # 3. Send request with tracking_data payload = { "prompt": "test", "width": 1344, "height": 768, "tracking_data": { "requestid": f"test-{uuid.uuid4().hex[:8]}", "trackingid": f"test-{uuid.uuid4().hex[:8]}", "callername": "local_test", }, } resp = requests.post(API_URL, json=payload, headers={"Authorization": f"Bearer {token}"}, verify=False, timeout=120)

## 12.5 Common Kusto Log Issues & Debugging Tips

The following lessons are summarized from real debugging experiences during ZImage, Gemma4, and other model deployments.

### Issue 1: EventHub Auth Errors Silently Swallowed

Symptom: Local test returns results normally, but Kusto shows zero logs.

Root cause: kusto_log.py catches all EventHub send exceptions with a bare try/except pass, making certificate or namespace misconfigurations completely invisible.

# ❌ Wrong — silently swallows auth failures try: client.send(event_data) except Exception: pass # logs lost, no indication # ✅ Correct — fail fast, expose config issues try: client.send(event_data) except Exception as e: logger.error(f"EventHub send failed: {e}", exc_info=True) raise # first failure should be immediately visible

### Issue 2: record.msg vs record.getMessage()

Symptom: Kusto log messages show raw template strings (e.g., "%s loaded in %d seconds") instead of formatted values.

Root cause: kusto_log.py uses record.msg which is the unformatted template. Use record.getMessage() to get the fully formatted string.

# ❌ record.msg → "Model %s loaded in %d seconds" # ✅ record.getMessage() → "Model gemma4 loaded in 42 seconds"

### Issue 3: Kusto Logs Lost on Process Crash

Symptom: OOM or CUDA errors during model loading crash the process, but Kusto shows no error logs.

Root cause: KustoHandler uses ScheduledBatchSender for periodic batch sending. When the process crashes, the scheduler thread dies with it and all buffered logs are lost.

# Solution: manually flush in crash handler import signal, atexit def flush_kusto_on_exit(): for handler in logging.root.handlers: if hasattr(handler, 'flush'): handler.flush() atexit.register(flush_kusto_on_exit) signal.signal(signal.SIGTERM, lambda *_: (flush_kusto_on_exit(), sys.exit(1)))

### Issue 4: SI/Prod Certificate-Namespace Mismatch

Symptom: Log sending shows no errors (if exceptions are caught), but Kusto queries return nothing.

Root cause: Using an SI certificate with a Prod EventHub namespace (or vice versa). Auth may pass but messages are routed to the wrong environment.

- SI environment: namespace has "si" suffix, use SI certificate
- Prod environment: namespace has no suffix, use Prod certificate
Recommendation: Put environment config in settings.json and switch via environment variables — avoid hardcoding.

### Issue 5: Logger Level Defaults to WARNING

Symptom: Code has logger.info() calls, but Kusto only shows WARNING and above.

Root cause: Python logging child loggers inherit the root logger's level (WARNING) by default. Without explicit configuration, all INFO and DEBUG logs are filtered out.

# Must explicitly set logger level logger = logging.getLogger("dlis_model") logger.setLevel(logging.INFO) # defaults to WARNING if not set

### Issue 6: Validate Kusto Logs During Local Testing

Important: Do not defer Kusto log validation to the online deployment stage. Integrate Kusto logging during local Docker testing to ensure logs are sent to EventHub correctly.

Prerequisites for local Kusto log testing:

- Correct environment PFX certificate file (SI or Prod)
- settings.json configured with the correct EventHub namespace and topic
- Network access to EventHub endpoint (corporate network or VPN)
Validation steps:

- After starting local Docker, send a test request and check container logs for EventHub send success/failure output
- Simultaneously query the corresponding Kusto environment table to confirm logs have arrived (typically 1-2 minute delay)
- If network restrictions prevent EventHub access, temporarily use a console handler to verify log format, but do full EventHub validation as soon as possible
### Issue 7: EventHub Four Topics Explained

DLIS EventHub provides four topics. Send logs to the correct topic based on log type:

- appsvc_info — general information logs (model loading, request processing, etc.)
- appsvc_warn — warning logs (non-fatal errors, performance degradation, etc.)
- appsvc_err — error logs (exceptions, crash info, etc.)
- appsvc_perf — performance logs (inference latency, throughput metrics, etc.)
Note: If you only send to appsvc_info, querying the errors table in Kusto will return nothing.

# 13. Common Issues & Solutions

## Issue 1: Container OOM Killed (CPU memory, not GPU)

Root Cause: OaasWrapper cannot find _opt directory -> falls back to BaseLLM -> loads all weights on CPU -> exceeds CPU memory limit

Solution: Ensure _opt directory structure is correct (Approach A), or use vllm.LLM() directly (Approach B, recommended)

## Issue 2: CUDA_VISIBLE_DEVICES UUID Format

Root Cause: DLIS sets CUDA_VISIBLE_DEVICES to GPU UUID format; vLLM internal int() conversion fails

Solution: In run.sh: unset CUDA_VISIBLE_DEVICES + in model.py: convert UUID to integer indices

## Issue 3: /Model is a Read-Only Filesystem

Root Cause: DLIS mounts Cosmos data to /Model in read-only mode

Solution: Writable Mirror approach: create mirror directory in /tmp, use symlinks to point to read-only model files

## Issue 4: Unable to find exposed port 8888

Root Cause: Usually not a port issue but a model loading crash (OOM, etc.); HTTP server never started

Solution: Fix model loading issues first; also ensure Dockerfile has EXPOSE 8888

## Issue 5: Pipeline Build pip install Timeout

Root Cause: CI agent cannot directly connect to pypi.org

Solution: Add Tsinghua PyPI mirror (see Appendix B)

## Issue 6: OaasWrapper _create_runner() Silently Swallows Exceptions

Root Cause: _create_runner() has try/except that catches all exceptions, prints, then returns None

Solution: Change to logger.error(exc_info=True), or use Approach B directly

## Issue 7: opt_type.txt Newline Character

Root Cause: echo "llm" writes with trailing newline

Solution: Use printf "llm" instead of echo "llm"

## Issue 8: Cosmos Mount Only Mounts Top-Level Directory

Root Cause: Placing certificates in subdirectories causes "not found" errors

Solution: Place certificates in the Cosmos root directory, not in subdirectories

## Issue 9: Docker in Logs Mismatches Config

Root Cause: Config file does not match the actual running image

Solution: Check that ModelPath in Polaris Job config points to the correct image tag

## Issue 10: Local Test Returns Results but Kusto Logs Are Empty

Root Cause: Using the wrong environment's certificate (e.g., namespace is Prod but certificate is SI)

Solution: Ensure namespace and certificate environment are consistent (see Sections 11 and 12)

Appendices

# Appendix A: Base Docker Image Selection & Updates

## A.1 Fast Build Approach (Recommended): Dockerfile_vllm_fast

FROM vllm/vllm-openai:latest # Already includes vllm, torch, transformers - no compilation needed # If a specific transformers version is needed: RUN python3 -m pip install transformers==5.5.3

- Pros: Build < 1 second, includes pre-compiled vllm + torch
- Cons: latest version is uncontrolled, may be updated upstream
## A.2 Full Build Approach: Dockerfile_vllm_0.10.0

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 # Key: install the correct version of torch first, then install vllm RUN pip install torch==2.10.0 torchvision==0.25.0 \ --index-url https://download.pytorch.org/whl/cu128 RUN pip install vllm==0.19.0

## A.3 Version Compatibility Experience

| Issue | Root Cause | Solution |
| --- | --- | --- |
| vllm/_C.abi3.so: undefined symbol | torch ABI mismatch | Install torch first; do not add torch/torchvision to requirements |
| torchvision::nms does not exist | Version mismatch | Ensure torchvision matches torch version |
| num_scheduler_steps not accepted | Removed in vllm 0.19.0 | Remove from vllm_runner.py |
| Gemma4VideoProcessor requires Torchvision | torchvision uninstalled | Reinstall the correct version |

Principle: Do not add torch, torchvision, or transformers to requirements-vllm.txt; let Dockerfile manage them.

## A.4 Local Manual Build Test

cd OaaS_LLMTemplate IMAGE_TAG="local-test" # Block 1: Build base image sudo docker build -t my-vllm-base:$IMAGE_TAG \ --file pipeline/Dockerfile_vllm_fast pipeline/ # Block 2: Install OaaS code and dependencies sudo docker run -d --name temp my-vllm-base:$IMAGE_TAG sleep infinity sudo docker cp . temp:/ sudo docker exec temp chmod +x /dlis_model/run.sh /dlis_model/async_run.sh /LLMModelOptimization.sh sudo docker exec temp python3 -m pip install -r /requirements-common.txt sudo docker exec temp python3 -m pip install -r /requirements-vllm.txt sudo docker exec temp python3 -m pip install -e / sudo docker commit temp my-final:$IMAGE_TAG sudo docker rm -f temp

# Appendix B: Tsinghua Mirror Configuration

DLIS CI pipeline agents and Docker builds may not be able to directly connect to pypi.org.

## B.1 CI Pipeline (azure-pipelines-unified.yml)

variables: PIP_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple UV_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple

## B.2 Docker Build (build_vllm_image.sh)

docker build \ --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \ --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \ --build-arg PIP_EXTRA_INDEX_URL=https://pypi.org/simple \ ... # docker exec installation phase: PIP_ARGS="-i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple" docker exec container pip install $PIP_ARGS -r requirements.txt

## B.3 Dockerfile Parameter Reception

ARG PIP_INDEX_URL ARG UV_INDEX_URL ARG PIP_EXTRA_INDEX_URL ARG UV_EXTRA_INDEX_URL

Note: Cannot use docker exec -e environment variable approach for mirror params; must use --build-arg or CLI arguments.

# Appendix C: Model Code Writing Guide

## C.1 model.py Core Structure

class ModelImp: def __init__(self): # Model initialization: load engine, configure parameters pass def Eval(self, data): # Single inference pass def EvalBatch(self, data_list): # Batch inference pass

## C.2 Approach A: Using OaasWrapper (For Simple LLM Inference)

from llm_opt.oaas_wrapper_v2 import OaasWrapper class ModelImp: def __init__(self): self.oaas_wrapper = OaasWrapper("model_dir_name", is_llm_model=True) def Eval(self, data): prompts = preprocess(data) outputs = self.oaas_wrapper.run(prompts) return postprocess(outputs)

- Requires opt_type.txt and best_setting.json in the _opt directory
Note: Do not set the quantization field in best_setting.json. vLLM auto-detects from model config.json.

## C.3 Approach B: Using vLLM Directly (Recommended, More Control)

from vllm import LLM, SamplingParams class ModelImp: def __init__(self): # CUDA_VISIBLE_DEVICES UUID fix cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "") if cuda_env and not cuda_env.replace(",", "").isdigit(): gpu_count = len(cuda_env.split(",")) os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count)) self.llm = LLM( model="/Model/model-name", tensor_parallel_size=1, gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")), trust_remote_code=True, dtype="auto", max_model_len=8192, enable_prefix_caching=True, ) self.sampling_params = SamplingParams( temperature=1.0, top_p=0.95, top_k=64, max_tokens=128, stop=["</end_token>"], ) def Eval(self, data): prompts = preprocess(data) outputs = self.llm.generate(prompts, self.sampling_params) return [o.outputs[0].text for o in outputs]

Advantages of Approach B:

- Removes the OaasWrapper middle layer; initialization failures are reported directly
- No need for _opt directory or best_setting.json
- Inference parameters are written directly in code, transparent and controllable
Siwen also uses vllm.LLM directly instead of OaasWrapper; this has become team consensus.

## C.4 Non-LLM Models (e.g., ZImage Diffusion)

class ModelImp: def __init__(self): from diffusers import ZImagePipeline model_path = os.environ.get("ZIMAGE_MODEL_PATH", "/Model/model_name") self.pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16) self.pipe.to("cuda:0") def Eval(self, data): result = self.pipe(prompt=data["prompt"], width=data["width"], height=data["height"]) ...

## C.5 Multi-Step Inference (e.g., Gemma4 Two-Step)

def Eval(self, data): # Step 1: Generate scene concepts step1_outputs = self._run_vllm(step1_prompts, self.step1_params) # Step 2: Expand into detailed prompts step2_outputs = self._run_vllm(step2_prompts, self.step2_params) return self.processor.postprocess(step2_outputs)

Key: Step 1 and Step 2 should use independent SamplingParams (different max_tokens and stop tokens).

## C.6 Tokenizer Thinking Mode Issue

Some quantized model tokenizers have a thinking prefix built into chat_template that needs patching:

tokenizer = self.llm.get_tokenizer() if hasattr(tokenizer, 'chat_template') and '<|channel>thought' in (tokenizer.chat_template or ''): tokenizer.chat_template = tokenizer.chat_template.replace( "<|channel>thought\n<channel|>", "" )

# Appendix D: External settings.json Configuration Override

## D.1 Background

If configurations in config.py are hardcoded, switching between SI/Prod environments requires code changes and image rebuilds. Reference Hanbang's implementation (user/hanbangliang/img-outpainting-v1 branch) for overriding via an external JSON file on Cosmos.

## D.2 Implementation

from pydantic_settings import BaseSettings, SettingsConfigDict DEFAULT_SETTINGS_JSON_PATH = os.environ.get("SETTINGS_JSON_PATH", "/Model/settings.json") class Settings(BaseSettings): model_config = SettingsConfigDict(extra="ignore") eventhub_namespace: str = "aggregation-logging.servicebus.windows.net" certificate_path: str = os.path.join("/Model", "AggSvcAuthCert-prod.pfx") ...

## D.3 Configuration Priority

- 1. init kwargs (explicit code parameters)
- 2. Environment variables (e.g., EVENTHUB_NAMESPACE=...)
- 3. JSON file (/Model/settings.json)
- 4. Code default values
## D.4 Usage

Place settings.json in the Cosmos root directory, writing only fields that need overriding:

{"eventhub_namespace": "aggregation-si-logging.servicebus.windows.net", "certificate_path": "/Model/AggSvcAuthCert-si.pfx"}

Without settings.json, code defaults are used (Prod environment).

## D.5 EventHub Credential Fault Tolerance

def _try_get_credential(tenant_id): try: return CertificateCredential(...) except Exception as e: logger.info("EventHub credential unavailable, kusto logs local-only") return None

Certificate loading failure should not block container startup; silently degrade to local logging.

# Appendix E: Central Log Debugging Tool

Use Central Log to debug Polaris Jobs and view container output logs (recommended by Desheng):

SELECT machine_name, log_level, log_time, description FROM dlissensitivelog WHERE file_name LIKE 'DLMSUserLog_ContainerOutput%.log' AND log_time BETWEEN TIMESTAMP '2026-02-27 00:00:00' AND TIMESTAMP '2026-02-28 18:00:00' AND machine_name = '<your_machine_name>' LIMIT 10000;

# Appendix F: Reference Links

| Resource | Link / Description |
| --- | --- |
| DLIS LLM Engine Official Docs | Optimizing a LLM model with DLIS LLM engine |
| OaaS Template Repo | OaaS_LLMTemplate (ADO Git) |
| Gen1->Gen2 Data Migration Tools | Data Transfer Tools (DLIS Wiki) |
| Central Log Query Guide | How to Use Central Log (DLIS Wiki) |
| DLIS Model Building Guide (Hao Zhang) | How_to_Build_Your_Own_DLIS_Model.docx |
| OaaS Deployment Guide v2 (ChangXu) | DLIS_Model_DeploymentWith_OaaS_v2.docx |
| Kusto Logs (SI) | bingadsppe.AdInsightMT \| Azure Data Explorer |
| Kusto Logs (Prod) | bingads.BingAdsTracing \| Azure Data Explorer |
| Auto Image Service DLIS Doc (ChunChen) | Auto Image Service — DLIS Model Documentation |
| DLIS Deployment Walkthrough (Siwen & Desheng) | Call with Desheng Cui — 2026-03-26 Recording |

[Attachment] DLIS_Model_DeploymentWith_OaaS_v2.docx (ChangXu)

[Attachment] How_to_Build_Your_Own_DLIS_Model.docx (Hao Zhang)

[Attachment] Auto Image Service — DLIS Model Documentation (ChunChen)

[Attachment] Call with Desheng Cui — 2026-03-26 DLIS Deployment Walkthrough (Siwen & Desheng)

# Appendix G: Gemma4 DLIS Deployment Iteration Summary

| Deployment # | Issue | Fix |
| --- | --- | --- |
| #1-#3 | _opt directory invisible -> OOM | Confirmed Cosmos sync cannot resolve |
| #4 | /Model read-only, cannot create _opt | Writable mirror approach |
| #5 | vLLM init failure silently swallowed | _create_runner() changed to raise exception |
| #6 | Changed to raise -> crash loop | Reverted to return None + logger.error |
| #7 | CUDA_VISIBLE_DEVICES UUID format | UUID -> integer index conversion |
| #8 | Cosmos directory + stale model.py | New flat directory + removed OaasWrapper |

Core lesson: OaasWrapper middle layer complexity far outweighs its benefits. Recommend using vllm.LLM() directly (Approach B).

# Appendix H: Revision History

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| v2 | 2026-02-26 | ChangXu | Initial document, complete deployment guide |
| v4 | 2026-04-22 | Jinjin Chen | Consolidated Gemma4/ZImage hands-on experience |
| v5 | 2026-04-22 | Jinjin Chen | Consolidated ChangXu v2, settings.json, Kusto testing |
| v5.1 | 2026-04-22 | Jinjin Chen | Restructured: main flow + appendices; consolidated team Teams chat insights |
| v5.2 | 2026-04-23 | Jinjin Chen | English version; added Kusto/Jarvis links and ChunChen doc reference |
