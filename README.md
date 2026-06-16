# Debug repo

This repository collects image-quality, image-to-landing-page relevance, creative ads, prompt-following, fine-tuning, deployment, and presentation/evaluation experiments. It is organized as a research workspace: most top-level folders are self-contained experiments or artifact bundles, while shared helper scripts live at the root or in `standalone_tools`.

## Repository map

- `.claude` - Local Claude/Copilot-style tool settings for this workspace.
- `.scope` - Scope configuration files for local tooling.
- `%temp%` - Tracked temporary download artifacts preserved for reproducibility.
- `APIUsage` - Papyrus / GPT image API access notes and sample requests.
- `AutoGenEvaluationSampling` - Sampling utilities and inputs for AutoGen-style evaluation runs.
- `CreativeAdsAgent` - Crawler, RAG, agent, and utility code for creative ads workflows.
- `CreativeAdsPrompt` - Prompt assets and reward-prompt material for creative ads evaluation.
- `decoration-relevance-video` - Web-video presentation project for decoration relevance evaluation.
- `dino_v3` - DINOv3-related experiment assets and notes.
- `doc_images` - Image assets referenced by repository documentation.
- `Gemma4` - Gemma 4 experiment scripts, data, and result artifacts.
- `Gemma4Deploy` - Deployment materials and notes for Gemma 4 related models.
- `ImgLPRelevance6` - Image-to-landing-page relevance evaluation scripts and result files.
- `LPCreativeAdsEvaluation` - Landing-page creative ads evaluation scripts and datasets.
- `Potential_Ads` - Small candidate-ad data and notes used by evaluation experiments.
- `PromptFollowingEvalution` - Prompt-following evaluation scripts and supporting files.
- `QwenFinetune` - Qwen fine-tuning data preparation, raw data, and training artifacts.
- `simple_image_quality_evaluation` - Simple image quality evaluation inputs, scripts, and output examples.
- `skillopt-video-evaluation` - Skill-optimization benchmark data, topics, skills, and generated rollout predictions.
- `standalone_tools` - Standalone helper scripts that can be used outside a single experiment folder.
- `teams_conversations` - Exported Teams conversation captures used as source/reference material.
- `ZImage` - ZImage generation and evaluation helper scripts.

## Root files

- `batch_zimage.py` runs batch ZImage generation jobs.
- `test_gpt55_chat.py` and `test_gpt55_two_step_benchmark.py` exercise GPT-5.5 chat and two-step benchmark flows.
- `gen_docx.py` and `gen_docx_en.py` generate Word documents from repository notes.
- `INFERENCE_ACCELERATION_SHARING*.md`, `DLIS_Model_Deployment_Guide_v4.md`, and weekly update files are project documentation and sharing materials.

## Working conventions

- Treat each top-level experiment directory as an independent unit unless a README states otherwise.
- Keep raw data, prepared data, generated outputs, and presentation build artifacts in their existing directory families.
- Before running scripts, inspect the local README and script arguments for required credentials, model endpoints, or input paths.
