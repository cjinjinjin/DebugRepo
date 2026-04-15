#!/bin/bash
# Download Gemma 4 E2B-it (2B effective, Dense) from HuggingFace
#
# Usage:
#   HF_TOKEN=hf_xxx bash Gemma4/download_model_e2b.sh

set -e

python Gemma4/download_model.py --repo_id google/gemma-4-E2B-it
