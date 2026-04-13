#!/bin/bash
# Download Gemma 4 E4B-it (4.5B effective, 8B total, Dense) from HuggingFace
#
# Usage:
#   HF_TOKEN=hf_xxx bash Gemma4/download_model_e4b.sh

set -e

python Gemma4/download_model.py --repo_id google/gemma-4-E4B-it
