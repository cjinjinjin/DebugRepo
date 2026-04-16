#!/bin/bash
# Gemma4 DLIS 端到端测试脚本
# 用法: bash test_request.sh [HOST:PORT]

ENDPOINT="${1:-http://localhost:8886}"

echo "=========================================="
echo "测试 Gemma4 DLIS 服务: $ENDPOINT"
echo "=========================================="

# 测试 1: 单请求 Eval
echo ""
echo "--- 测试 1: 单请求 (Eval) ---"
time curl -s -X POST "$ENDPOINT/score" \
  -H "Content-Type: application/json" \
  -d '{
    "landing_page_content": "Welcome to TrailMaster Outdoor Gear. Premium hiking boots, ultralight backpacks, and camping essentials for your next adventure. Free shipping on orders over $99.",
    "url": "https://trailmaster.example.com",
    "num_prompts": 5
  }' | python3 -m json.tool

echo ""
echo "--- 测试 2: 最少输入 ---"
time curl -s -X POST "$ENDPOINT/score" \
  -H "Content-Type: application/json" \
  -d '{
    "landing_page_content": "Buy the best coffee beans online. Fresh roasted daily."
  }' | python3 -m json.tool

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
