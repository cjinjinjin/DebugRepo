#!/bin/bash
# 检查 Gemma 4 推理中间结果：查看各 GPU shard 进度、合格率、样例内容
#
# 自动查找 /tmp/gemma4_shards_* 下最新修改的目录
#
# Usage:
#   bash Gemma4/check_inference_results.sh                # 查看 /tmp 下最新 shard 进度
#   bash Gemma4/check_inference_results.sh <output.jsonl>  # 检查合并后的最终结果

set -e

# ── 判断输入 ──────────────────────────────────────────────────────────────
if [ -n "${1:-}" ] && [ -f "${1}" ]; then
    OUTPUT_FILE="$1"
    MODE="single"
else
    # 找 /tmp 下最近修改的 gemma4_shards 目录
    SHARD_DIR=$(find /tmp -maxdepth 1 -name 'gemma4_shards_*' -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "${SHARD_DIR}" ]; then
        echo "[ERROR] 未找到 /tmp/gemma4_shards_* 目录"
        exit 1
    fi
    MODE="shards"
fi

echo "============================================"
if [ "${MODE}" = "shards" ]; then
    echo "检查中间结果: ${SHARD_DIR}"
else
    echo "检查推理结果: $(basename "${OUTPUT_FILE}")"
fi
echo "============================================"

# ── Shard 模式 ────────────────────────────────────────────────────────────
if [ "${MODE}" = "shards" ]; then
    echo ""
    echo "--- 各 GPU shard 进度 ---"
    TOTAL_DONE=0
    TOTAL_INPUT=0
    for i in 0 1 2 3 4 5 6 7; do
        inp="${SHARD_DIR}/shard_${i}_input.jsonl"
        out="${SHARD_DIR}/shard_${i}_output.jsonl"
        if [ -f "$inp" ]; then
            IN_LINES=$(wc -l < "$inp")
            TOTAL_INPUT=$((TOTAL_INPUT + IN_LINES))
            if [ -f "$out" ]; then
                OUT_LINES=$(wc -l < "$out")
                TOTAL_DONE=$((TOTAL_DONE + OUT_LINES))
                echo "  GPU ${i}: ${OUT_LINES}/${IN_LINES} done"
            else
                echo "  GPU ${i}: 0/${IN_LINES} (no output yet)"
            fi
        fi
    done
    echo ""
    echo "  总进度: ${TOTAL_DONE}/${TOTAL_INPUT}"

    echo ""
    echo "--- 合格率统计 (所有已完成 shard) ---"
    python3 -c "
import json, sys, glob

shard_dir = '${SHARD_DIR}'
files = sorted(glob.glob(shard_dir + '/shard_*_output.jsonl'))
total = 0
n_compliant = 0
n_tags = 0
output_lens = []

for path in files:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            if r.get('format_compliant', False):
                n_compliant += 1
            prompts = r.get('generated_prompts', [])
            raw = r.get('raw_output', '')
            if len(prompts) == 5 and (not prompts[0] or prompts[0] != raw.strip()):
                n_tags += 1
            output_lens.append(len(raw))

if total == 0:
    print('  (暂无输出)')
    sys.exit(0)

print(f'  总样本数:          {total}')
print(f'  Format compliant:  {n_compliant}/{total} ({100*n_compliant/total:.1f}%)')
print(f'  All 5 tags:        {n_tags}/{total} ({100*n_tags/total:.1f}%)')
print(f'  平均输出长度:      {sum(output_lens)/len(output_lens):.0f} chars')
print(f'  最短/最长输出:     {min(output_lens)}/{max(output_lens)} chars')
"

    # 取最后一个 shard 的最后一条作为样例
    SAMPLE_FILE=$(ls "${SHARD_DIR}"/shard_*_output.jsonl 2>/dev/null | tail -1)
    if [ -n "${SAMPLE_FILE}" ]; then
        OUTPUT_FILE="${SAMPLE_FILE}"
    else
        exit 0
    fi
fi

# ── 单文件模式 ────────────────────────────────────────────────────────────
if [ "${MODE}" = "single" ]; then
    TOTAL=$(wc -l < "${OUTPUT_FILE}")
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo ""
    echo "已完成样本数: ${TOTAL}"
    echo "文件大小: ${FILE_SIZE}"

    echo ""
    echo "--- 合格率统计 ---"
    python3 -c "
import json, sys

path = '${OUTPUT_FILE}'
total = 0
n_compliant = 0
n_tags = 0
output_lens = []

with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        total += 1
        if r.get('format_compliant', False):
            n_compliant += 1
        prompts = r.get('generated_prompts', [])
        raw = r.get('raw_output', '')
        if len(prompts) == 5 and (not prompts[0] or prompts[0] != raw.strip()):
            n_tags += 1
        output_lens.append(len(raw))

if total == 0:
    print('  (空文件)')
    sys.exit(0)

print(f'  总样本数:          {total}')
print(f'  Format compliant:  {n_compliant}/{total} ({100*n_compliant/total:.1f}%)')
print(f'  All 5 tags:        {n_tags}/{total} ({100*n_tags/total:.1f}%)')
print(f'  平均输出长度:      {sum(output_lens)/len(output_lens):.0f} chars')
print(f'  最短/最长输出:     {min(output_lens)}/{max(output_lens)} chars')
"
fi

# ── 最后 3 条样本摘要 ────────────────────────────────────────────────────
echo ""
echo "--- 最后 3 条样本摘要 ---"
python3 -c "
import json

path = '${OUTPUT_FILE}'
records = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

for r in records[-3:]:
    rid = r.get('id', '?')
    compliant = r.get('format_compliant', False)
    prompts = r.get('generated_prompts', [])
    raw_len = len(r.get('raw_output', ''))
    status = 'OK' if compliant else 'FAIL'
    print(f'  [{status}] id={rid}, prompts={len(prompts)}, raw_len={raw_len}')
    if prompts:
        p1 = prompts[0][:80].replace('\n', ' ')
        print(f'         Prompt1: {p1}...')
    print()
"

echo "============================================"
