"""
Filter annotated LPs:
  - 3-annotator majority vote per (UrlHash, PromptTag)
  - Keep LPs where: bad_count <= 1 AND all 5 prompt texts are distinct
Output: filtered rows from the prompt TSV, saved to filtered_lps.tsv
"""
import csv, re, sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")

ANNOTATION_TSV = "UHRS_Task_lp_labeling_0306_random1K_Quality.tsv"
PROMPT_TSV     = "UHRS2K_SD_Sample1000.GPT5PromptsCreator.0131-2-latest.tsv"
OUTPUT_TSV     = "filtered_lps.tsv"

LABEL_ORDER = {"Good": 0, "Fair": 1, "Bad": 2, "Imageloadfail": 3, "Logo": 3}

def parse_lp_id(lp_id):
    m = re.match(r"(Random\d+_\d+)_(Prompt\d+)_", lp_id)
    return (m.group(1), m.group(2)) if m else (None, None)

def majority_vote(labels):
    """Return most common label among 3 annotators. Tie-break: worse label wins."""
    c = Counter(labels)
    # sort by count desc, then by severity asc so ties go to worse label
    return sorted(c, key=lambda l: (-c[l], LABEL_ORDER.get(l, 99)))[0]

# ── Step 1: aggregate 3 annotations per (UrlHash, PromptTag) ─────────────────
judgments = defaultdict(list)  # (url_hash, prompt_tag) -> [label, ...]
with open(ANNOTATION_TSV, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        uh, pt = parse_lp_id(row["lp_id"])
        if uh:
            judgments[(uh, pt)].append(row["FinalDecision"])

agg_label = {k: majority_vote(v) for k, v in judgments.items()}

print(f"Unique (UrlHash, PromptTag) pairs: {len(agg_label)}")
label_dist = Counter(agg_label.values())
print(f"Aggregated label distribution: {dict(label_dist)}")

# ── Step 2: group by UrlHash, apply filter ───────────────────────────────────
with open(PROMPT_TSV, encoding="utf-8") as f:
    prompt_rows = list(csv.DictReader(f, delimiter="\t"))

# index prompt rows by (UrlHash, Tag)
prompt_index = {(r["UrlHash"], r["Tag"]): r for r in prompt_rows}

url_hashes = sorted({r["UrlHash"] for r in prompt_rows})
print(f"\nTotal LPs in prompt file: {len(url_hashes)}")

passed, fail_bad, fail_distinct, fail_missing = [], 0, 0, 0

for uh in url_hashes:
    tags = ["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"]
    labels = []
    prompts = []
    missing = False
    for tag in tags:
        key = (uh, tag)
        if key not in agg_label:
            missing = True
            break
        labels.append(agg_label[key])
        pr = prompt_index.get(key)
        prompts.append(pr["Prompt"].strip() if pr else "")

    if missing:
        fail_missing += 1
        continue

    bad_count = labels.count("Bad")
    all_distinct = len(set(prompts)) == 5

    if bad_count > 1:
        fail_bad += 1
    elif not all_distinct:
        fail_distinct += 1
    else:
        passed.append(uh)

print(f"Fail (missing annotation): {fail_missing}")
print(f"Fail (bad > 1)           : {fail_bad}")
print(f"Fail (duplicate prompts) : {fail_distinct}")
print(f"Passed                   : {len(passed)}")

# ── Step 3: write output ─────────────────────────────────────────────────────
passed_set = set(passed)
out_rows = [r for r in prompt_rows if r["UrlHash"] in passed_set]

# add aggregated label column
for r in out_rows:
    r["agg_label"] = agg_label.get((r["UrlHash"], r["Tag"]), "")

fieldnames = list(prompt_rows[0].keys()) + ["agg_label"]
with open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\nSaved {len(out_rows)} rows ({len(passed)} LPs) to {OUTPUT_TSV}")
