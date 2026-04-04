"""
Generate DPO format-preference training data by corrupting well-formatted SFT outputs.

For each SFT sample with a correctly formatted assistant response, we apply 2-3
random corruption strategies to produce format-broken "rejected" examples paired
with the original correct "chosen" response.  This teaches the model to prefer
proper output format (think block + 5 PromptN tags) via DPO.

12 corruption strategies across 5 categories:
  A. Missing tags:     drop_one_prompt, drop_two_prompts, drop_all_prompts
  B. Think violations: unclosed_think, no_think, bloated_think
  C. Structural:       wrong_tag_order, duplicate_tag_index, extra_text_after
  D. Length:           overly_long_prompt
  E. Repetition:       high_repetition, copy_paste_prompts

Usage:
  python prepare_dpo_format.py
  python prepare_dpo_format.py --num_corruptions 3 --eval_ratio 0.1 --seed 42
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Import reward_fn for cross-validation
sys.path.insert(0, str(SCRIPT_DIR))
from reward_grpo import reward_fn

SYSTEM_PROMPT_COT = (
    "You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, "
    "specialized in high-performing Native Advertisement visuals.\n\n"
    "Given a landing page URL and its extracted content fields, your task is to "
    "generate five (5) high-quality English image generation prompts for Native Ads.\n\n"
    "First, reason about the product inside <think>...</think> tags. "
    "Extract the following from the landing page:\n"
    "- ProductType: Physical Product / Digital Product / Service\n"
    "- SpecificProduct: concise noun phrase\n"
    "- Category: broad product/service category\n"
    "- VisualAnchors: 2-3 specific physical elements implied by the page\n"
    "- LifestyleVibe: emotional tone of the experience\n"
    "- CoreValueSignals: up to 3 from [professional, premium, affordable, "
    "efficient, reliable, simple]\n\n"
    "Then output exactly 5 prompts. Each prompt must:\n"
    "- Be <=150 words\n"
    "- Embed all safety, realism, quality, and exclusion constraints\n"
    "- Feel native and non-promotional\n"
    "- Show the product outcome or value naturally in context\n"
    "- Avoid stereotypes, text/logos in image, and stock-photo aesthetics\n"
    "- Ensure correct anatomy, natural hands, sharp focus, clean composition\n\n"
    "Output format:\n"
    "<think>\n"
    "ProductType: ...\n"
    "SpecificProduct: ...\n"
    "Category: ...\n"
    "VisualAnchors: ...\n"
    "LifestyleVibe: ...\n"
    "CoreValueSignals: ...\n"
    "</think>\n"
    "<Prompt1>...</Prompt1>\n"
    "<Prompt2>...</Prompt2>\n"
    "<Prompt3>...</Prompt3>\n"
    "<Prompt4>...</Prompt4>\n"
    "<Prompt5>...</Prompt5>"
)

FILLER_PHRASES = [
    "The scene is set in a beautiful environment with natural lighting.",
    "A warm and inviting atmosphere fills the composition.",
    "The overall mood is calm and professional.",
    "Colors are balanced and harmonious throughout.",
    "Every element contributes to a cohesive visual narrative.",
    "Details are sharp and textures are realistic.",
    "The framing creates a sense of depth and dimension.",
    "Soft shadows and highlights add visual interest.",
    "The composition follows the rule of thirds naturally.",
    "A subtle gradient in the background adds sophistication.",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_assistant(text: str):
    """Extract think block and prompt contents from a well-formatted response."""
    think_match = re.search(r"<think>([\s\S]*?)</think>", text)
    think_block = think_match.group(0) if think_match else ""
    think_content = think_match.group(1) if think_match else ""

    prompts = {}
    for i in range(1, 6):
        m = re.search(rf"<Prompt{i}>([\s\S]*?)</Prompt{i}>", text)
        if m:
            prompts[i] = m.group(1)
    return think_block, think_content, prompts


def rebuild_response(think_block: str, prompts: dict) -> str:
    """Rebuild response from think block and prompt dict."""
    parts = []
    if think_block:
        parts.append(think_block)
    for i in sorted(prompts.keys()):
        parts.append(f"<Prompt{i}>{prompts[i]}</Prompt{i}>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Corruption functions — each returns corrupted text
# ---------------------------------------------------------------------------

def corrupt_drop_one_prompt(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if len(prompts) < 5:
        return None
    drop_idx = random.choice(list(prompts.keys()))
    new_prompts = {k: v for k, v in prompts.items() if k != drop_idx}
    return rebuild_response(think_block, new_prompts)


def corrupt_drop_two_prompts(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if len(prompts) < 5:
        return None
    drop_idxs = random.sample(list(prompts.keys()), 2)
    new_prompts = {k: v for k, v in prompts.items() if k not in drop_idxs}
    return rebuild_response(think_block, new_prompts)


def corrupt_drop_all_prompts(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if not think_block:
        return None
    return think_block


def corrupt_unclosed_think(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if not think_block:
        return None
    broken_think = f"<think>{think_content}"  # no </think>
    return rebuild_response(broken_think, prompts)


def corrupt_no_think(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if not prompts:
        return None
    return rebuild_response("", prompts)


def corrupt_bloated_think(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if not think_content:
        return None
    words = think_content.split()
    while len(words) < 450:
        words.extend(think_content.split())
    bloated = " ".join(words[:450])
    new_think = f"<think>{bloated}</think>"
    return rebuild_response(new_think, prompts)


def corrupt_wrong_tag_order(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if len(prompts) < 4:
        return None
    keys = list(prompts.keys())
    # Swap two adjacent indices
    i = random.randint(0, len(keys) - 2)
    keys[i], keys[i + 1] = keys[i + 1], keys[i]
    parts = [think_block] if think_block else []
    for k in keys:
        parts.append(f"<Prompt{k}>{prompts[k]}</Prompt{k}>")
    return "\n".join(parts)


def corrupt_duplicate_tag_index(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if len(prompts) < 5:
        return None
    keys = sorted(prompts.keys())
    dup_src = random.choice(keys[:-1])
    dup_tgt = keys[-1]  # replace last tag's index with dup_src's index
    parts = [think_block] if think_block else []
    for k in keys:
        idx = dup_src if k == dup_tgt else k
        parts.append(f"<Prompt{idx}>{prompts[k]}</Prompt{idx}>")
    return "\n".join(parts)


def corrupt_extra_text_after(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if not prompts:
        return None
    base = rebuild_response(think_block, prompts)
    extra = "\n\nI hope these prompts are helpful for your native ad campaign! Let me know if you need any modifications or additional prompts."
    return base + extra


def corrupt_overly_long_prompt(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if len(prompts) < 3:
        return None
    new_prompts = dict(prompts)
    targets = random.sample(list(prompts.keys()), min(2, len(prompts)))
    for idx in targets:
        original = prompts[idx]
        padding = " ".join(random.choices(FILLER_PHRASES, k=8))
        new_prompts[idx] = original + " " + padding
    return rebuild_response(think_block, new_prompts)


def corrupt_high_repetition(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if len(prompts) < 3:
        return None
    repeated_phrase = "natural lighting with warm golden tones and soft shadows creating depth"
    new_prompts = dict(prompts)
    for idx in list(prompts.keys())[:3]:
        new_prompts[idx] = repeated_phrase + ". " + prompts[idx] + " " + repeated_phrase
    return rebuild_response(think_block, new_prompts)


def corrupt_copy_paste_prompts(text: str, think_block: str, think_content: str, prompts: dict) -> str:
    if len(prompts) < 5:
        return None
    src_idx = random.choice(list(prompts.keys()))
    src_content = prompts[src_idx]
    targets = [k for k in prompts if k != src_idx]
    replace_targets = random.sample(targets, min(2, len(targets)))
    new_prompts = dict(prompts)
    for t in replace_targets:
        new_prompts[t] = src_content
    return rebuild_response(think_block, new_prompts)


CORRUPTION_REGISTRY = {
    "drop_one_prompt":    corrupt_drop_one_prompt,
    "drop_two_prompts":   corrupt_drop_two_prompts,
    "drop_all_prompts":   corrupt_drop_all_prompts,
    "unclosed_think":     corrupt_unclosed_think,
    "no_think":           corrupt_no_think,
    "bloated_think":      corrupt_bloated_think,
    "wrong_tag_order":    corrupt_wrong_tag_order,
    "duplicate_tag_index": corrupt_duplicate_tag_index,
    "extra_text_after":   corrupt_extra_text_after,
    "overly_long_prompt": corrupt_overly_long_prompt,
    "high_repetition":    corrupt_high_repetition,
    "copy_paste_prompts": corrupt_copy_paste_prompts,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/sft_train_cot.jsonl")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--num_corruptions", type=int, default=3,
                        help="Number of corruption variants per SFT sample (2-4)")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SFT data
    input_path = SCRIPT_DIR / args.input_file
    samples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} SFT samples from {input_path}")

    corruption_names = list(CORRUPTION_REGISTRY.keys())
    all_pairs = []
    skipped = 0
    reward_violations = 0

    for sample in samples:
        messages = sample.get("messages", [])
        # SFT schema: messages = [system, user, assistant]
        system_msg = None
        user_msg = None
        assistant_msg = None
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]

        if not assistant_msg or not user_msg:
            skipped += 1
            continue

        think_block, think_content, prompts = parse_assistant(assistant_msg)
        if len(prompts) < 5 or not think_block:
            skipped += 1
            continue

        # Select random corruptions for this sample
        n = min(args.num_corruptions, len(corruption_names))
        selected = random.sample(corruption_names, n)

        for cname in selected:
            corrupt_fn = CORRUPTION_REGISTRY[cname]
            corrupted = corrupt_fn(assistant_msg, think_block, think_content, prompts)
            if corrupted is None or corrupted == assistant_msg:
                continue

            # Cross-validate with reward_fn
            chosen_reward = reward_fn([assistant_msg])[0]
            rejected_reward = reward_fn([corrupted])[0]
            if chosen_reward <= rejected_reward:
                reward_violations += 1
                continue

            url_hash = sample.get("url_hash", sample.get("id", "unknown"))
            lp_url = sample.get("lp_url", "")

            pair = {
                "id": f"{url_hash}_fmt_{cname}",
                "url_hash": url_hash,
                "lp_url": lp_url,
                "corruption_type": cname,
                "system": system_msg or SYSTEM_PROMPT_COT,
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "rejected_messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": corrupted},
                ],
            }
            all_pairs.append(pair)

    print(f"\nGenerated {len(all_pairs)} DPO format-preference pairs")
    print(f"Skipped samples (incomplete format): {skipped}")
    print(f"Reward violations (chosen <= rejected): {reward_violations}")

    # Train/eval split by url_hash
    unique_hashes = sorted({p["url_hash"] for p in all_pairs})
    random.shuffle(unique_hashes)
    n_eval = max(1, int(len(unique_hashes) * args.eval_ratio))
    eval_set = set(unique_hashes[:n_eval])

    train_pairs = [p for p in all_pairs if p["url_hash"] not in eval_set]
    eval_pairs = [p for p in all_pairs if p["url_hash"] in eval_set]
    random.shuffle(train_pairs)
    random.shuffle(eval_pairs)

    print(f"Split: train={len(train_pairs)}, eval={len(eval_pairs)}")

    # Corruption type distribution
    type_dist = Counter(p["corruption_type"] for p in all_pairs)
    print(f"\nCorruption type distribution:")
    for ctype, count in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {ctype:25s}: {count}")

    # Write
    train_path = out_dir / "dpo_format_train_cot.jsonl"
    eval_path = out_dir / "dpo_format_eval_cot.jsonl"

    for pairs, path in [(train_pairs, train_path), (eval_pairs, eval_path)]:
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote {len(pairs)} -> {path}")

    # Stats
    stats = {
        "total_pairs": len(all_pairs),
        "train": len(train_pairs),
        "eval": len(eval_pairs),
        "skipped_incomplete": skipped,
        "reward_violations": reward_violations,
        "corruption_type_dist": dict(sorted(type_dist.items())),
    }
    stats_path = out_dir / "dataset_stats_dpo_format.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Stats -> {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
