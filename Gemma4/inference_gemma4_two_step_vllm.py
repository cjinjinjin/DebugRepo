"""
Gemma 4 two-step inference via vLLM offline engine.

Improves diversity by splitting generation into two steps:
  Step 1: Generate 5 diverse scene concepts (one phrase each)
  Step 2: Expand each scene into a full 30-50 word prompt

All N records are batched per step for maximum throughput via vLLM
continuous batching.

Usage:
  python Gemma4/inference_gemma4_two_step_vllm.py \
      --model_id /vc_data/.../gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_samples 2 \
      --temperature 1.0 \
      --tensor_parallel_size 2 \
      --output_file Gemma4/results/gemma4_two_step_vllm_test.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# System prompts (same as inference_gemma4_two_step.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_STEP1_SCENES = """You are an expert Ad Creative Director specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, generate 5 DIVERSE scene concepts for Native Ad images.

Each scene must use a DIFFERENT visual approach:
1. One close-up product/detail shot
2. One lifestyle scene with a person using/experiencing the product
3. One environmental/contextual setting showing the product in its natural habitat
4. One outcome/result-focused scene showing the benefit
5. One mood/atmosphere-driven composition

Each scene description should be a SHORT phrase (5-10 words) that captures the setting, subject, and mood.

Output exactly 5 scene descriptions (no reasoning, no thinking):
<Scene1>...</Scene1>
<Scene2>...</Scene2>
<Scene3>...</Scene3>
<Scene4>...</Scene4>
<Scene5>...</Scene5>"""

SYSTEM_PROMPT_STEP2_EXPAND = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page and a scene concept, expand the scene into a detailed image generation prompt for a Native Ad.

The prompt must:
- Be 30–50 words
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

Output exactly one prompt (no reasoning, no thinking):
<Prompt>...</Prompt>"""


# ---------------------------------------------------------------------------
# Utilities (same as single-prompt vLLM script)
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_user_content_from_messages(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def extract_lp_fields_from_messages(messages: list[dict]) -> dict:
    FIELD_LABELS = {
        "FinalDestinationURLUrl": "URL",
        "PrimaryContentNoTitleNoHeading": "Primary Content",
    }
    lp_fields = {}
    for msg in messages:
        if msg.get("role") == "user":
            content = msg["content"]
            for key, label in FIELD_LABELS.items():
                pattern = rf"- {re.escape(label)}: (.+?)(?:\n- |\Z)"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    lp_fields[key] = match.group(1).strip()
                    continue
                pattern2 = rf"\[{re.escape(label)}\]\n(.+?)(?:\n\[|\Z)"
                match2 = re.search(pattern2, content, re.DOTALL)
                if match2:
                    lp_fields[key] = match2.group(1).strip()
    return lp_fields


def truncate_lp_content(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + " ..."


def truncate_user_content(content: str, max_chars: int) -> str:
    if max_chars <= 0:
        return content
    content_labels = ["Primary Content", "Page Content", "PrimaryContentNoTitleNoHeading"]
    for label in content_labels:
        for pattern in [
            rf"(\[{re.escape(label)}\]\n)(.*?)(\n\[|\Z)",
            rf"(- {re.escape(label)}: )(.*?)(\n- |\Z)",
        ]:
            m = re.search(pattern, content, re.DOTALL)
            if m and len(m.group(2)) > max_chars:
                truncated = truncate_lp_content(m.group(2), max_chars)
                return content[:m.start(2)] + truncated + content[m.start(3):]
    return content


def build_user_message(lp_fields: dict, max_lp_chars: int = 0) -> str:
    FIELD_LABELS = {
        "FinalDestinationURLUrl": "URL",
        "PrimaryContentNoTitleNoHeading": "Primary Content",
    }
    lines = ["Generate 5 diverse scene concepts for the following landing page:\n"]
    for key, label in FIELD_LABELS.items():
        val = lp_fields.get(key, "").strip()
        if val:
            if max_lp_chars > 0 and key == "PrimaryContentNoTitleNoHeading":
                val = truncate_lp_content(val, max_lp_chars)
            lines.append(f"- {label}: {val}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers (same as inference_gemma4_two_step.py)
# ---------------------------------------------------------------------------

def parse_scenes(text: str) -> list[str]:
    """Extract <Scene1>...</Scene5> tags from Step 1 output."""
    scenes = []
    for i in range(1, 6):
        m = re.search(rf"<Scene{i}>(.*?)</Scene{i}>", text, re.DOTALL)
        if m:
            scenes.append(m.group(1).strip())
    return scenes


def parse_single_prompt(text: str) -> str:
    """Extract <Prompt>...</Prompt> tag from Step 2 output."""
    m = re.search(r"<Prompt>(.*?)</Prompt>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Gemma 4 two-step inference via vLLM (offline)"
    )
    p.add_argument("--model_id",
                    default="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it")
    p.add_argument("--max_lp_chars", type=int, default=0,
                    help="Truncate Primary Content to N chars (0=no truncation)")
    # IO
    p.add_argument("--input_file", required=True, help="Input JSONL")
    p.add_argument("--output_file",
                    default="Gemma4/results/gemma4_two_step_vllm_output.jsonl")
    p.add_argument("--num_samples", type=int, default=0,
                    help="Limit number of input samples (0=all)")
    # Generation params
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=64)
    # vLLM engine params
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--dtype", type=str, default="auto",
                    help="Model dtype (auto, half, bfloat16). Use 'half' for GPTQ models.")
    p.add_argument("--kv_cache_dtype", type=str, default="auto",
                    help="KV cache dtype (auto, fp8). fp8 reduces memory and may improve throughput.")
    p.add_argument("--enable_prefix_caching", action="store_true", default=False,
                    help="Enable prefix caching for shared system prompts")
    return p.parse_args()


def main():
    args = parse_args()

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load vLLM engine
    print(f"Loading vLLM engine: {args.model_id}")
    print(f"  tensor_parallel_size={args.tensor_parallel_size}")
    print(f"  gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"  max_model_len={args.max_model_len}")

    llm_kwargs = dict(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=True,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    if args.kv_cache_dtype != "auto":
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    print(f"  kv_cache_dtype={args.kv_cache_dtype}")
    print(f"  enable_prefix_caching={args.enable_prefix_caching}")

    llm = LLM(**llm_kwargs)

    # Load data
    records = load_jsonl(args.input_file)
    if args.num_samples and args.num_samples < len(records):
        records = records[:args.num_samples]

    # Prepare user content for each record
    user_contents = []
    input_type = "lp_fields"
    for r in records:
        if "lp_fields" in r:
            user_contents.append(build_user_message(r["lp_fields"], args.max_lp_chars))
        elif "messages" in r:
            uc = extract_user_content_from_messages(r["messages"])
            if uc:
                if args.max_lp_chars > 0:
                    uc = truncate_user_content(uc, args.max_lp_chars)
                user_contents.append(uc)
                input_type = "user_content"
            else:
                lp = extract_lp_fields_from_messages(r["messages"])
                user_contents.append(build_user_message(lp, args.max_lp_chars))
        else:
            user_contents.append(build_user_message(r, args.max_lp_chars))

    total = len(records)
    print(f"\nLoaded {total} records from {args.input_file}")
    print(f"Input type: {input_type}")
    print(f"Mode: two-step (scene planning + expansion) via vLLM")
    print(f"Temperature: {args.temperature}")

    # =====================================================================
    # TTFT measurement: generate 1 token per prompt to measure prefill
    # =====================================================================
    step1_prompts_for_ttft = []
    for uc in user_contents:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STEP1_SCENES},
            {"role": "user", "content": uc},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        step1_prompts_for_ttft.append(chat_text)

    print(f"\nMeasuring TTFT (prefill) with max_tokens=1...")
    ttft_params = SamplingParams(max_tokens=1, temperature=0)
    ttft_start = time.time()
    llm.generate(step1_prompts_for_ttft, ttft_params)
    ttft_elapsed = time.time() - ttft_start
    ttft_per_prompt = ttft_elapsed / len(step1_prompts_for_ttft)
    print(f"TTFT measurement done: {ttft_elapsed:.2f}s total, {ttft_per_prompt:.3f}s/prompt ({len(step1_prompts_for_ttft)} prompts)\n")

    # =====================================================================
    # Step 1: Generate 5 scene concepts — batch all N records
    # =====================================================================
    print(f"{'='*60}")
    print(f"Step 1: Generating 5 scene concepts for {total} records...")
    print(f"{'='*60}")

    step1_prompts = step1_prompts_for_ttft  # reuse already-built prompts

    step1_params = SamplingParams(
        max_tokens=120,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=["</Scene5>"],
    )

    step1_start = time.time()
    step1_outputs = llm.generate(step1_prompts, step1_params)
    step1_elapsed = time.time() - step1_start

    # Collect Step 1 token stats
    step1_total_in = 0
    step1_total_out = 0

    # Parse scenes from each record
    all_scenes = []  # list of list[str], one per record
    step1_raw_texts = []
    n_full_scenes = 0
    for i, output in enumerate(step1_outputs):
        raw = output.outputs[0].text
        # Append the stop string back for parsing
        if not raw.rstrip().endswith("</Scene5>"):
            raw = raw + "</Scene5>"
        step1_raw_texts.append(raw)

        n_in = len(output.prompt_token_ids) if output.prompt_token_ids else 0
        n_out = len(output.outputs[0].token_ids) if output.outputs else 0
        step1_total_in += n_in
        step1_total_out += n_out

        scenes = parse_scenes(raw)
        all_scenes.append(scenes)
        status = f"{len(scenes)} scenes" if len(scenes) == 5 else f"WARNING: {len(scenes)} scenes"
        if len(scenes) == 5:
            n_full_scenes += 1
        print(f"  [{i+1}/{total}] {status}: {scenes[:3]}...")

    print(f"\nStep 1 done: {step1_elapsed:.1f}s | {n_full_scenes}/{total} records with 5 scenes")
    print(f"  Input tokens: {step1_total_in} | Output tokens: {step1_total_out}")

    # =====================================================================
    # Step 2: Expand each scene — batch all N×5 prompts
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"Step 2: Expanding scenes into prompts...")
    print(f"{'='*60}")

    step2_prompts = []
    step2_prompt_map = []  # (record_idx, scene_idx) for each prompt

    for rec_idx, (uc, scenes) in enumerate(zip(user_contents, all_scenes)):
        # Base content for expansion
        if args.max_lp_chars > 0:
            base_content = truncate_user_content(uc, args.max_lp_chars) if input_type == "user_content" else uc
        else:
            base_content = uc

        for scene_idx, scene in enumerate(scenes):
            scene_content = (
                f"{base_content}\n\n"
                f"Expand this scene concept into a detailed prompt:\n"
                f"<Scene>{scene}</Scene>"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_STEP2_EXPAND},
                {"role": "user", "content": scene_content},
            ]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            step2_prompts.append(chat_text)
            step2_prompt_map.append((rec_idx, scene_idx))

    print(f"Total Step 2 prompts: {len(step2_prompts)} ({total} records x scenes)")

    step2_params = SamplingParams(
        max_tokens=128,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=["</Prompt>"],
    )

    step2_start = time.time()
    step2_outputs = llm.generate(step2_prompts, step2_params)
    step2_elapsed = time.time() - step2_start

    # Collect Step 2 token stats
    step2_total_in = 0
    step2_total_out = 0

    # Group expanded prompts by record
    expanded_by_record = {}  # rec_idx -> list of (scene_idx, prompt_text, raw_text)
    for i, (output, (rec_idx, scene_idx)) in enumerate(zip(step2_outputs, step2_prompt_map)):
        raw = output.outputs[0].text
        # Append stop string for parsing
        if not raw.rstrip().endswith("</Prompt>"):
            raw = raw + "</Prompt>"

        n_in = len(output.prompt_token_ids) if output.prompt_token_ids else 0
        n_out = len(output.outputs[0].token_ids) if output.outputs else 0
        step2_total_in += n_in
        step2_total_out += n_out

        prompt = parse_single_prompt(raw)
        if rec_idx not in expanded_by_record:
            expanded_by_record[rec_idx] = []
        expanded_by_record[rec_idx].append((scene_idx, prompt, raw))

    # Sort by scene_idx within each record
    for rec_idx in expanded_by_record:
        expanded_by_record[rec_idx].sort(key=lambda x: x[0])

    print(f"\nStep 2 done: {step2_elapsed:.1f}s")
    print(f"  Input tokens: {step2_total_in} | Output tokens: {step2_total_out}")

    # =====================================================================
    # Combine results and write output
    # =====================================================================
    total_elapsed = step1_elapsed + step2_elapsed
    total_in_tokens = step1_total_in + step2_total_in
    total_out_tokens = step1_total_out + step2_total_out

    format_regex = re.compile(
        r"<Prompt1>[\s\S]+?</Prompt1>\s*"
        r"<Prompt2>[\s\S]+?</Prompt2>\s*"
        r"<Prompt3>[\s\S]+?</Prompt3>\s*"
        r"<Prompt4>[\s\S]+?</Prompt4>\s*"
        r"<Prompt5>[\s\S]+?</Prompt5>"
    )

    n_compliant = 0
    n_all_parsed = 0

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_file, "w", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")

    for rec_idx, record in enumerate(records):
        scenes = all_scenes[rec_idx]
        expanded = expanded_by_record.get(rec_idx, [])
        prompts = [p for _, p, _ in expanded]

        # Combine into <Prompt1>...<Prompt5> format
        formatted_parts = []
        for i, p in enumerate(prompts, 1):
            formatted_parts.append(f"<Prompt{i}>{p}</Prompt{i}>")
        combined_raw = "\n\n".join(formatted_parts)

        compliant = bool(format_regex.search(combined_raw)) if len(prompts) >= 5 else False
        all_parsed = all(p and len(p.split()) > 5 for p in prompts) and len(prompts) == 5

        if compliant:
            n_compliant += 1
        if all_parsed:
            n_all_parsed += 1

        print(f"  [{rec_idx+1}/{total}] id={record.get('id', '')} | "
              f"{len(prompts)} prompts | compliant={compliant}")
        for i, p in enumerate(prompts):
            print(f"    Prompt {i+1} ({len(p.split())} words): {p[:80]}...")

        result = {
            "id": record.get("id", ""),
            "generated_prompts": prompts,
            "raw_output": combined_raw,
            "scenes": scenes,
            "step1_raw": step1_raw_texts[rec_idx],
            "individual_raw_outputs": [r for _, _, r in expanded],
            "format_compliant": compliant,
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()

    # =====================================================================
    # Summary
    # =====================================================================
    decode_tok_per_s = total_out_tokens / total_elapsed if total_elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"Two-Step vLLM Inference Summary")
    print(f"{'='*60}")
    print(f"Records:             {total}")
    print(f"Full scenes (5/5):   {n_full_scenes}/{total} ({100*n_full_scenes/total:.1f}%)")
    print(f"Format compliance:   {n_compliant}/{total} ({100*n_compliant/total:.1f}%)")
    print(f"All prompts parsed:  {n_all_parsed}/{total} ({100*n_all_parsed/total:.1f}%)")
    print(f"{'='*60}")
    print(f"Timing:")
    print(f"  TTFT (prefill):    {ttft_per_prompt:.3f}s/prompt")
    print(f"  Step 1 (scenes):   {step1_elapsed:.1f}s ({step1_elapsed/total:.2f}s/sample)")
    print(f"  Step 2 (expand):   {step2_elapsed:.1f}s ({step2_elapsed/len(step2_prompts):.2f}s/prompt)")
    print(f"  Total inference:   {total_elapsed:.1f}s ({total_elapsed/total:.1f}s/sample)")
    print(f"{'='*60}")
    print(f"Tokens:")
    print(f"  Step 1 input:      {step1_total_in}  output: {step1_total_out}")
    print(f"  Step 2 input:      {step2_total_in}  output: {step2_total_out}")
    print(f"  Total input:       {total_in_tokens}  output: {total_out_tokens}")
    print(f"  Decode throughput: {decode_tok_per_s:.1f} tok/s")
    print(f"{'='*60}")
    print(f"Saved {total} results -> {args.output_file}")


if __name__ == "__main__":
    main()
