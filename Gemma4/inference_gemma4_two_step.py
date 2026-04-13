"""
Gemma 4 two-step inference: scene planning + prompt expansion.

Improves diversity of 5 generated image prompts by splitting into two steps:
  Step 1: Generate 5 diverse scene concepts (one sentence each)
  Step 2: Expand each scene into a full 80-150 word prompt

Reuses model loading and utilities from inference_gemma4.py.

Usage:
  python Gemma4/inference_gemma4_two_step.py \
      --model_id /vc_data/.../gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/dpo_combined_eval_cot.jsonl \
      --num_samples 5 \
      --no_think \
      --output_file Gemma4/results/gemma4_two_step_test.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from inference_gemma4 import (
    Gemma4PromptGenerator,
    build_user_message,
    extract_lp_fields_from_messages,
    extract_user_content_from_messages,
    load_jsonl,
    parse_gemma_response,
    parse_output_prompts,
    truncate_user_content,
)

# ---------------------------------------------------------------------------
# Two-step system prompts
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
# Parsing helpers
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
# Two-step generation
# ---------------------------------------------------------------------------

def generate_two_step(
    gen: Gemma4PromptGenerator,
    user_content: str = None,
    lp_fields: dict = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    do_sample: bool = True,
) -> dict:
    """Two-step generation: plan 5 diverse scenes, then expand each.

    Returns dict with 'content' (formatted with <Prompt1>-<Prompt5> tags)
    and 'thinking' (empty).
    """
    orig_system_prompt = gen.system_prompt
    gen_kwargs = dict(temperature=temperature, top_p=top_p, top_k=top_k, do_sample=do_sample)

    # --- Step 1: Generate 5 scene concepts ---
    print("\n[TWO-STEP] Step 1: Generating 5 scene concepts...")
    gen.system_prompt = SYSTEM_PROMPT_STEP1_SCENES
    step1_response = gen.generate(
        user_content=user_content,
        lp_fields=lp_fields,
        max_new_tokens=128,
        **gen_kwargs,
    )
    step1_text = step1_response.get("content", "") if isinstance(step1_response, dict) else str(step1_response)
    scenes = parse_scenes(step1_text)
    print(f"[TWO-STEP] Parsed {len(scenes)} scenes")
    for i, s in enumerate(scenes, 1):
        print(f"  Scene {i}: {s}")

    if len(scenes) < 5:
        print(f"[TWO-STEP] WARNING: Only {len(scenes)} scenes parsed, expected 5. Raw output:")
        print(f"  {step1_text[:500]}")
        gen.system_prompt = orig_system_prompt
        return {"thinking": "", "content": step1_text}

    # --- Step 2: Batch-expand all 5 scenes in parallel ---
    print("\n[TWO-STEP] Step 2: Batch-expanding 5 scenes in parallel...")
    gen.system_prompt = SYSTEM_PROMPT_STEP2_EXPAND

    if user_content:
        base_content = user_content
        if gen.max_lp_chars > 0:
            base_content = truncate_user_content(base_content, gen.max_lp_chars)
    elif lp_fields:
        base_content = build_user_message(lp_fields, gen.max_lp_chars)
    else:
        base_content = ""

    # Build 5 input texts
    input_texts = []
    for scene in scenes:
        scene_content = (
            f"{base_content}\n\n"
            f"Expand this scene concept into a detailed prompt:\n"
            f"<Scene>{scene}</Scene>"
        )
        input_texts.append(gen.build_input_from_content(scene_content))

    # Tokenize with left-padding for batch generation
    # Gemma4Processor wraps a tokenizer; use the inner tokenizer for batch text encoding
    if hasattr(gen.processor, "tokenizer"):
        tokenizer = gen.processor.tokenizer
    else:
        tokenizer = gen.processor
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    batch_inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
    ).to(gen.model.device)
    input_lens = (batch_inputs["attention_mask"]).sum(dim=1).tolist()

    print(f"  Batch size: {len(input_texts)}, max input len: {batch_inputs['input_ids'].shape[1]}")

    with torch.inference_mode():
        outputs = gen.model.generate(
            **batch_inputs,
            max_new_tokens=256,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
        )

    tokenizer.padding_side = orig_padding_side

    # Decode each sequence
    expanded_prompts = []
    for i in range(len(scenes)):
        # Skip padding + input tokens
        seq_len = outputs[i].shape[0]
        pad_len = batch_inputs["input_ids"].shape[1] - input_lens[i]
        gen_start = pad_len + input_lens[i]
        gen_tokens = outputs[i][gen_start:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=False)

        # parse_response lives on the processor, not the inner tokenizer
        if hasattr(gen.processor, "parse_response"):
            parsed = gen.processor.parse_response(response)
        else:
            parsed = gen._parse_response_fallback(response)

        text = parsed.get("content", "") if isinstance(parsed, dict) else str(parsed)
        prompt = parse_single_prompt(text)
        expanded_prompts.append(prompt)
        print(f"  Prompt {i+1} ({len(prompt.split())} words, {len(gen_tokens)} tokens): {prompt[:100]}...")

    gen.system_prompt = orig_system_prompt

    # Format output with <Prompt1>-<Prompt5> tags
    formatted_parts = []
    for i, p in enumerate(expanded_prompts, 1):
        formatted_parts.append(f"<Prompt{i}>{p}</Prompt{i}>")
    combined_content = "\n".join(formatted_parts)

    return {"thinking": "", "content": combined_content}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Gemma 4 two-step inference: scene planning + prompt expansion"
    )
    p.add_argument("--model_id",
                    default="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it")
    p.add_argument("--processor_id", default="")
    p.add_argument("--adapter_path", default="")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--use_gptq", action="store_true", default=False)
    p.add_argument("--no_think", action="store_true", default=False,
                    help="Disable thinking mode")
    p.add_argument("--max_lp_chars", type=int, default=0,
                    help="Truncate Primary Content to N chars (0=no truncation)")
    p.add_argument("--attn_impl", type=str, default="")
    # IO
    p.add_argument("--input_file", required=True, help="Input JSONL")
    p.add_argument("--output_file", default="Gemma4/results/gemma4_two_step_output.jsonl")
    p.add_argument("--num_samples", type=int, default=0,
                    help="Limit number of samples (0=all)")
    # Generation params
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--do_sample", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    gen = Gemma4PromptGenerator(
        model_id=args.model_id,
        processor_id=args.processor_id or None,
        adapter_path=args.adapter_path or None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_gptq=args.use_gptq,
        no_cot=True,  # two-step always uses no-CoT base prompt (scenes replace it)
        max_lp_chars=args.max_lp_chars,
        enable_thinking=not args.no_think,
        attn_impl=args.attn_impl,
    )

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
    }

    records = load_jsonl(args.input_file)
    if args.num_samples and args.num_samples < len(records):
        records = records[:args.num_samples]

    batch_inputs = []
    input_type = "lp_fields"
    for r in records:
        if "lp_fields" in r:
            batch_inputs.append(r["lp_fields"])
        elif "messages" in r:
            user_content = extract_user_content_from_messages(r["messages"])
            if user_content:
                batch_inputs.append(user_content)
                input_type = "user_content"
            else:
                batch_inputs.append(extract_lp_fields_from_messages(r["messages"]))
        else:
            batch_inputs.append(r)

    print(f"Loaded {len(records)} records from {args.input_file}")
    print(f"Input type: {input_type}")
    print(f"Mode: two-step (scene planning + expansion)")
    start_time = time.time()

    # Format compliance regex (no-think)
    format_regex = re.compile(
        r"<Prompt1>[\s\S]+?</Prompt1>\s*"
        r"<Prompt2>[\s\S]+?</Prompt2>\s*"
        r"<Prompt3>[\s\S]+?</Prompt3>\s*"
        r"<Prompt4>[\s\S]+?</Prompt4>\s*"
        r"<Prompt5>[\s\S]+?</Prompt5>"
    )

    n_compliant = 0
    n_tags = 0
    total = len(records)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_file, "w", encoding="utf-8")

    for idx, (record, inp) in enumerate(zip(records, batch_inputs)):
        if input_type == "user_content":
            raw_response = generate_two_step(gen, user_content=inp, **gen_kwargs)
        else:
            raw_response = generate_two_step(gen, lp_fields=inp, **gen_kwargs)

        raw = parse_gemma_response(raw_response)
        prompts = parse_output_prompts(raw)

        compliant = bool(format_regex.search(raw))
        has_all_tags = len(prompts) == 5 and prompts[0] != raw.strip()

        if compliant:
            n_compliant += 1
        if has_all_tags:
            n_tags += 1

        result = {
            "id": record.get("id", ""),
            "generated_prompts": prompts,
            "raw_output": raw,
            "format_compliant": compliant,
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

        done = idx + 1
        print(f"  Processed {done}/{total} | compliant: {n_compliant}/{done}")

    out_f.close()

    elapsed = time.time() - start_time
    print(f"\nTotal inference time: {elapsed:.1f}s ({elapsed/total:.1f}s/sample)")
    print(f"Saved {total} results -> {args.output_file}")

    print(f"\n{'='*60}")
    print(f"Format compliance: {n_compliant}/{total} ({100*n_compliant/total:.1f}%)")
    print(f"All 5 tags present: {n_tags}/{total} ({100*n_tags/total:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
