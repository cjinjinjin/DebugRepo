"""
Gemma 4 26B-A4B-it zero-shot inference for LP → image prompt generation.

Directly uses the same system prompt and eval data from Qwen3 experiments,
no fine-tuning required.

Usage:
  # Single query
  python Gemma4/inference_gemma4.py \
      --model_id /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it \
      --url "https://example.com/product" \
      --title "Product Title"

  # Batch inference from JSONL
  python Gemma4/inference_gemma4.py \
      --model_id /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/sft_eval_cot.jsonl \
      --output_file Gemma4/results/gemma4_zeroshot_eval.jsonl \
      --max_new_tokens 2048 --batch_size 1

  # With LoRA adapter (after SFT)
  python Gemma4/inference_gemma4.py \
      --model_id /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it \
      --adapter_path path/to/lora_adapter \
      --input_file QwenFinetune/data/sft_eval_cot.jsonl \
      --output_file Gemma4/results/gemma4_sft_eval.jsonl
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# System prompt — identical to the one used in Qwen3 training/inference
SYSTEM_PROMPT = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, your task is to generate five (5) high-quality English image generation prompts for Native Ads.

Each prompt must:
- Be ≤150 words
- Embed all safety, realism, quality, and exclusion constraints
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

You must first reason about the product in a <think> block containing these fields:
- ProductType: what kind of product/service
- SpecificProduct: the exact product name
- Category: product category
- VisualAnchors: key visual elements to include
- LifestyleVibe: mood and lifestyle context
- CoreValueSignals: core value propositions

Then output exactly 5 prompts in this format:
<think>
ProductType: ...
SpecificProduct: ...
Category: ...
VisualAnchors: ...
LifestyleVibe: ...
CoreValueSignals: ...
</think>
<Prompt1>...</Prompt1>
<Prompt2>...</Prompt2>
<Prompt3>...</Prompt3>
<Prompt4>...</Prompt4>
<Prompt5>...</Prompt5>"""


# ---------------------------------------------------------------------------
# Field mapping (same as Qwen3 inference.py)
# ---------------------------------------------------------------------------

FIELD_LABELS = {
    "FinalDestinationURLUrl": "URL",
    "DocumentTitle": "Document Title",
    "VisualTitle": "Visual Title",
    "Heading": "Heading",
    "Title_CB": "Title (CB)",
    "VisualTitle_CB": "Visual Title (CB)",
    "Heading_CB": "Heading (CB)",
    "BestSnippet_CB": "Best Snippet (CB)",
    "MetaDescription_CB": "Meta Description (CB)",
    "PrimaryContentNoTitleNoHeading": "Primary Content",
}


def build_user_message(lp_fields: dict) -> str:
    """Build user message from LP fields."""
    lines = ["Generate 5 image prompts for the following landing page:\n"]
    for key, label in FIELD_LABELS.items():
        val = lp_fields.get(key, "").strip()
        if val:
            lines.append(f"- {label}: {val}")
    return "\n".join(lines)


def extract_lp_fields_from_messages(messages: list[dict]) -> dict:
    """Extract LP fields from SFT-format messages (user content with bracket labels)."""
    lp_fields = {}
    for msg in messages:
        if msg.get("role") == "user":
            content = msg["content"]
            # Parse "- Label: value" or "[Label]\nvalue" format
            for key, label in FIELD_LABELS.items():
                # Try "- Label: value" format
                pattern = rf"- {re.escape(label)}: (.+?)(?:\n- |\Z)"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    lp_fields[key] = match.group(1).strip()
                    continue
                # Try "[Label]\nvalue" format (swift infer output)
                pattern2 = rf"\[{re.escape(label)}\]\n(.+?)(?:\n\[|\Z)"
                match2 = re.search(pattern2, content, re.DOTALL)
                if match2:
                    lp_fields[key] = match2.group(1).strip()
    return lp_fields


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class Gemma4PromptGenerator:
    """Wraps Gemma 4 for LP → image prompt generation."""

    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype=torch.bfloat16,
        enable_thinking: bool = True,
    ):
        self.model_id = model_id
        self.enable_thinking = enable_thinking

        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"Loading processor from {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Quantization config
        bnb_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        print(f"Loading model from {model_id} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch_dtype,
        )

        # Load LoRA adapter if provided
        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            print(f"Loading LoRA adapter from {adapter_path} ...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        print("Model ready.")

    def build_input(self, lp_fields: dict) -> str:
        """Build chat-template formatted input string."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(lp_fields)},
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    @torch.inference_mode()
    def generate(
        self,
        lp_fields: dict,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        do_sample: bool = True,
    ) -> str:
        """Generate image prompts for a single LP. Returns raw output string."""
        input_text = self.build_input(lp_fields)

        inputs = self.processor(
            text=input_text,
            return_tensors="pt",
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
        )

        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )

        # Parse thinking vs final answer
        parsed = self.processor.parse_response(response)

        return parsed

    def generate_batch(
        self,
        lp_fields_list: list[dict],
        batch_size: int = 1,
        **gen_kwargs,
    ) -> list[str]:
        """Batch inference. Returns list of raw output strings."""
        results = []
        for i in range(0, len(lp_fields_list), batch_size):
            batch = lp_fields_list[i: i + batch_size]
            for lp_fields in batch:
                result = self.generate(lp_fields, **gen_kwargs)
                results.append(result)
            done = min(i + batch_size, len(lp_fields_list))
            print(f"  Processed {done}/{len(lp_fields_list)}")
        return results


# ---------------------------------------------------------------------------
# Output parsing (compatible with evaluate.py)
# ---------------------------------------------------------------------------

def parse_gemma_response(response) -> str:
    """Convert Gemma parse_response output to raw text compatible with evaluate.py.

    Gemma's thinking block uses <|channel>thought ... <channel|> format.
    We convert it to <think>...</think> for compatibility with the existing
    evaluate.py that expects Qwen-style output.
    """
    if isinstance(response, dict):
        # parse_response returns dict with 'thought' and 'response' keys
        thinking = response.get("thought", "")
        answer = response.get("response", "")
        if thinking:
            return f"<think>\n{thinking}\n</think>\n{answer}"
        return answer
    return str(response)


def parse_output_prompts(text: str) -> list[str]:
    """Extract <Prompt1>...</Prompt5> tags from model output."""
    prompts = []
    for i in range(1, 6):
        pattern = rf"<Prompt{i}>(.*?)</Prompt{i}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            prompts.append(match.group(1).strip())
    if not prompts:
        prompts = [text.strip()]
    return prompts


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} results -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Gemma 4 zero-shot / SFT inference")
    p.add_argument("--model_id", default="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it",
                    help="HuggingFace model ID or local path")
    p.add_argument("--adapter_path", default="",
                    help="Optional LoRA adapter path (for SFT/DPO checkpoints)")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--no_think", action="store_true", default=False,
                    help="Disable thinking mode")
    # Single query
    p.add_argument("--url", default="", help="Landing page URL")
    p.add_argument("--title", default="", help="Document title")
    p.add_argument("--heading", default="", help="Heading text")
    p.add_argument("--content", default="", help="Primary content text")
    # Batch mode
    p.add_argument("--input_file", default="", help="Input JSONL")
    p.add_argument("--output_file", default="Gemma4/results/gemma4_output.jsonl")
    p.add_argument("--batch_size", type=int, default=1)
    # Generation params (Gemma 4 recommended defaults)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--do_sample", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    gen = Gemma4PromptGenerator(
        model_id=args.model_id,
        adapter_path=args.adapter_path or None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        enable_thinking=not args.no_think,
    )

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
    }

    if args.input_file:
        # Batch mode
        records = load_jsonl(args.input_file)
        lp_fields_list = []
        for r in records:
            if "lp_fields" in r:
                lp_fields_list.append(r["lp_fields"])
            elif "messages" in r:
                lp_fields_list.append(extract_lp_fields_from_messages(r["messages"]))
            else:
                lp_fields_list.append(r)

        print(f"Loaded {len(records)} records from {args.input_file}")
        start_time = time.time()

        raw_outputs = gen.generate_batch(lp_fields_list, batch_size=args.batch_size, **gen_kwargs)

        elapsed = time.time() - start_time
        print(f"Total inference time: {elapsed:.1f}s ({elapsed/len(records):.1f}s/sample)")

        # Format compliance check
        format_regex = re.compile(
            r"<think>[\s\S]+?</think>\s*"
            r"<Prompt1>[\s\S]+?</Prompt1>\s*"
            r"<Prompt2>[\s\S]+?</Prompt2>\s*"
            r"<Prompt3>[\s\S]+?</Prompt3>\s*"
            r"<Prompt4>[\s\S]+?</Prompt4>\s*"
            r"<Prompt5>[\s\S]+?</Prompt5>"
        )
        format_regex_no_think = re.compile(
            r"<Prompt1>[\s\S]+?</Prompt1>\s*"
            r"<Prompt2>[\s\S]+?</Prompt2>\s*"
            r"<Prompt3>[\s\S]+?</Prompt3>\s*"
            r"<Prompt4>[\s\S]+?</Prompt4>\s*"
            r"<Prompt5>[\s\S]+?</Prompt5>"
        )

        n_compliant = 0
        n_tags = 0
        results = []

        for record, raw_response in zip(records, raw_outputs):
            raw = parse_gemma_response(raw_response)
            prompts = parse_output_prompts(raw)

            compliant = bool(format_regex.search(raw) or format_regex_no_think.search(raw))
            has_all_tags = len(prompts) == 5 and prompts[0] != raw.strip()

            if compliant:
                n_compliant += 1
            if has_all_tags:
                n_tags += 1

            results.append({
                "id": record.get("id", ""),
                "lp_fields": lp_fields_list[records.index(record)] if records.index(record) < len(lp_fields_list) else {},
                "generated_prompts": prompts,
                "raw_output": raw,
                "format_compliant": compliant,
            })

        write_jsonl(results, args.output_file)

        total = len(results)
        print(f"\n{'='*60}")
        print(f"Format compliance (full): {n_compliant}/{total} ({100*n_compliant/total:.1f}%)")
        print(f"All 5 tags present:       {n_tags}/{total} ({100*n_tags/total:.1f}%)")
        print(f"{'='*60}")

    else:
        # Single query mode
        lp_fields = {
            "FinalDestinationURLUrl": args.url,
            "DocumentTitle": args.title,
            "Heading": args.heading,
            "PrimaryContentNoTitleNoHeading": args.content,
        }
        raw_response = gen.generate(lp_fields, **gen_kwargs)
        raw = parse_gemma_response(raw_response)
        prompts = parse_output_prompts(raw)

        print("\n" + "=" * 60)
        print("Raw output:")
        print("=" * 60)
        print(raw)
        print("\n" + "=" * 60)
        print("Parsed Prompts:")
        print("=" * 60)
        for i, p in enumerate(prompts, 1):
            print(f"\n[Prompt {i}]\n{p}")


if __name__ == "__main__":
    main()
