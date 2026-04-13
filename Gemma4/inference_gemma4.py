"""
Gemma 4 26B-A4B-it zero-shot inference for LP → image prompt generation.

Directly uses the same system prompt and eval data from Qwen3 experiments,
no fine-tuning required.

Usage:
  # Single query (only URL + content, matching production fields)
  python Gemma4/inference_gemma4.py \
      --model_id /vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it \
      --url "https://example.com/product" \
      --content "Product description text"

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
- Be 80–150 words
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

SYSTEM_PROMPT_NO_COT = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, your task is to generate five (5) high-quality English image generation prompts for Native Ads.

Each prompt must:
- Be 80–150 words
- Embed all safety, realism, quality, and exclusion constraints
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

Output exactly 5 prompts in this format (no reasoning, no thinking, just the prompts):
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
    "PrimaryContentNoTitleNoHeading": "Primary Content",
}


def truncate_lp_content(text: str, max_chars: int) -> str:
    """Truncate LP content to max_chars, cutting at last word boundary."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + " ..."


def build_user_message(lp_fields: dict, max_lp_chars: int = 0) -> str:
    """Build user message from LP fields."""
    lines = ["Generate 5 image prompts for the following landing page:\n"]
    for key, label in FIELD_LABELS.items():
        val = lp_fields.get(key, "").strip()
        if val:
            if max_lp_chars > 0 and key == "PrimaryContentNoTitleNoHeading":
                val = truncate_lp_content(val, max_lp_chars)
            lines.append(f"- {label}: {val}")
    return "\n".join(lines)


def truncate_user_content(content: str, max_chars: int) -> str:
    """Truncate the main content section within a raw user message string.
    Handles various label formats: Primary Content, Page Content, etc."""
    if max_chars <= 0:
        return content
    # Match the longest content section in various formats
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


def extract_user_content_from_messages(messages: list[dict]) -> str:
    """Extract user message content from SFT/DPO-format messages."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    return ""


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
        processor_id: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_gptq: bool = False,
        no_cot: bool = False,
        max_lp_chars: int = 0,
        torch_dtype=torch.bfloat16,
        enable_thinking: bool = True,
        attn_impl: str = "",
    ):
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.system_prompt = SYSTEM_PROMPT_NO_COT if no_cot else SYSTEM_PROMPT
        self.max_lp_chars = max_lp_chars

        from transformers import AutoModelForCausalLM

        proc_id = processor_id or model_id
        print(f"Loading processor from {proc_id} ...")
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(proc_id)
        except (ValueError, ImportError, OSError, AttributeError):
            print(f"[WARN] AutoProcessor failed, falling back to AutoTokenizer with patched config ...")
            from transformers import AutoTokenizer
            import json as _json, shutil, tempfile
            _tok_cfg = Path(proc_id) / "tokenizer_config.json"
            _load_path = proc_id
            _tmpdir = None
            if _tok_cfg.exists():
                with open(_tok_cfg, "r") as _f:
                    _cfg = _json.load(_f)
                if isinstance(_cfg.get("extra_special_tokens"), list):
                    _tmpdir = tempfile.mkdtemp(prefix="tok_patch_")
                    # Copy all tokenizer files to temp dir
                    for f in Path(proc_id).iterdir():
                        if f.is_file() and ("token" in f.name.lower() or f.suffix == ".json" or f.suffix == ".model"):
                            shutil.copy2(f, _tmpdir)
                    _cfg["extra_special_tokens"] = {}
                    with open(Path(_tmpdir) / "tokenizer_config.json", "w") as _f:
                        _json.dump(_cfg, _f, indent=2, ensure_ascii=False)
                    _load_path = _tmpdir
            self.processor = AutoTokenizer.from_pretrained(_load_path)
            if _tmpdir:
                shutil.rmtree(_tmpdir, ignore_errors=True)

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
        if use_gptq:
            from gptqmodel import GPTQModel
            self.model = GPTQModel.load(model_id, device_map=device)
        else:
            # Workaround: transformers 5.x bug where config.quantization_config is None
            # for compressed-tensors format. Manually load and inject it.
            from transformers import AutoConfig
            _config = AutoConfig.from_pretrained(model_id)
            if getattr(_config, "quantization_config", None) is None:
                _qc_path = Path(model_id) / "config.json"
                if _qc_path.exists():
                    import json as _json
                    with open(_qc_path, "r") as _f:
                        _raw = _json.load(_f)
                    if "quantization_config" in _raw:
                        _config.quantization_config = _raw["quantization_config"]
                        print(f"[INFO] Manually injected quantization_config (quant_method: {_raw['quantization_config'].get('quant_method', 'unknown')})")

            load_kwargs = {
                "config": _config,
                "quantization_config": bnb_config,
                "device_map": device,
                "dtype": torch_dtype,
            }
            if attn_impl:
                load_kwargs["attn_implementation"] = attn_impl
                print(f"Attention implementation: {attn_impl}")

            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        # Load LoRA adapter if provided
        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            print(f"Loading LoRA adapter from {adapter_path} ...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        print("Model ready.")

    @staticmethod
    def _parse_response_fallback(response: str) -> dict:
        """Fallback parse_response for when AutoTokenizer is used (no parse_response method).
        Splits Gemma 4 thinking format: <think>...</think> content"""
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        if think_match:
            content = response[think_match.end():].strip()
        else:
            content = response.strip()
        # Remove EOS tokens
        for tok in ("<eos>", "<end_of_turn>", "</s>"):
            content = content.replace(tok, "").strip()
            thinking = thinking.replace(tok, "").strip()
        return {"thinking": thinking, "content": content}

    def build_input(self, lp_fields: dict) -> str:
        """Build chat-template formatted input string from LP field dict."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_user_message(lp_fields, self.max_lp_chars)},
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def build_input_from_content(self, user_content: str) -> str:
        """Build chat-template formatted input string from raw user message content."""
        if self.max_lp_chars > 0:
            user_content = truncate_user_content(user_content, self.max_lp_chars)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
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
        lp_fields: dict = None,
        user_content: str = None,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        do_sample: bool = True,
    ) -> str:
        """Generate image prompts for a single LP. Returns raw output string.

        Either lp_fields (dict) or user_content (str) must be provided.
        If user_content is provided, it is used directly as the user message.
        """
        if user_content:
            input_text = self.build_input_from_content(user_content)
        else:
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

        print(f"\n[DEBUG] Generated {outputs[0].shape[-1] - input_len} new tokens")
        print(f"[DEBUG] Raw decoded response:\n{repr(response[:500])}")

        # Parse thinking vs final answer
        if hasattr(self.processor, "parse_response"):
            parsed = self.processor.parse_response(response)
        else:
            parsed = self._parse_response_fallback(response)

        print(f"[DEBUG] parse_response type: {type(parsed)}")
        if isinstance(parsed, dict):
            print(f"[DEBUG] parse_response keys: {list(parsed.keys())}")
            print(f"[DEBUG] thinking (first 200): {repr(str(parsed.get('thinking', ''))[:200])}")
            print(f"[DEBUG] content (first 200): {repr(str(parsed.get('content', ''))[:200])}")
        else:
            print(f"[DEBUG] parse_response value: {repr(str(parsed))[:500]}")

        return parsed

    def generate_batch(
        self,
        inputs: list,
        input_type: str = "lp_fields",
        batch_size: int = 1,
        **gen_kwargs,
    ) -> list[str]:
        """Batch inference. Returns list of raw output strings.

        inputs: list of lp_fields dicts (input_type="lp_fields")
                or list of user content strings (input_type="user_content")
        """
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i: i + batch_size]
            for item in batch:
                if input_type == "user_content":
                    result = self.generate(user_content=item, **gen_kwargs)
                else:
                    result = self.generate(lp_fields=item, **gen_kwargs)
                results.append(result)
            done = min(i + batch_size, len(inputs))
            print(f"  Processed {done}/{len(inputs)}")
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
        # parse_response returns dict with 'thinking' and 'content' keys
        thinking = response.get("thinking", "")
        answer = response.get("content", "")
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
    p.add_argument("--processor_id", default="",
                    help="Optional processor path (use when model dir lacks processor files, e.g. quantized models)")
    p.add_argument("--adapter_path", default="",
                    help="Optional LoRA adapter path (for SFT/DPO checkpoints)")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--use_gptq", action="store_true", default=False,
                    help="Load model via GPTQModel (for GPTQ quantized checkpoints)")
    p.add_argument("--no_think", action="store_true", default=False,
                    help="Disable thinking mode")
    p.add_argument("--no_cot", action="store_true", default=False,
                    help="Use no-CoT system prompt (skip <think> block, output prompts only)")
    p.add_argument("--max_lp_chars", type=int, default=0,
                    help="Truncate Primary Content to this many chars (0=no truncation, recommended: 2000)")
    p.add_argument("--attn_impl", type=str, default="",
                    help="Attention implementation: sdpa, flash_attention_2, eager (default: model default)")
    # Single query
    p.add_argument("--url", default="", help="Landing page URL")
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
        processor_id=args.processor_id or None,
        adapter_path=args.adapter_path or None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_gptq=args.use_gptq,
        no_cot=args.no_cot,
        max_lp_chars=args.max_lp_chars,
        enable_thinking=not args.no_think,
        attn_impl=args.attn_impl,
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
        batch_inputs = []
        input_type = "lp_fields"

        for r in records:
            if "lp_fields" in r:
                batch_inputs.append(r["lp_fields"])
            elif "messages" in r:
                # Directly use the user message content from SFT/DPO data
                # instead of extracting fields and rebuilding (avoids label mismatch)
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
        start_time = time.time()

        # Format compliance regexes
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
        total = len(records)

        # Stream results: write each record immediately after inference
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.output_file, "w", encoding="utf-8")

        for idx, (record, inp) in enumerate(zip(records, batch_inputs)):
            if input_type == "user_content":
                raw_response = gen.generate(user_content=inp, **gen_kwargs)
            else:
                raw_response = gen.generate(lp_fields=inp, **gen_kwargs)

            raw = parse_gemma_response(raw_response)
            prompts = parse_output_prompts(raw)

            compliant = bool(format_regex.search(raw) or format_regex_no_think.search(raw))
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
        print(f"\nTotal inference time: {elapsed:.1f}s ({elapsed/len(records):.1f}s/sample)")
        print(f"Saved {total} results -> {args.output_file}")

        print(f"\n{'='*60}")
        print(f"Format compliance (full): {n_compliant}/{total} ({100*n_compliant/total:.1f}%)")
        print(f"All 5 tags present:       {n_tags}/{total} ({100*n_tags/total:.1f}%)")
        print(f"{'='*60}")

    else:
        # Single query mode
        lp_fields = {
            "FinalDestinationURLUrl": args.url,
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
