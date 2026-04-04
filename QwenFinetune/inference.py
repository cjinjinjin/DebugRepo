"""
Inference with fine-tuned Qwen LoRA model.

Replaces the GPT5 two-step pipeline (LPUnderstanding + ImagePromptCreator)
with a single Qwen forward pass.

Usage:
  # Single query
  python inference.py \
      --adapter_path checkpoints/qwen35_sft_lora/lora_adapter \
      --url "https://example.com/product" \
      --title "Product Title"

  # Batch inference from JSONL file
  python inference.py \
      --adapter_path checkpoints/qwen35_sft_lora/lora_adapter \
      --input_file data/test_lp_fields.jsonl \
      --output_file results/generated_prompts.jsonl

  # Merge adapter into base model and save (for production deployment)
  python inference.py \
      --adapter_path checkpoints/qwen35_sft_lora/lora_adapter \
      --base_model  Qwen/Qwen2.5-35B-Instruct \
      --merge_and_save merged_model/
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Regex pattern for constrained decoding: enforces <think>...</think> + 5 <PromptN> tags
CONSTRAINED_PATTERN = (
    r"<think>[\s\S]{10,3000}</think>\s*"
    r"<Prompt1>[\s\S]{10,1500}</Prompt1>\s*"
    r"<Prompt2>[\s\S]{10,1500}</Prompt2>\s*"
    r"<Prompt3>[\s\S]{10,1500}</Prompt3>\s*"
    r"<Prompt4>[\s\S]{10,1500}</Prompt4>\s*"
    r"<Prompt5>[\s\S]{10,1500}</Prompt5>"
)

# Import the system prompt used during training
SYSTEM_PROMPT = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, your task is to generate five (5) high-quality English image generation prompts for Native Ads.

Each prompt must:
- Be ≤150 words
- Embed all safety, realism, quality, and exclusion constraints
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

Output exactly 5 prompts in this format:
<Prompt1>...</Prompt1>
<Prompt2>...</Prompt2>
<Prompt3>...</Prompt3>
<Prompt4>...</Prompt4>
<Prompt5>...</Prompt5>"""


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class QwenPromptGenerator:
    """
    Wraps the fine-tuned Qwen model for LP → image prompt generation.
    """

    def __init__(
        self,
        adapter_path: str,
        base_model: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype=torch.bfloat16,
        constrained: bool = False,
    ):
        self.adapter_path = adapter_path
        self.torch_dtype = torch_dtype
        self.constrained = constrained
        self._logits_processor = None

        # Determine where to load base model from
        if base_model:
            model_path = base_model
        else:
            # Try to read base_model_name_or_path from adapter config
            config_path = Path(adapter_path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    adapter_cfg = json.load(f)
                model_path = adapter_cfg.get("base_model_name_or_path", adapter_path)
            else:
                model_path = adapter_path  # merged model

        print(f"Loading tokenizer from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from transformers import BitsAndBytesConfig
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        print(f"Loading base model from {model_path} ...")
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch_dtype,
        )

        # Check if adapter_path is a PEFT adapter directory
        if (Path(adapter_path) / "adapter_config.json").exists() and adapter_path != model_path:
            print(f"Loading LoRA adapter from {adapter_path} ...")
            self.model = PeftModel.from_pretrained(base, adapter_path)
        else:
            # It's a merged/standalone model
            self.model = base

        self.model.eval()
        print("Model ready.")

        if self.constrained:
            self._init_constrained_decoding()

    def _init_constrained_decoding(self):
        """Initialize outlines regex-guided logits processor."""
        try:
            from outlines.processors import RegexLogitsProcessor
            print(f"Initializing constrained decoding with outlines ...")
            self._logits_processor = RegexLogitsProcessor(
                CONSTRAINED_PATTERN, tokenizer=self.tokenizer
            )
            print("Constrained decoding ready.")
        except ImportError:
            print(
                "WARNING: outlines not installed. Install with: pip install outlines\n"
                "Falling back to unconstrained generation.",
                file=sys.stderr,
            )
            self.constrained = False

    def build_input(self, lp_fields: dict) -> str:
        """Build the chat-template formatted input string."""
        field_labels = {
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
        lines = ["Generate 5 image prompts for the following landing page:\n"]
        for key, label in field_labels.items():
            val = lp_fields.get(key, "").strip()
            if val:
                lines.append(f"- {label}: {val}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        lp_fields: dict,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """
        Generate image prompts for a single LP.
        Returns a list of raw output strings (one per return sequence).
        """
        input_text = self.build_input(lp_fields)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1536,
        ).to(self.model.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generate_kwargs = {**inputs, "generation_config": gen_config}
        if self.constrained and self._logits_processor is not None:
            generate_kwargs["logits_processor"] = [self._logits_processor]

        output_ids = self.model.generate(**generate_kwargs)
        # Decode only the newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for seq_ids in output_ids:
            new_ids = seq_ids[prompt_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text.strip())
        return results

    def generate_batch(
        self,
        lp_fields_list: list[dict],
        batch_size: int = 4,
        **gen_kwargs,
    ) -> list[list[str]]:
        """Batch inference over a list of LP field dicts."""
        all_results = []
        for i in range(0, len(lp_fields_list), batch_size):
            batch = lp_fields_list[i: i + batch_size]
            input_texts = [self.build_input(lp) for lp in batch]

            enc = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536,
            ).to(self.model.device)

            max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
            temperature = gen_kwargs.get("temperature", 0.7)
            top_p = gen_kwargs.get("top_p", 0.9)
            do_sample = gen_kwargs.get("do_sample", True)

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            with torch.inference_mode():
                generate_kwargs = {**enc, "generation_config": gen_config}
                if self.constrained and self._logits_processor is not None:
                    generate_kwargs["logits_processor"] = [self._logits_processor]
                output_ids = self.model.generate(**generate_kwargs)

            prompt_len = enc["input_ids"].shape[1]
            for j, seq_ids in enumerate(output_ids):
                new_ids = seq_ids[prompt_len:]
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                all_results.append([text.strip()])

            print(f"  Processed {min(i + batch_size, len(lp_fields_list))}/{len(lp_fields_list)}")

        return all_results


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output_prompts(text: str) -> list[str]:
    """Extract <Prompt1>...</Prompt5> tags from model output."""
    prompts = []
    for i in range(1, 6):
        pattern = rf"<Prompt{i}>(.*?)</Prompt{i}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            prompts.append(match.group(1).strip())
    if not prompts:
        # Fallback: return the whole text as a single prompt
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
    print(f"Saved {len(records)} results → {path}")


# ---------------------------------------------------------------------------
# Merge adapter into base model
# ---------------------------------------------------------------------------

def merge_and_save(adapter_path: str, base_model: str, output_dir: str) -> None:
    """Merge LoRA weights into the base model and save for deployment."""
    from peft import PeftModel
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"Merging adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", required=True,
                   help="Path to LoRA adapter directory or merged model")
    p.add_argument("--base_model", default="",
                   help="Base model path (if not stored in adapter_config.json)")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    # Single query mode
    p.add_argument("--url", default="", help="Landing page URL")
    p.add_argument("--title", default="", help="Document title")
    p.add_argument("--heading", default="", help="Heading text")
    p.add_argument("--content", default="", help="Primary content text")
    # Batch mode
    p.add_argument("--input_file", default="", help="Input JSONL with lp_fields")
    p.add_argument("--output_file", default="results/generated_prompts.jsonl")
    p.add_argument("--batch_size", type=int, default=4)
    # Generation params
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--do_sample", action="store_true", default=True)
    # Constrained decoding
    p.add_argument("--constrained", action="store_true", default=False,
                   help="Enable regex-constrained decoding (requires outlines)")
    # Merge mode
    p.add_argument("--merge_and_save", default="",
                   help="If set, merge adapter into base model and save to this path")
    return p.parse_args()


def main():
    args = parse_args()

    # Merge-and-save shortcut (no GPU inference needed)
    if args.merge_and_save:
        if not args.base_model:
            print("ERROR: --base_model required for --merge_and_save")
            sys.exit(1)
        merge_and_save(args.adapter_path, args.base_model, args.merge_and_save)
        return

    # Load model
    gen = QwenPromptGenerator(
        adapter_path=args.adapter_path,
        base_model=args.base_model or None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        constrained=args.constrained,
    )
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
    }

    if args.input_file:
        # Batch mode
        records = load_jsonl(args.input_file)
        lp_fields_list = [r.get("lp_fields", r) for r in records]
        all_outputs = gen.generate_batch(lp_fields_list, batch_size=args.batch_size, **gen_kwargs)

        results = []
        for record, outputs in zip(records, all_outputs):
            prompts = parse_output_prompts(outputs[0])
            results.append({
                "id": record.get("id", ""),
                "lp_fields": record.get("lp_fields", record),
                "generated_prompts": prompts,
                "raw_output": outputs[0],
            })
        write_jsonl(results, args.output_file)

    else:
        # Single query mode from CLI args
        lp_fields = {
            "FinalDestinationURLUrl": args.url,
            "DocumentTitle": args.title,
            "Heading": args.heading,
            "PrimaryContentNoTitleNoHeading": args.content,
        }
        outputs = gen.generate(lp_fields, **gen_kwargs)
        prompts = parse_output_prompts(outputs[0])

        print("\n" + "=" * 60)
        print("Generated Image Prompts:")
        print("=" * 60)
        for i, p in enumerate(prompts, 1):
            print(f"\n[Prompt {i}]\n{p}")


if __name__ == "__main__":
    main()
