import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple

"""
Gemma4 T2I Prompt Generation - DLIS dlis_inter.py (Two-Step)

True two-step inference with two vLLM engine calls:
  Step 1: Generate 5 diverse scene concepts (short phrases)
  Step 2: Expand each scene into a full 30-50 word image prompt (batch of 5)

Requires modified model.py to call:
  1. preprocess(data) → step1 prompts
  2. engine.run(step1_prompts) → step1 outputs
  3. build_step2_prompts(step1_outputs, metadata) → step2 prompts
  4. engine.run(step2_prompts) → step2 outputs
  5. postprocess(step2_outputs, metadata) → final result

Request format (JSON):
{
    "landing_page_content": "Welcome to our outdoor adventure store...",
    "url": "https://example.com",
    "num_prompts": 5,           # optional, default 5
    "max_lp_chars": 5000        # optional, default 5000
}

Response format (JSON):
{
    "generated_prompts": ["prompt1", "prompt2", ...],
    "scenes": ["scene1", "scene2", ...],
    "raw_output": "...",
    "format_compliant": true,
    "Status": "Success"
}
"""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_STEP1 = """You are an expert Ad Creative Director specialized in high-performing Native Advertisement visuals.

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

SYSTEM_PROMPT_STEP2 = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page and a scene concept, expand the scene into a detailed image generation prompt for a Native Ad.

The prompt must:
- Be 30–50 words
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

Output exactly one prompt (no reasoning, no thinking):
<Prompt>...</Prompt>"""

FORMAT_REGEX = re.compile(
    r"<Prompt1>[\s\S]+?</Prompt1>\s*"
    r"<Prompt2>[\s\S]+?</Prompt2>\s*"
    r"<Prompt3>[\s\S]+?</Prompt3>\s*"
    r"<Prompt4>[\s\S]+?</Prompt4>\s*"
    r"<Prompt5>[\s\S]+?</Prompt5>"
)

if os.path.exists("/Model"):
    print(os.listdir("/Model"))


class PreAndPostProcessor:
    def __init__(self, processor=None):
        self.processor = processor

    # ------------------------------------------------------------------
    # Step 1: preprocess — build scene generation prompts
    # ------------------------------------------------------------------
    def preprocess(self, input_data):
        """
        Build Step 1 prompt: generate 5 diverse scene concepts.

        Returns:
            (vllm_inputs, metadata)
            - vllm_inputs: list with 1 prompt dict for scene generation
            - metadata: list with 1 metadata dict (carries user_message for Step 2)
        """
        if isinstance(input_data, str):
            input_data = json.loads(input_data)
        elif not isinstance(input_data, dict):
            raise ValueError("Input must be string or dict")

        assert 'landing_page_content' in input_data, \
            "Input data must contain 'landing_page_content'"

        lp_content = input_data['landing_page_content']
        url = input_data.get('url', '')
        num_prompts = input_data.get('num_prompts', 5)
        max_lp_chars = int(input_data.get('max_lp_chars',
                                          os.getenv('MAX_DOC_LENGTH', '5000')))

        # Truncate LP content
        if len(lp_content) > max_lp_chars:
            lp_content = lp_content[:max_lp_chars] + '... [truncated]'

        # Build user message (reused in Step 2)
        user_lines = [
            f"Generate {num_prompts} diverse scene concepts for the following landing page:\n"
        ]
        if url:
            user_lines.append(f"- URL: {url}")
        user_lines.append(f"- Primary Content: {lp_content}")
        user_message = "\n".join(user_lines)

        # Build Step 1 chat prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STEP1},
            {"role": "user", "content": user_message},
        ]
        prompt_text = self._apply_chat_template(messages)

        vllm_input = {'prompt': prompt_text}
        meta = {
            'url': url,
            'num_prompts': num_prompts,
            'user_message': user_message,  # saved for Step 2
        }

        return ([vllm_input], [meta])

    # ------------------------------------------------------------------
    # Step 2: build expansion prompts from Step 1 output
    # ------------------------------------------------------------------
    def build_step2_prompts(self, step1_output, metadata):
        """
        Parse scenes from Step 1 output and build Step 2 expansion prompts.

        Args:
            step1_output: generated text from Step 1 (str or list[str])
            metadata: metadata list from preprocess

        Returns:
            (step2_prompts, step2_metadata)
            - step2_prompts: list of prompt dicts (one per scene, typically 5)
            - step2_metadata: updated metadata with scenes info
        """
        # Handle nested list from vLLM runner: [[text], [text], ...]
        step1_text = step1_output
        while isinstance(step1_text, list):
            step1_text = step1_text[0] if step1_text else ""

        meta = metadata[0] if isinstance(metadata, list) else metadata
        user_message = meta.get('user_message', '')

        # Append closing tag if stopped by stop string
        if not step1_text.rstrip().endswith("</Scene5>"):
            step1_text = step1_text + "</Scene5>"

        # Parse scenes
        scenes = self._parse_scenes(step1_text)
        if not scenes:
            print(f"Warning: No scenes parsed from Step 1 output: {step1_text[:200]}")

        # Build Step 2 prompts — one per scene
        step2_prompts = []
        for scene in scenes:
            scene_content = (
                f"{user_message}\n\n"
                f"Expand this scene concept into a detailed prompt:\n"
                f"<Scene>{scene}</Scene>"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_STEP2},
                {"role": "user", "content": scene_content},
            ]
            prompt_text = self._apply_chat_template(messages)
            step2_prompts.append({'prompt': prompt_text})

        # Update metadata with scenes
        meta['scenes'] = scenes
        meta['step1_raw'] = step1_text

        return (step2_prompts, [meta])

    # ------------------------------------------------------------------
    # Final: postprocess Step 2 outputs
    # ------------------------------------------------------------------
    def postprocess(self, step2_outputs, metadata):
        """
        Parse Step 2 outputs into final structured response.

        Args:
            step2_outputs: list of generated texts from Step 2 (one per scene)
            metadata: metadata from build_step2_prompts

        Returns:
            dict with generated_prompts, scenes, raw_output, format_compliant, Status
        """
        meta = metadata[0] if isinstance(metadata, list) else metadata
        scenes = meta.get('scenes', [])
        step1_raw = meta.get('step1_raw', '')
        num_prompts = meta.get('num_prompts', 5)

        # Ensure step2_outputs is a flat list of strings
        # vLLM runner returns [[text], [text], ...], flatten to [text, text, ...]
        if isinstance(step2_outputs, str):
            step2_outputs = [step2_outputs]
        flat_outputs = []
        for item in step2_outputs:
            while isinstance(item, list):
                item = item[0] if item else ""
            flat_outputs.append(item)
        step2_outputs = flat_outputs

        # Parse each Step 2 output
        prompts = []
        step2_raws = []
        for raw in step2_outputs:
            if not raw.rstrip().endswith("</Prompt>"):
                raw = raw + "</Prompt>"
            step2_raws.append(raw)
            prompt = self._parse_single_prompt(raw)
            prompts.append(prompt)

        non_empty = [p for p in prompts if p]

        # Build combined raw output in <Prompt1>...<Prompt5> format
        formatted_parts = []
        for i, p in enumerate(prompts, 1):
            formatted_parts.append(f"<Prompt{i}>{p}</Prompt{i}>")
        combined_raw = "\n\n".join(formatted_parts)

        # Check format compliance
        compliant = bool(FORMAT_REGEX.search(combined_raw)) if len(prompts) >= 5 else len(non_empty) == num_prompts

        result = {
            'generated_prompts': prompts,
            'scenes': scenes,
            'raw_output': combined_raw,
            'step1_raw': step1_raw,
            'individual_raw_outputs': step2_raws,
            'format_compliant': compliant,
            'Status': 'Success' if non_empty else 'Failed',
        }

        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply chat template via processor or fallback."""
        if self.processor is not None:
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self._format_gemma_chat(messages)

    @staticmethod
    def _parse_scenes(text: str) -> List[str]:
        """Extract <Scene1>...<Scene5> from Step 1 output."""
        scenes = []
        for i in range(1, 6):
            m = re.search(rf"<Scene{i}>(.*?)</Scene{i}>", text, re.DOTALL)
            if m:
                scenes.append(m.group(1).strip())
        return scenes

    @staticmethod
    def _parse_single_prompt(text: str) -> str:
        """Extract <Prompt>...</Prompt> from Step 2 output."""
        m = re.search(r"<Prompt>(.*?)</Prompt>", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    @staticmethod
    def _format_gemma_chat(messages: List[Dict]) -> str:
        """Fallback chat template for Gemma models (no processor available)."""
        parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                parts.append(f"<start_of_turn>user\n{content}")
            elif role == 'user':
                if parts and parts[-1].startswith("<start_of_turn>user"):
                    parts[-1] += f"\n\n{content}<end_of_turn>"
                else:
                    parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == 'assistant':
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)
