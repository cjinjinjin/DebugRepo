import os
from utils.llm_client import fill_template, call_llm
from utils.xml_parser import parse_prompt_tags


class ImagePromptCreatorAgent:
    """
    Step 4 of pipeline: given LP understanding + original LP fields,
    generate 5 image prompts.
    """

    def __init__(self, config):
        self.config = config
        self.template_path = os.path.join(
            config.prompts_dir, "ImagePromptCreator.txt"
        )

    def run(self, state) -> list:
        u = state.lp_understanding
        lp = state.lp_fields

        variables = {
            # Input A: LPUnderstanding outputs
            "ProductIntent": u.product_intent,
            "ProductCategory": u.product_category,
            "VisualContext": u.visual_context,
            "AudienceAndContext": u.audience_and_context,
            "ValueSignals": u.value_signals,
            "ConfidenceLevel": u.confidence_level,
            # Input B: original LP fields (pass-through)
            **lp.to_template_vars(),
        }

        print("\n[ImagePromptCreator] Calling LLM...")
        prompt = fill_template(self.template_path, variables)
        raw = call_llm(prompt, stream=True, label="ImagePromptCreator")

        prompts = parse_prompt_tags(raw)

        if not prompts:
            print("  [ImagePromptCreator] WARNING: No prompts parsed from LLM output.")
            # Fallback: treat entire response as one prompt
            if raw.strip():
                prompts = [raw.strip()]

        print(f"  [ImagePromptCreator] Generated {len(prompts)} prompt(s).")
        return prompts
