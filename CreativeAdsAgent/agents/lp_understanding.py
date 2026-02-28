import os
from utils.llm_client import fill_template, call_llm
from utils.xml_parser import parse_lp_understanding, LPUnderstanding


class LPUnderstandingAgent:
    """
    Step 3 of pipeline: given LP fields + RAG context,
    call LLM to produce structured product understanding.
    """

    def __init__(self, config):
        self.config = config
        self.template_path = os.path.join(
            config.prompts_dir, "LPUnderstanding.txt"
        )

    def run(self, state) -> LPUnderstanding:
        lp = state.lp_fields
        variables = lp.to_template_vars()
        variables["RAGContext"] = state.rag_context or ""

        print("\n[LPUnderstanding] Calling LLM...")
        prompt = fill_template(self.template_path, variables)
        raw = call_llm(prompt, stream=True, label="LPUnderstanding")

        result = parse_lp_understanding(raw)

        if not result.product_intent:
            print("  [LPUnderstanding] WARNING: Could not parse ProductIntent — using raw output.")
            result.confidence_level = "Low"

        return result
