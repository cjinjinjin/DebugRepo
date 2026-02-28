import os
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from utils.llm_client import fill_template, call_llm
from utils.xml_parser import parse_refine_prompt

_REFINE_TIMEOUT_S = 60  # per-prompt timeout in seconds


class ImagePromptRefinerAgent:
    """
    Step 5 of pipeline: refine each raw prompt into a natural,
    diffusion-model-ready description. Runs 5 prompts concurrently.
    """

    def __init__(self, config):
        self.config = config
        self.template_path = os.path.join(
            config.prompts_dir, "ImagePromptRefiner.txt"
        )

    def run(self, state) -> list:
        raw_prompts = state.raw_prompts
        if not raw_prompts:
            print("  [ImagePromptRefiner] No raw prompts to refine.")
            return []

        print(f"\n[ImagePromptRefiner] Refining {len(raw_prompts)} prompt(s) concurrently...")

        results = [None] * len(raw_prompts)

        def refine_one(idx: int, raw_prompt: str) -> tuple:
            variables = {"Prompt": raw_prompt}
            filled = fill_template(self.template_path, variables)
            # stream=False for concurrent calls to avoid interleaved output
            raw_out = call_llm(filled, stream=False, label=f"Refiner[{idx+1}]")
            refined = parse_refine_prompt(raw_out)
            if not refined:
                print(f"  [Refiner {idx+1}] Parse failed, using raw prompt as fallback.")
                refined = raw_prompt
            return idx, refined

        with ThreadPoolExecutor(max_workers=min(5, len(raw_prompts))) as pool:
            futures = {
                pool.submit(refine_one, i, p): i
                for i, p in enumerate(raw_prompts)
            }
            for future in as_completed(futures, timeout=_REFINE_TIMEOUT_S * len(raw_prompts)):
                try:
                    idx, refined = future.result(timeout=_REFINE_TIMEOUT_S)
                    results[idx] = refined
                    print(f"  [Refiner {idx+1}] Done.")
                except TimeoutError:
                    idx = futures[future]
                    print(f"  [Refiner {idx+1}] Timed out after {_REFINE_TIMEOUT_S}s. Using raw prompt.")
                    results[idx] = raw_prompts[idx]
                except Exception as e:
                    idx = futures[future]
                    print(f"  [Refiner {idx+1}] Error: {e}. Using raw prompt.")
                    results[idx] = raw_prompts[idx]

        return results
