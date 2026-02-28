import os
from utils.llm_client import fill_template, call_llm
from utils.xml_parser import parse_questions, parse_vlm_answers


class VLMEvaluator:
    """
    Optional Step 6: for each refined prompt, generate 12 verification
    questions and score them against a generated image (if provided).
    In CLI mode without actual image generation, this runs question
    generation only and returns the questions for human review.
    """

    def __init__(self, config):
        self.config = config
        self.qgen_template = os.path.join(
            config.prompts_dir, "VLMRewardPrompt", "QuestionGeneration.txt"
        )
        self.answer_template = os.path.join(
            config.prompts_dir, "VLMRewardPrompt", "VLMAnswerPrompt.txt"
        )

    def generate_questions(self, refined_prompt: str) -> list:
        """Generate 12 yes/no verification questions for a refined prompt."""
        variables = {"IMAGE INPUT PROMPT": refined_prompt}
        try:
            with open(self.qgen_template, "r", encoding="utf-8") as f:
                template = f.read()
            # The template uses the prompt inline, not a {placeholder}
            prompt = template + f"\n\nIMAGE INPUT PROMPT: {refined_prompt}"
            raw = call_llm(prompt, stream=False, label="QuestionGeneration")
            return parse_questions(raw)
        except Exception as e:
            print(f"  [VLMEvaluator] Question generation error: {e}")
            return []

    def run(self, state) -> tuple:
        """
        Returns (questions_per_prompt, scores_per_prompt).
        scores = fraction of 'yes' answers (0.0-1.0) per prompt.
        Without actual images, scores will be empty [].
        """
        all_questions = []
        for i, refined in enumerate(state.refined_prompts):
            print(f"\n[VLMEvaluator] Generating questions for Prompt {i+1}...")
            qs = self.generate_questions(refined)
            all_questions.append(qs)
            print(f"  Generated {len(qs)} questions.")

        # Image scoring requires actual generated images — not run in CLI mode
        scores = []
        return all_questions, scores
