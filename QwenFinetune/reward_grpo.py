"""
GRPO reward function for output format regularization.

Rewards correct structure:
  <think> block with 6 required fields
  Exactly 5 <PromptN>...</PromptN> tags in order (N=1..5)
  Each prompt <= 150 words
  No repeated n-grams across prompts (diversity penalty)

swift grpo loads this via --external_plugins ./reward_grpo.py
The entry point must be a function named `reward_fn` with signature:
  reward_fn(completions: list[str], **kwargs) -> list[float]
"""

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THINK_FIELDS = [
    "ProductType:",
    "SpecificProduct:",
    "Category:",
    "VisualAnchors:",
    "LifestyleVibe:",
    "CoreValueSignals:",
]

PROMPT_MAX_WORDS = 150

# Repetition: penalize if any 6-gram appears > 1 time across all 5 prompts
NGRAM_SIZE = 6
NGRAM_MAX_REPEAT = 1


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------

def _score_think(completion: str) -> float:
    """0.0 – 0.25: think block presence and field completeness."""
    m = re.search(r"<think>([\s\S]*?)</think>", completion)
    if not m:
        return 0.0
    score = 0.10  # block exists
    content = m.group(1)
    per_field = 0.15 / len(THINK_FIELDS)
    for field in THINK_FIELDS:
        if field in content:
            score += per_field
    return score


def _score_prompts(completion: str) -> float:
    """0.0 – 0.50: correct number, order, and length of Prompt tags."""
    # Find all <PromptN>...</PromptN> pairs
    pairs = re.findall(r"<Prompt(\d)>([\s\S]*?)</Prompt\1>", completion)
    n = len(pairs)

    if n == 0:
        return 0.0

    # Partial credit for count (up to 0.30)
    count_score = 0.30 if n == 5 else (0.06 * n)

    # Order bonus (0.10): indices must be exactly [1,2,3,4,5]
    indices = [int(p[0]) for p in pairs]
    order_score = 0.10 if indices == list(range(1, n + 1)) else 0.0

    # Per-prompt length score (up to 0.10): each of 5 prompts <= 150 words
    length_score = 0.0
    per_prompt = 0.02
    for _, text in pairs:
        wc = len(text.split())
        if wc <= PROMPT_MAX_WORDS:
            length_score += per_prompt
        else:
            # Soft penalty proportional to excess
            excess_ratio = (wc - PROMPT_MAX_WORDS) / PROMPT_MAX_WORDS
            length_score += per_prompt * max(0.0, 1.0 - excess_ratio)

    return count_score + order_score + length_score


def _score_diversity(completion: str) -> float:
    """
    0.0 – 0.25: penalize repeated n-grams across the 5 prompts.
    Full score if no n-gram repeats; linearly decays to 0 at 10+ repeats.
    """
    pairs = re.findall(r"<Prompt\d>([\s\S]*?)</Prompt\d>", completion)
    if len(pairs) < 2:
        return 0.0

    all_text = " ".join(pairs).lower()
    tokens = re.findall(r"\b\w+\b", all_text)

    if len(tokens) < NGRAM_SIZE:
        return 0.25

    ngram_counts = Counter(
        tuple(tokens[i: i + NGRAM_SIZE]) for i in range(len(tokens) - NGRAM_SIZE + 1)
    )
    n_repeated = sum(1 for c in ngram_counts.values() if c > NGRAM_MAX_REPEAT)
    # Decay: 0 repeats -> 0.25, 10+ repeats -> 0.0
    score = 0.25 * max(0.0, 1.0 - n_repeated / 10.0)
    return score


# ---------------------------------------------------------------------------
# Entry point expected by swift grpo --external_plugins
# ---------------------------------------------------------------------------

def reward_fn(completions: list, **kwargs) -> list:
    """
    Args:
        completions: list of generated response strings (one per sample in the batch)
    Returns:
        list of float rewards in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        # completions may be Message objects or plain strings
        if hasattr(completion, "content"):
            text = completion.content
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        r = _score_think(text) + _score_prompts(text) + _score_diversity(text)
        rewards.append(float(min(1.0, max(0.0, r))))
    return rewards
