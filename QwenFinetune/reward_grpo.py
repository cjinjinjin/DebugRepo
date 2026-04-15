"""
GRPO reward function v2 — format + content quality.

v1 problem: format-only reward (tag closure +1.0 = 69% of max) caused reward
hacking — model learned to write 5 ultra-short tags (avg 18.7 words) for high
scores while producing useless output.

v2 changes:
  - Halve format weights (tags +0.1 each, think +0.1)
  - Add minimum length enforcement (< 20 words → -0.2 per prompt)
  - Add length sweet-spot reward (40-150 words → +0.1 per prompt)
  - Add CoT field completeness check (+0.05 per field, +0.1 bonus for all 6)
  - Add descriptiveness score (content-word ratio)
  - Over-length penalty only (> 150 words → -0.1 per prompt)

Score range: approx -3.0 to +2.0
  Good output (5 × 80-word prompts + full CoT):  ~+1.7
  Empty shells (5 × 3-word tags + empty think):   ~-0.4

swift grpo loads this via --external_plugins ./reward_grpo.py
Entry point: reward_fn(completions, **kwargs) -> list[float]
"""

import re
from collections import Counter
from swift.rewards import orms


PROMPT_MAX_WORDS = 150
PROMPT_MIN_WORDS = 20
PROMPT_GOOD_MIN  = 40
THINK_MAX_WORDS  = 300

COT_FIELDS = [
    "ProductType",
    "SpecificProduct",
    "Category",
    "VisualAnchors",
    "LifestyleVibe",
    "CoreValueSignals",
]

STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "will", "your",
    "they", "their", "them", "what", "when", "where", "which", "while",
    "also", "about", "more", "into", "than", "then", "some", "such",
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one", "our", "out",
}


# ---------------------------------------------------------------------------
# A. Format scores (reduced weight)
# ---------------------------------------------------------------------------

def _score_prompt_tags(text: str) -> float:
    """
    +0.1 for each complete <PromptN>...</PromptN> pair (N=1..5), max +0.5.
    """
    pairs = re.findall(r"<Prompt([1-5])>([\s\S]*?)</Prompt\1>", text)
    return 0.1 * len(pairs)


def _score_think_closed(text: str) -> float:
    """
    +0.1 if <think>...</think> is properly closed.
    """
    return 0.1 if re.search(r"<think>[\s\S]*?</think>", text) else 0.0


# ---------------------------------------------------------------------------
# B. Content quality scores (NEW)
# ---------------------------------------------------------------------------

def _score_prompt_min_length(text: str) -> float:
    """
    Per-prompt length scoring:
      < 20 words  → -0.2  (too short / empty shell)
      20-39 words → 0.0   (acceptable but not great)
      40-150 words → +0.1 (sweet spot)
    Range: [-1.0, +0.5]
    """
    pairs = re.findall(r"<Prompt[1-5]>([\s\S]*?)</Prompt[1-5]>", text)
    score = 0.0
    for content in pairs:
        wc = len(content.split())
        if wc < PROMPT_MIN_WORDS:
            score -= 0.2
        elif wc >= PROMPT_GOOD_MIN:
            score += 0.1
    return score


def _score_think_cot_fields(text: str) -> float:
    """
    Check for CoT field presence in think block.
    +0.05 per field found (6 fields max = +0.3)
    +0.1 bonus if all 6 present (total +0.4)
    """
    m = re.search(r"<think>([\s\S]*?)</think>", text)
    if not m:
        return 0.0
    think_content = m.group(1)
    found = sum(1 for f in COT_FIELDS if f + ":" in think_content)
    score = 0.05 * found
    if found == len(COT_FIELDS):
        score += 0.1
    return score


def _score_prompt_descriptiveness(text: str) -> float:
    """
    Measure how descriptive/specific the prompts are by computing the ratio
    of content words (4+ letter non-stopwords) to total words.

    Higher ratio = more specific nouns/adjectives = better T2I prompts.
    Score = 0.3 * avg_content_ratio across prompts. Range: [0, +0.3]
    """
    pairs = re.findall(r"<Prompt[1-5]>([\s\S]*?)</Prompt[1-5]>", text)
    if not pairs:
        return 0.0

    ratios = []
    for content in pairs:
        words = re.findall(r"\b[a-zA-Z]+\b", content.lower())
        if len(words) < 3:
            ratios.append(0.0)
            continue
        content_words = [w for w in words if len(w) >= 4 and w not in STOPWORDS]
        ratios.append(len(content_words) / len(words))

    avg_ratio = sum(ratios) / len(ratios)
    return 0.3 * avg_ratio


# ---------------------------------------------------------------------------
# C. Penalties (retained / adjusted)
# ---------------------------------------------------------------------------

def _penalty_bigram_repetition(text: str) -> float:
    """
    Proportional penalty up to -2.0 based on bigram repetition ratio
    across the content of all 5 prompts combined.
    """
    pairs = re.findall(r"<Prompt[1-5]>([\s\S]*?)</Prompt[1-5]>", text)
    if len(pairs) < 2:
        return 0.0

    all_tokens = re.findall(r"\b\w+\b", " ".join(pairs).lower())
    if len(all_tokens) < 2:
        return 0.0

    bigrams = [tuple(all_tokens[i:i+2]) for i in range(len(all_tokens) - 1)]
    counts = Counter(bigrams)
    total = len(bigrams)
    repeated = sum(c for c in counts.values() if c > 1)
    ratio = repeated / total
    return -2.0 * ratio


def _penalty_think_length(text: str) -> float:
    """
    Linear penalty up to -0.3 if think block exceeds 300 words.
    """
    m = re.search(r"<think>([\s\S]*?)</think>", text)
    if not m:
        return 0.0
    wc = len(m.group(1).split())
    if wc <= THINK_MAX_WORDS:
        return 0.0
    excess_ratio = (wc - THINK_MAX_WORDS) / THINK_MAX_WORDS
    return -0.3 * min(1.0, excess_ratio)


def _penalty_prompt_overlength(text: str) -> float:
    """
    -0.1 per prompt that exceeds 150 words. Range: [-0.5, 0]
    """
    pairs = re.findall(r"<Prompt[1-5]>([\s\S]*?)</Prompt[1-5]>", text)
    score = 0.0
    for content in pairs:
        wc = len(content.split())
        if wc > PROMPT_MAX_WORDS:
            score -= 0.1
    return score


# ---------------------------------------------------------------------------
# Entry point expected by swift grpo --external_plugins
# ---------------------------------------------------------------------------

def reward_fn(completions: list, **kwargs) -> list:
    """
    Args:
        completions: list of generated response strings (one per sample in batch)
    Returns:
        list of float rewards
    """
    rewards = []
    for completion in completions:
        if hasattr(completion, "content"):
            text = completion.content
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        r = (
            # A. Format (reduced weight)
            _score_prompt_tags(text)
            + _score_think_closed(text)
            # B. Content quality (NEW)
            + _score_prompt_min_length(text)
            + _score_think_cot_fields(text)
            + _score_prompt_descriptiveness(text)
            # C. Penalties
            + _penalty_bigram_repetition(text)
            + _penalty_think_length(text)
            + _penalty_prompt_overlength(text)
        )
        rewards.append(float(r))
    return rewards


class FormatQualityReward:
    def __init__(self, args=None, **kwargs):
        pass

    def __call__(self, completions, **kwargs):
        return reward_fn(completions, **kwargs)


# Register into swift's orms so --reward_funcs format_quality works
orms['format_quality'] = FormatQualityReward
