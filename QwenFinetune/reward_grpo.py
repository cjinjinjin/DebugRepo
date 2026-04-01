"""
GRPO reward function for output format regularization.

Reward components (can go negative):
  +0.2 per complete <PromptN>...</PromptN> tag, N=1..5   (+1.0 max)
  +0.2  <think>...</think> closed properly               (binary)
  -0.3  think block > 300 words                          (linear penalty)
  -2.0  bigram repetition ratio across 5 prompts         (proportional, max -2.0)
  ±0.05 per prompt <= 150 words                          (+0.05 each, -0.05 if over)

swift grpo loads this via --external_plugins ./reward_grpo.py
Entry point: reward_fn(completions, **kwargs) -> list[float]
"""

import re
from collections import Counter


PROMPT_MAX_WORDS = 150
THINK_MAX_WORDS  = 300


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------

def _score_prompt_tags(text: str) -> float:
    """
    +0.2 for each complete <PromptN>...</PromptN> pair (N=1..5), max +1.0.
    Only counts tags whose index is 1-5 and opening/closing match.
    """
    pairs = re.findall(r"<Prompt([1-5])>([\s\S]*?)</Prompt\1>", text)
    return 0.2 * len(pairs)


def _score_think_closed(text: str) -> float:
    """
    +0.2 if <think>...</think> is properly closed.
    """
    return 0.2 if re.search(r"<think>[\s\S]*?</think>", text) else 0.0


def _penalty_think_length(text: str) -> float:
    """
    Linear penalty up to -0.3 if think block exceeds 300 words.
    At 300 words: 0.0. At 600 words: -0.3. Capped at -0.3.
    """
    m = re.search(r"<think>([\s\S]*?)</think>", text)
    if not m:
        return 0.0
    wc = len(m.group(1).split())
    if wc <= THINK_MAX_WORDS:
        return 0.0
    excess_ratio = (wc - THINK_MAX_WORDS) / THINK_MAX_WORDS  # 0→0, 1.0→full
    return -0.3 * min(1.0, excess_ratio)


def _penalty_bigram_repetition(text: str) -> float:
    """
    Proportional penalty up to -2.0 based on bigram repetition ratio
    across the content of all 5 prompts combined.

    repetition_ratio = (repeated bigram tokens) / (total bigram tokens)
    penalty = -2.0 * repetition_ratio
    """
    pairs = re.findall(r"<Prompt[1-5]>([\s\S]*?)</Prompt[1-5]>", text)
    if len(pairs) < 2:
        return 0.0

    all_tokens = re.findall(r"\b\w+\b", " ".join(pairs).lower())
    if len(all_tokens) < 2:
        return 0.0

    bigrams = [tuple(all_tokens[i:i+2]) for i in range(len(all_tokens) - 1)]
    counts = Counter(bigrams)
    total   = len(bigrams)
    # tokens that appear in a repeated bigram (count > 1)
    repeated = sum(c for c in counts.values() if c > 1)
    ratio = repeated / total
    return -2.0 * ratio


def _score_prompt_lengths(text: str) -> float:
    """
    +0.05 per prompt that is <= 150 words, -0.05 if over. Range: -0.25 to +0.25.
    """
    pairs = re.findall(r"<Prompt[1-5]>([\s\S]*?)</Prompt[1-5]>", text)
    score = 0.0
    for content in pairs:
        wc = len(content.split())
        score += 0.05 if wc <= PROMPT_MAX_WORDS else -0.05
    return score


# ---------------------------------------------------------------------------
# Entry point expected by swift grpo --external_plugins
# ---------------------------------------------------------------------------

def reward_fn(completions: list, **kwargs) -> list:
    """
    Args:
        completions: list of generated response strings (one per sample in batch)
    Returns:
        list of float rewards (unbounded below, max ~1.45)
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
            _score_prompt_tags(text)
            + _score_think_closed(text)
            + _penalty_think_length(text)
            + _penalty_bigram_repetition(text)
            + _score_prompt_lengths(text)
        )
        rewards.append(float(r))
    return rewards
