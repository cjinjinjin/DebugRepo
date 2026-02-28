import re
from dataclasses import dataclass


@dataclass
class LPUnderstanding:
    product_intent: str = ""
    product_category: str = ""
    visual_context: str = ""
    audience_and_context: str = ""
    value_signals: str = ""
    confidence_level: str = ""
    raw_llm_output: str = ""


def parse_xml_tag(text: str, tag: str) -> str:
    """Extract content of first <tag>...</tag> occurrence."""
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_lp_understanding(raw: str) -> LPUnderstanding:
    return LPUnderstanding(
        product_intent=parse_xml_tag(raw, "ProductIntent"),
        product_category=parse_xml_tag(raw, "ProductCategory"),
        visual_context=parse_xml_tag(raw, "VisualContext"),
        audience_and_context=parse_xml_tag(raw, "AudienceAndContext"),
        value_signals=parse_xml_tag(raw, "ValueSignals"),
        confidence_level=parse_xml_tag(raw, "ConfidenceLevel"),
        raw_llm_output=raw,
    )


def parse_prompt_tags(raw: str) -> list:
    """Extract <Prompt1> ... <Prompt5> blocks."""
    results = []
    for i in range(1, 6):
        content = parse_xml_tag(raw, f"Prompt{i}")
        if content:
            results.append(content)
    return results


def parse_refine_prompt(raw: str) -> str:
    return parse_xml_tag(raw, "RefinePrompt")


def parse_questions(raw: str) -> list:
    """Extract <Q1> ... <Q12> verification questions."""
    questions = []
    for i in range(1, 13):
        q = parse_xml_tag(raw, f"Q{i}")
        if q:
            questions.append(q)
    return questions


def parse_vlm_answers(raw: str) -> list:
    """Extract <Q1>yes|no|n/a</Q1> ... <Q12> VLM answers."""
    answers = []
    for i in range(1, 13):
        a = parse_xml_tag(raw, f"Q{i}")
        answers.append(a.strip().lower() if a else "n/a")
    return answers
