import re
from bs4 import BeautifulSoup


def extract_fields_from_html(html: str, url: str = "") -> dict:
    """
    Extract LP fields from raw HTML string.
    Returns dict with keys matching LPUnderstanding.txt placeholders.
    """
    soup = BeautifulSoup(html, "lxml")

    # DocumentTitle
    title_tag = soup.find("title")
    document_title = title_tag.get_text(strip=True) if title_tag else ""

    # VisualTitle: <h1> or og:title
    h1 = soup.find("h1")
    og_title = soup.find("meta", property="og:title")
    visual_title = ""
    if h1:
        visual_title = h1.get_text(strip=True)
    elif og_title:
        visual_title = og_title.get("content", "")

    # Heading: first <h2>
    h2 = soup.find("h2")
    heading = h2.get_text(strip=True) if h2 else ""

    # MetaDescription: <meta name="description"> or og:description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    og_desc = soup.find("meta", property="og:description")
    meta_description = ""
    if meta_desc:
        meta_description = meta_desc.get("content", "")
    elif og_desc:
        meta_description = og_desc.get("content", "")

    # PrimaryContent: remove nav/footer/header, get text
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    body_text = soup.get_text(separator=" ", strip=True)
    # Remove excessive whitespace
    body_text = re.sub(r"\s+", " ", body_text).strip()

    # Remove title and heading from primary content to match field name
    primary_content = body_text
    if document_title and document_title in primary_content:
        primary_content = primary_content.replace(document_title, "", 1).strip()
    if heading and heading in primary_content:
        primary_content = primary_content.replace(heading, "", 1).strip()

    # Truncate to 2000 chars
    primary_content = primary_content[:2000]

    # BestSnippet: first meaningful sentence (50-200 chars)
    best_snippet = ""
    sentences = re.split(r"[.!?]", body_text)
    for s in sentences:
        s = s.strip()
        if 30 < len(s) < 250:
            best_snippet = s
            break
    if not best_snippet:
        best_snippet = body_text[:200]

    return {
        "DocumentTitle": document_title,
        "VisualTitle": visual_title,
        "Heading": heading,
        "Title_CB": document_title,         # CB = Bing Crawler field alias
        "VisualTitle_CB": visual_title,
        "Heading_CB": heading,
        "BestSnippet_CB": best_snippet,
        "MetaDescription_CB": meta_description,
        "PrimaryContentNoTitleNoHeading": primary_content,
    }


def detect_paywall(html: str, primary_content: str) -> bool:
    """Heuristic: short content + login/subscribe form."""
    word_count = len(primary_content.split())
    html_lower = html.lower()
    has_gate = any(kw in html_lower for kw in [
        "subscribe", "sign in", "log in", "login", "create account",
        "members only", "premium content",
    ])
    return word_count < 100 and has_gate
