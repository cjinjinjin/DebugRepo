import time
from dataclasses import dataclass, field
from typing import Optional
from .field_extractor import extract_fields_from_html, detect_paywall


@dataclass
class LPFields:
    url: str
    document_title: str = ""
    visual_title: str = ""
    heading: str = ""
    title_cb: str = ""
    visual_title_cb: str = ""
    heading_cb: str = ""
    best_snippet_cb: str = ""
    meta_description_cb: str = ""
    primary_content: str = ""
    crawl_method: str = "failed"   # "trafilatura" | "playwright" | "bs4" | "partial_paywall" | "failed"

    def to_template_vars(self) -> dict:
        return {
            "FinalDestinationURLUrl": self.url,
            "DocumentTitle": self.document_title,
            "VisualTitle": self.visual_title,
            "Heading": self.heading,
            "Title_CB": self.title_cb,
            "VisualTitle_CB": self.visual_title_cb,
            "Heading_CB": self.heading_cb,
            "BestSnippet_CB": self.best_snippet_cb,
            "MetaDescription_CB": self.meta_description_cb,
            "PrimaryContentNoTitleNoHeading": self.primary_content,
        }


class LPCrawler:
    """
    Three-tier crawling strategy with graceful degradation:
    1. trafilatura  (fast, static HTML)
    2. playwright   (JS-heavy / SPA)
    3. requests + BeautifulSoup  (last resort, meta fields only)
    """

    def __init__(self, config=None):
        self.timeout_ms = getattr(config, "playwright_timeout_ms", 15000) if config else 15000
        self.retries = getattr(config, "crawler_retries", 2) if config else 2

    def crawl(self, url: str) -> LPFields:
        result = self._try_trafilatura(url)
        if result and result.primary_content:
            return result

        print(f"  [Crawler] trafilatura insufficient, trying playwright...")
        result = self._try_playwright(url)
        if result and result.primary_content:
            return result

        print(f"  [Crawler] playwright failed, falling back to requests+bs4...")
        result = self._try_requests_bs4(url)
        if result:
            return result

        print(f"  [Crawler] All strategies failed. Using URL-only mode.")
        return LPFields(url=url, crawl_method="failed")

    # ─── Strategy 1: trafilatura ───────────────────────────────────────────
    def _try_trafilatura(self, url: str) -> Optional[LPFields]:
        try:
            import trafilatura
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None

            # Extract full text
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )

            # Also extract metadata
            metadata = trafilatura.extract_metadata(downloaded)

            fields = extract_fields_from_html(downloaded, url)
            if metadata:
                if metadata.title and not fields["DocumentTitle"]:
                    fields["DocumentTitle"] = metadata.title
                if metadata.description and not fields["MetaDescription_CB"]:
                    fields["MetaDescription_CB"] = metadata.description

            if text:
                fields["PrimaryContentNoTitleNoHeading"] = text[:2000]

            lp = _fields_dict_to_lp(url, fields, "trafilatura")

            # Paywall detection
            if detect_paywall(downloaded, lp.primary_content):
                lp.crawl_method = "partial_paywall"
                print(f"  [Crawler] Paywall detected, limited content available.")

            return lp
        except Exception as e:
            print(f"  [Crawler] trafilatura error: {e}")
            return None

    # ─── Strategy 2: playwright ────────────────────────────────────────────
    def _try_playwright(self, url: str) -> Optional[LPFields]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("  [Crawler] playwright not installed. Skipping.")
            return None

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                )
                page = context.new_page()
                page.goto(url, timeout=self.timeout_ms)
                page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
                html = page.content()
                browser.close()

            fields = extract_fields_from_html(html, url)
            lp = _fields_dict_to_lp(url, fields, "playwright")

            if detect_paywall(html, lp.primary_content):
                lp.crawl_method = "partial_paywall"

            return lp
        except Exception as e:
            print(f"  [Crawler] playwright error: {e}")
            return None

    # ─── Strategy 3: requests + BeautifulSoup ─────────────────────────────
    def _try_requests_bs4(self, url: str) -> Optional[LPFields]:
        try:
            import requests
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            }
            for attempt in range(self.retries):
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    resp.raise_for_status()
                    fields = extract_fields_from_html(resp.text, url)
                    return _fields_dict_to_lp(url, fields, "bs4")
                except Exception as e:
                    if attempt < self.retries - 1:
                        time.sleep(3)
                    else:
                        raise e
        except Exception as e:
            print(f"  [Crawler] requests+bs4 error: {e}")
            return None


def _fields_dict_to_lp(url: str, fields: dict, method: str) -> LPFields:
    return LPFields(
        url=url,
        document_title=fields.get("DocumentTitle", ""),
        visual_title=fields.get("VisualTitle", ""),
        heading=fields.get("Heading", ""),
        title_cb=fields.get("Title_CB", ""),
        visual_title_cb=fields.get("VisualTitle_CB", ""),
        heading_cb=fields.get("Heading_CB", ""),
        best_snippet_cb=fields.get("BestSnippet_CB", ""),
        meta_description_cb=fields.get("MetaDescription_CB", ""),
        primary_content=fields.get("PrimaryContentNoTitleNoHeading", ""),
        crawl_method=method,
    )
