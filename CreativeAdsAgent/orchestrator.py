import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from config import Config
from crawler.lp_crawler import LPCrawler, LPFields
from rag.retriever import RAGRetriever
from agents.lp_understanding import LPUnderstandingAgent
from agents.prompt_creator import ImagePromptCreatorAgent
from agents.prompt_refiner import ImagePromptRefinerAgent
from agents.vlm_evaluator import VLMEvaluator
from utils.xml_parser import LPUnderstanding
from utils.stream_printer import print_step


@dataclass
class PipelineState:
    url: str
    lp_fields: Optional[LPFields] = None
    rag_context: str = ""
    lp_understanding: Optional[LPUnderstanding] = None
    raw_prompts: list = field(default_factory=list)
    refined_prompts: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    vlm_scores: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def save(self, path: str):
        data = {
            "url": self.url,
            "lp_fields": asdict(self.lp_fields) if self.lp_fields else None,
            "rag_context": self.rag_context,
            "lp_understanding": asdict(self.lp_understanding) if self.lp_understanding else None,
            "raw_prompts": self.raw_prompts,
            "refined_prompts": self.refined_prompts,
            "questions": self.questions,
            "vlm_scores": self.vlm_scores,
            "errors": self.errors,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n[Pipeline] Results saved → {path}")


class PipelineOrchestrator:

    def __init__(self, config: Config):
        self.config = config
        self.crawler = LPCrawler(config)
        self.rag = RAGRetriever(config)
        self.lp_agent = LPUnderstandingAgent(config)
        self.creator = ImagePromptCreatorAgent(config)
        self.refiner = ImagePromptRefinerAgent(config)
        self.evaluator = VLMEvaluator(config)

    def run(self, url: str, eval_mode: bool = False) -> PipelineState:
        state = PipelineState(url=url)

        # ── Step 1: Crawl LP ────────────────────────────────────────────────
        print_step("Step 1 / 5 — LP Crawling", url)
        try:
            state.lp_fields = self.crawler.crawl(url)
            print_step("LP Fields Extracted", state.lp_fields)
        except Exception as e:
            msg = f"Crawler failed: {e}"
            print(f"  ERROR: {msg}")
            state.errors.append(msg)
            state.lp_fields = LPFields(url=url, crawl_method="failed")

        # ── Step 2: RAG Retrieval ───────────────────────────────────────────
        print_step("Step 2 / 5 — RAG Retrieval")
        try:
            state.rag_context = self.rag.query(state.lp_fields)
            if state.rag_context:
                print(f"  Retrieved {len(state.rag_context.split(chr(10)+chr(10)))} KB entries.")
            else:
                print("  No RAG context (index not built or query empty).")
        except Exception as e:
            msg = f"RAG failed: {e}"
            print(f"  ERROR: {msg}")
            state.errors.append(msg)
            state.rag_context = ""

        # ── Step 3: LP Understanding ────────────────────────────────────────
        print_step("Step 3 / 5 — LP Understanding")
        try:
            state.lp_understanding = self.lp_agent.run(state)
            print_step("LP Understanding Result", state.lp_understanding)
        except Exception as e:
            msg = f"LPUnderstanding failed: {e}"
            print(f"  ERROR: {msg}")
            state.errors.append(msg)
            state.lp_understanding = LPUnderstanding(confidence_level="Low")

        # ── Step 4: Image Prompt Creator ────────────────────────────────────
        print_step("Step 4 / 5 — Image Prompt Creator")
        try:
            state.raw_prompts = self.creator.run(state)
        except Exception as e:
            msg = f"PromptCreator failed: {e}"
            print(f"  ERROR: {msg}")
            state.errors.append(msg)
            state.raw_prompts = []

        # ── Step 5: Image Prompt Refiner ────────────────────────────────────
        print_step("Step 5 / 5 — Image Prompt Refiner")
        try:
            state.refined_prompts = self.refiner.run(state)
        except Exception as e:
            msg = f"PromptRefiner failed: {e}"
            print(f"  ERROR: {msg}")
            state.errors.append(msg)
            state.refined_prompts = state.raw_prompts  # fallback

        # ── Step 6 (optional): VLM Evaluation ──────────────────────────────
        if eval_mode:
            print_step("Step 6 (optional) — VLM Question Generation")
            try:
                state.questions, state.vlm_scores = self.evaluator.run(state)
            except Exception as e:
                msg = f"VLMEvaluator failed: {e}"
                print(f"  ERROR: {msg}")
                state.errors.append(msg)

        return state
