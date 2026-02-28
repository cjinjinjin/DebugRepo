import json
import os
import re
import numpy as np
from typing import Optional


class RAGRetriever:
    """
    FAISS-based retriever over the offline knowledge base.
    Falls back gracefully if index is not built.
    """

    def __init__(self, config):
        self.config = config
        self.index = None
        self.metadata = []
        self._load_index()

    def _load_index(self):
        index_path = self.config.faiss_index_path
        meta_path = self.config.kb_metadata_path

        if not os.path.exists(index_path):
            print(f"  [RAG] Index not found at {index_path}. Run: python -m rag.kb_builder")
            return
        try:
            import faiss
            self.index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.metadata.append(json.loads(line))
            print(f"  [RAG] Loaded {len(self.metadata)} KB entries.")
        except Exception as e:
            print(f"  [RAG] Failed to load index: {e}")

    def query(self, lp_fields, top_k: Optional[int] = None) -> str:
        """
        Build query from LP fields, retrieve top-k chunks, format as context string.
        """
        if self.index is None:
            return ""

        k = top_k or self.config.rag_top_k
        query_text = self._build_query(lp_fields)
        if not query_text.strip():
            return ""

        try:
            from utils.llm_client import embed_text
            vec = embed_text(query_text)
            vec_np = np.array([vec], dtype=np.float32)

            # L2-normalize
            norm = np.linalg.norm(vec_np)
            if norm > 0:
                vec_np = vec_np / norm

            distances, indices = self.index.search(vec_np, k)
            retrieved = []
            for idx in indices[0]:
                if 0 <= idx < len(self.metadata):
                    retrieved.append(self.metadata[idx])

            return self._format_context(retrieved)
        except Exception as e:
            print(f"  [RAG] Query failed: {e}")
            return ""

    def _build_query(self, lp_fields) -> str:
        """Combine URL domain + title + meta description for retrieval query."""
        parts = []

        # Extract domain keywords from URL
        url = lp_fields.url
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if domain_match:
            domain = domain_match.group(1)
            # Strip TLD, split by dots/hyphens
            domain_words = re.split(r"[.\-]", domain)
            parts.extend([w for w in domain_words if len(w) > 2])

        # Extract path keywords
        path_match = re.search(r"https?://[^/]+(/[^\?#]*)", url)
        if path_match:
            path = path_match.group(1)
            path_words = re.split(r"[/\-_]", path)
            parts.extend([w for w in path_words if len(w) > 2])

        if lp_fields.document_title:
            parts.append(lp_fields.document_title)
        if lp_fields.meta_description_cb:
            parts.append(lp_fields.meta_description_cb[:200])
        if lp_fields.heading:
            parts.append(lp_fields.heading)

        return " ".join(parts)[:500]

    def _format_context(self, records: list) -> str:
        if not records:
            return ""
        lines = []
        for r in records:
            category = r.get("category", r.get("archetype", r.get("strategy", "")))
            text = r.get("text", "")
            visual_strategy = r.get("visual_strategy", "")
            value_signals = r.get("value_signals", [])
            line = f"[{category}] {text}"
            if visual_strategy:
                line += f" | Visual Strategy: {visual_strategy}"
            if value_signals:
                line += f" | Value Signals: {', '.join(value_signals)}"
            lines.append(line)
        return "\n\n".join(lines)
