import os
from dataclasses import dataclass, field


@dataclass
class Config:
    llm_backend: str = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_api_base: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    faiss_index_path: str = ""
    kb_metadata_path: str = ""
    prompts_dir: str = ""
    playwright_timeout_ms: int = 15000
    crawler_retries: int = 2
    rag_top_k: int = 5
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7

    @classmethod
    def from_env(cls, llm_backend: str = "openai") -> "Config":
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(base_dir)
        return cls(
            llm_backend=llm_backend,
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            llm_api_key=os.getenv("OPENAI_API_KEY", ""),
            llm_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            embedding_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
            faiss_index_path=os.getenv(
                "FAISS_INDEX_PATH",
                os.path.join(base_dir, "rag", "kb.faiss"),
            ),
            kb_metadata_path=os.getenv(
                "KB_META_PATH",
                os.path.join(base_dir, "rag", "kb_meta.jsonl"),
            ),
            prompts_dir=os.getenv(
                "PROMPTS_DIR",
                os.path.join(repo_root, "CreativeAdsPrompt"),
            ),
            playwright_timeout_ms=int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "15000")),
            crawler_retries=int(os.getenv("CRAWLER_RETRIES", "2")),
            rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        )
