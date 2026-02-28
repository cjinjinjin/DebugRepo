"""
Knowledge base builder — run once to create the FAISS index.

Usage:
    cd CreativeAdsAgent
    python -m rag.kb_builder
"""
import json
import os
import numpy as np


KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
INDEX_OUT = os.path.join(os.path.dirname(__file__), "kb.faiss")
META_OUT = os.path.join(os.path.dirname(__file__), "kb_meta.jsonl")


def load_jsonl_files(kb_dir: str) -> list:
    records = []
    for fname in os.listdir(kb_dir):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(kb_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    return records


def build_index(config=None):
    try:
        import faiss
    except ImportError:
        print("[KB Builder] faiss-cpu not installed. Run: pip install faiss-cpu")
        return

    from utils.llm_client import embed_batch, set_config
    if config:
        set_config(config)

    print(f"[KB Builder] Loading records from {KB_DIR}...")
    records = load_jsonl_files(KB_DIR)
    print(f"[KB Builder] {len(records)} records loaded.")

    texts = [r.get("text", "") for r in records]

    print("[KB Builder] Embedding records...")
    # Batch in groups of 100 to avoid API limits
    all_vectors = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embed_batch(batch)
        all_vectors.extend(vecs)
        print(f"  Embedded {min(i + batch_size, len(texts))} / {len(texts)}")

    vectors_np = np.array(all_vectors, dtype=np.float32)

    # L2-normalize for inner product = cosine similarity
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vectors_np = vectors_np / norms

    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    faiss.write_index(index, INDEX_OUT)
    print(f"[KB Builder] FAISS index saved → {INDEX_OUT}")

    with open(META_OUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[KB Builder] Metadata saved → {META_OUT}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import Config
    from utils.llm_client import set_config
    cfg = Config.from_env()
    set_config(cfg)
    build_index(cfg)
