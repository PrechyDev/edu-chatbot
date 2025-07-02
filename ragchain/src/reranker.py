from sentence_transformers import CrossEncoder
from typing import List
from langchain.schema import Document

# Lightweight reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query: str, docs: List[Document], top_k: int = 5):
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    for i, score in enumerate(scores):
        docs[i].metadata["relevance_score"] = float(score)

    reranked = sorted(docs, key=lambda d: d.metadata["relevance_score"], reverse=True)
    return reranked[:top_k]
