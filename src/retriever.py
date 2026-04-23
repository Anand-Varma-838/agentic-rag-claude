"""
retriever.py
------------
Two-stage hybrid retrieval:
  Stage 1 — Recall: semantic search + BM25 merged via Reciprocal Rank Fusion
  Stage 2 — Precision: cross-encoder reranking of top candidates
"""

import os
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from vectorstore import VectorStore

load_dotenv()

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 10))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 4))
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~70MB, CPU-friendly


class HybridRetriever:
    """
    Combines semantic search (ChromaDB) + keyword search (BM25)
    then reranks with a cross-encoder for high precision results.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[Dict[str, Any]] = []

    def build_bm25_index(self, corpus: Optional[List[Dict[str, Any]]] = None):
        """
        Build BM25 index from corpus.
        If corpus is None, loads all chunks from the vector store.
        """
        if corpus is None:
            corpus = self.vector_store.get_all()
        self._corpus = corpus
        if not corpus:
            return
        tokenized = [doc["text"].lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self._bm25 is None or not self._corpus:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(scores, self._corpus),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]
        return [
            {**doc, "bm25_score": float(score)}
            for score, doc in ranked
            if score > 0
        ]

    @staticmethod
    def _reciprocal_rank_fusion(
        semantic: List[Dict],
        keyword: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """Merge two ranked lists using RRF. Higher score = better rank."""
        scores: Dict[str, float] = {}
        docs: Dict[str, Dict] = {}
        for rank, doc in enumerate(semantic):
            key = doc["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs[key] = doc
        for rank, doc in enumerate(keyword):
            key = doc["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs[key] = doc
        return sorted(docs.values(), key=lambda d: scores[d["text"]], reverse=True)

    def retrieve(self, query: str, top_k: int = TOP_K_RERANK) -> List[Dict[str, Any]]:
        """
        Full pipeline: hybrid recall → RRF merge → cross-encoder rerank.
        Returns top_k results with rerank_score attached.
        """
        semantic_hits = self.vector_store.search(query, top_k=TOP_K_RETRIEVAL)
        keyword_hits = self._bm25_search(query, top_k=TOP_K_RETRIEVAL)
        merged = self._reciprocal_rank_fusion(semantic_hits, keyword_hits)
        candidates = merged[: TOP_K_RETRIEVAL * 2]

        if not candidates:
            return []

        pairs = [(query, doc["text"]) for doc in candidates]
        ce_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(ce_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return [
            {**doc, "rerank_score": float(score)}
            for score, doc in reranked
        ]

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format results into a context block for the LLM prompt."""
        if not results:
            return "No relevant documents found."
        parts = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            source = meta.get("source", "unknown")
            page = meta.get("page", "")
            citation = source + (f", page {page}" if page else "")
            parts.append(f"[{i}] Source: {citation}\n{r['text']}")
        return "\n\n---\n\n".join(parts)
