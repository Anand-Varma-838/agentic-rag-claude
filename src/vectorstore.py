"""
vectorstore.py
--------------
Handles document loading, chunking, embedding (sentence-transformers),
and storage/retrieval via ChromaDB.
No external API keys needed — embeddings run fully locally.
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME = "rag_documents"
EMBED_MODEL = "all-MiniLM-L6-v2"   # ~80MB, fast, CPU-friendly, no API key
CHUNK_SIZE = 400    # words per chunk
CHUNK_OVERLAP = 50  # word overlap between chunks


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """sentence-transformers wrapper. Downloads model once, then cached."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False).tolist()[0]


# ---------------------------------------------------------------------------
# Document chunk
# ---------------------------------------------------------------------------

class DocumentChunk:
    def __init__(self, text: str, source: str, page: int = 0, chunk_index: int = 0):
        self.id = str(uuid.uuid4())
        self.text = text
        self.source = source
        self.page = page
        self.chunk_index = chunk_index

    def metadata(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str, page: int = 0) -> List[DocumentChunk]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        window = words[i: i + CHUNK_SIZE]
        chunk_str = " ".join(window).strip()
        if len(chunk_str) > 50:
            chunks.append(DocumentChunk(chunk_str, source=source, page=page, chunk_index=idx))
            idx += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_pdf(path: str) -> List[DocumentChunk]:
    reader = PdfReader(path)
    chunks = []
    source = Path(path).name
    for page_num, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            chunks.extend(chunk_text(text, source=source, page=page_num + 1))
    return chunks


def load_txt(path: str) -> List[DocumentChunk]:
    source = Path(path).name
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return chunk_text(text, source=source, page=0)


def load_document(path: str) -> List[DocumentChunk]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in {".txt", ".md"}:
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: .pdf .txt .md")


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """ChromaDB-backed store with local sentence-transformer embeddings."""

    def __init__(self):
        self.embedder = LocalEmbedder()
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.metadata() for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        total = self.collection.count()
        if total == 0:
            return []
        n = min(top_k, total)
        query_emb = self.embedder.embed_one(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all chunks — used to build BM25 index."""
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            {"text": result["documents"][i], "metadata": result["metadatas"][i]}
            for i in range(len(result["documents"]))
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
