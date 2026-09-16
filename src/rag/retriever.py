# -*- coding: utf-8 -*-
"""
MODULE HYBRID RETRIEVER (DENSE & SPARSE RETRIEVAL COORDINATOR)
Dự án: Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi AuraCare
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import importlib
from typing import List, Dict, Any, Optional

from src.core.config import (
    SYSTEM_LOGS_PATH,
    MEDICAL_DOCS_DIR,
    CHROMA_PERSIST_DIR,
    DEFAULT_EMBEDDING_MODEL
)

# Nạp động LangChain & ChromaDB
HAS_LANGCHAIN = False
Chroma = None
HuggingFaceEmbeddings = None
RecursiveCharacterTextSplitter = None
Document = None

try:
    chroma_mod = importlib.import_module("langchain_community.vectorstores")
    Chroma = getattr(chroma_mod, "Chroma", None)

    hf_mod = importlib.import_module("langchain_huggingface")
    HuggingFaceEmbeddings = getattr(hf_mod, "HuggingFaceEmbeddings", None)

    splitter_mod = importlib.import_module("langchain_text_splitters")
    RecursiveCharacterTextSplitter = getattr(splitter_mod, "RecursiveCharacterTextSplitter", None)

    doc_mod = importlib.import_module("langchain_core.documents")
    Document = getattr(doc_mod, "Document", None)

    if Chroma and HuggingFaceEmbeddings and Document:
        HAS_LANGCHAIN = True
except Exception:
    try:
        chroma_mod = importlib.import_module("langchain.vectorstores")
        Chroma = getattr(chroma_mod, "Chroma", None)

        hf_mod = importlib.import_module("langchain.embeddings")
        HuggingFaceEmbeddings = getattr(hf_mod, "HuggingFaceEmbeddings", None)

        splitter_mod = importlib.import_module("langchain.text_splitter")
        RecursiveCharacterTextSplitter = getattr(splitter_mod, "RecursiveCharacterTextSplitter", None)

        doc_mod = importlib.import_module("langchain.docstore.document")
        Document = getattr(doc_mod, "Document", None)

        if Chroma and HuggingFaceEmbeddings and Document:
            HAS_LANGCHAIN = True
    except Exception:
        HAS_LANGCHAIN = False


class HybridRetriever:
    """Điều phối truy xuất đa tầng kết hợp Dense Vector (Chroma) và Sparse Text (TF-IDF/Cache)."""

    def __init__(
        self,
        logs_file: str = str(SYSTEM_LOGS_PATH),
        docs_dir: str = str(MEDICAL_DOCS_DIR),
        chroma_dir: str = str(CHROMA_PERSIST_DIR),
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    ):
        self.logs_file = logs_file
        self.docs_dir = docs_dir
        self.chroma_dir = chroma_dir
        self.embedding_model_name = embedding_model_name

        self.vector_store = None
        self.embeddings = None
        self.documents_cache: List[Dict[str, Any]] = []

        self._initialize()

    def _initialize(self):
        """Khởi tạo kho véc tơ và bộ nhớ cache."""
        try:
            if HAS_LANGCHAIN:
                os.makedirs(self.chroma_dir, exist_ok=True)
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=self.embedding_model_name,
                        model_kwargs={"device": "cpu"}
                    )
                except Exception as e_emb:
                    self.embeddings = None

                self.rebuild_vector_db()
            else:
                self.load_documents_cache()
        except Exception:
            self.load_documents_cache()

    def load_documents_cache(self):
        """Nạp toàn bộ tài liệu y tế và logs vào cache bộ nhớ cho Edge Fail-safe Engine."""
        self.documents_cache = []

        # 1. Đọc Medical Docs
        if os.path.exists(self.docs_dir):
            for filepath in glob.glob(os.path.join(self.docs_dir, "*.txt")):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        self.documents_cache.append({
                            "source": os.path.basename(filepath),
                            "type": "medical_doc",
                            "content": content
                        })
                except Exception:
                    pass

        # 2. Đọc System Logs
        if os.path.exists(self.logs_file):
            try:
                with open(self.logs_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                    for line in lines:
                        self.documents_cache.append({
                            "source": "system_logs.txt",
                            "type": "system_log",
                            "content": line
                        })
            except Exception:
                pass

    def rebuild_vector_db(self):
        """Đọc system_logs.txt và medical_docs để nạp/cập nhật vào ChromaDB."""
        if not HAS_LANGCHAIN or not self.embeddings:
            self.load_documents_cache()
            return

        try:
            docs = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            if os.path.exists(self.docs_dir):
                for filepath in glob.glob(os.path.join(self.docs_dir, "*.txt")):
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            docs.append(Document(
                                page_content=chunk,
                                metadata={"source": os.path.basename(filepath), "type": "medical"}
                            ))

            if os.path.exists(self.logs_file):
                with open(self.logs_file, "r", encoding="utf-8") as f:
                    log_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                    for line in log_lines:
                        docs.append(Document(
                            page_content=line,
                            metadata={"source": "system_logs.txt", "type": "log"}
                        ))

            if docs:
                self.vector_store = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    persist_directory=self.chroma_dir
                )
        except Exception:
            self.load_documents_cache()

    def get_recent_logs(self, n: int = 8) -> str:
        """Trích xuất n dòng log mới nhất từ system_logs.txt."""
        if not os.path.exists(self.logs_file):
            return "Chưa có dữ liệu nhật ký hệ thống."

        try:
            with open(self.logs_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                recent = lines[-n:] if len(lines) >= n else lines
                return "\n".join(recent)
        except Exception as e:
            return f"Không thể đọc nhật ký: {str(e)}"

    def retrieve_candidates(self, queries: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Quét và gom ứng viên tài liệu từ ChromaDB, Local TF-IDF và In-Memory Cache."""
        raw_candidates = []

        for q in queries:
            # 1. ChromaDB (Dense Vector Search)
            if HAS_LANGCHAIN and self.vector_store:
                try:
                    results = self.vector_store.similarity_search(q, k=top_k)
                    for doc in results:
                        raw_candidates.append({
                            "text": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "score": 0.45
                        })
                except Exception:
                    pass

            # 2. Local TF-IDF (Sparse Vector Search)
            try:
                from src.rag.local_vector_retriever import get_local_retriever
                local_retriever = get_local_retriever()
                local_results = local_retriever.search(q, top_k=top_k)
                for res in local_results:
                    if res.get("score", 0) > 0.03:
                        raw_candidates.append({
                            "text": res["text"],
                            "source": res["source"],
                            "score": float(res.get("score", 0.1))
                        })
            except Exception:
                pass

        # Thu thập thêm từ memory cache nếu thiếu ứng viên
        if len(raw_candidates) < top_k:
            self.load_documents_cache()
            for doc in self.documents_cache[:8]:
                raw_candidates.append({
                    "text": doc["content"],
                    "source": doc.get("source", "cache"),
                    "score": 0.05
                })

        # Loại bỏ trùng lặp
        seen_texts = set()
        unique_candidates = []
        for cand in raw_candidates:
            snippet = cand["text"].strip()[:100]
            if snippet not in seen_texts:
                seen_texts.add(snippet)
                unique_candidates.append(cand)

        return unique_candidates


_retriever_instance = None

def get_hybrid_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
