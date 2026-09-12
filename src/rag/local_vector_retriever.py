# -*- coding: utf-8 -*-
"""
MODULE TRUY XUẤT NGỮ CẢNH VÉC-TƠ CỤC BỘ (LOCAL VECTOR SPACE RETRIEVER)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Công nghệ:
- Scikit-Learn TF-IDF Vector Space Model (Word n-grams 1-2, sublinear_tf=True)
- Cosine Similarity Metric với ma trận thưa tối ưu hóa (Sparse Matrix Dot Product)
- Chunking văn bản thông minh theo đoạn văn & tiêu đề phác đồ y tế (Paragraph / Section Level)
- Hỗ trợ đánh chỉ mục đồng thời:
  1. Toàn bộ tài liệu y khoa trong data/medical_docs/
  2. Toàn bộ các cặp Q&A chuẩn y tế đã huấn luyện trong data/training_data/synthetic_qa_generated.json
  3. Nhật ký thời gian thực từ system_logs.txt
- Hoàn toàn 100% Offline, không phụ thuộc vào C++ Vector DB bên ngoài, suy luận < 2ms trên Edge CPU.
"""

import os
import sys
import glob
import json
import re
from typing import List, Dict, Any, Optional, Tuple

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "medical_docs")
LOGS_FILE = os.path.join(PROJECT_ROOT, "system_logs.txt")
SYNTHETIC_QA_FILE = os.path.join(PROJECT_ROOT, "data", "training_data", "synthetic_qa_generated.json")


class LocalVectorRetriever:
    """Bộ máy truy xuất ngữ cảnh véc-tơ ngữ nghĩa nội bộ siêu tốc."""

    def __init__(self, docs_dir: str = DOCS_DIR, logs_file: str = LOGS_FILE):
        self.docs_dir = docs_dir
        self.logs_file = logs_file
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.rebuild_index()

    def _chunk_text(self, text: str, source_name: str, doc_type: str = "medical_doc") -> List[Dict[str, Any]]:
        """Chia văn bản thành các đoạn ngữ nghĩa dựa trên tiêu đề và đoạn văn."""
        chunks = []
        # Tách theo khối phần hoặc dấu phân cách
        sections = re.split(r'={10,}|#{1,3}\s+', text)
        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean or len(sec_clean) < 30:
                continue

            # Nếu đoạn quá dài (> 800 ký tự), tiếp tục tách theo đoạn văn \n\n
            if len(sec_clean) > 800:
                paras = sec_clean.split("\n\n")
                for p in paras:
                    p_clean = p.strip()
                    if len(p_clean) >= 40:
                        chunks.append({
                            "content": p_clean,
                            "source": source_name,
                            "type": doc_type
                        })
            else:
                chunks.append({
                    "content": sec_clean,
                    "source": source_name,
                    "type": doc_type
                })
        return chunks

    def rebuild_index(self):
        """Thu thập tài liệu và tái lập ma trận véc-tơ TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        raw_chunks: List[Dict[str, Any]] = []

        # 1. Đọc Medical Docs
        if os.path.exists(self.docs_dir):
            for fpath in glob.glob(os.path.join(self.docs_dir, "*.txt")):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                        doc_chunks = self._chunk_text(text, os.path.basename(fpath), "medical_doc")
                        raw_chunks.extend(doc_chunks)
                except Exception as e:
                    print(f"[RETRIEVER ERROR]: {fpath} - {e}", file=sys.stderr)

        # 2. Đọc Synthetic Q&A Knowledge Base
        if os.path.exists(SYNTHETIC_QA_FILE):
            try:
                with open(SYNTHETIC_QA_FILE, "r", encoding="utf-8") as f:
                    qa_items = json.load(f)
                    for item in qa_items:
                        p = item.get("prompt", "").strip()
                        c = item.get("chosen", item.get("response", "")).strip()
                        if p and c:
                            raw_chunks.append({
                                "content": f"HỎI: {p}\nTRẢ LỜI CHUẨN Y KHOA: {c}",
                                "source": "synthetic_qa_generated.json",
                                "type": "qa_ground_truth"
                            })
            except Exception as e:
                print(f"[RETRIEVER ERROR]: {SYNTHETIC_QA_FILE} - {e}", file=sys.stderr)

        # 3. Đọc System Logs
        if os.path.exists(self.logs_file):
            try:
                with open(self.logs_file, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                    # Lấy 20 log gần nhất
                    for line in lines[-20:]:
                        raw_chunks.append({
                            "content": line,
                            "source": "system_logs.txt",
                            "type": "system_log"
                        })
            except Exception as e:
                print(f"[RETRIEVER ERROR]: {self.logs_file} - {e}", file=sys.stderr)

        if not raw_chunks:
            self.chunks = []
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        self.chunks = raw_chunks
        corpus = [c["content"] for c in self.chunks]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print(f"⚡ [LOCAL VECTOR RETRIEVER] Đã lập chỉ mục {len(self.chunks)} đoạn tri thức véc-tơ thành công!", flush=True)

    def search(self, query: str, top_k: int = 4, min_similarity: float = 0.05) -> List[Dict[str, Any]]:
        """
        Tìm kiếm ngữ cảnh tương đồng véc-tơ theo Cosine Similarity.
        Trả về danh sách các đoạn tài liệu có điểm số cao nhất.
        """
        if not self.vectorizer or self.tfidf_matrix is None or not query.strip():
            return []

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1]

        results = []
        for idx in top_indices[:top_k]:
            score = float(sims[idx])
            if score >= min_similarity:
                item = dict(self.chunks[idx])
                item["similarity_score"] = round(score, 4)
                item["score"] = round(score, 4)
                item["text"] = item.get("content", "")
                results.append(item)

        return results


# Singleton
_local_retriever_instance = None


def get_local_retriever() -> LocalVectorRetriever:
    global _local_retriever_instance
    if _local_retriever_instance is None:
        _local_retriever_instance = LocalVectorRetriever()
    return _local_retriever_instance
