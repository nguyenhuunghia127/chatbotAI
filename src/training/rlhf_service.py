# -*- coding: utf-8 -*-
"""
MODULE HUẤN LUYỆN TĂNG CƯỜNG DỰA TRÊN PHẢN HỒI NGƯỜI DÙNG (RLHF & ACTIVE LEARNING SERVICE)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Mục tiêu:
1. Thu thập phản hồi người dùng (Thumbs Up 👍 / Thumbs Down 👎 / Câu trả lời chuẩn).
2. Lưu trữ cặp dữ liệu DPO (Direct Preference Optimization): (Prompt, Chosen, Rejected).
3. Cơ chế Tăng Cường Ngữ Cảnh Tức Thì (Dynamic In-Context Few-Shot Reinforcement):
   - Tự động nạp các câu trả lời đạt điểm thưởng cao (+1) vào bộ nhớ RAG làm mẫu học Few-Shot.
   - Con bot thông minh lên tức thì sau mỗi lần người dùng đánh giá mà không cần đợi train lại mô hình lớn.
4. Tự động xuất Dataset chuẩn định dạng HuggingFace / Alpaca / DPO phục vụ Fine-Tuning LoRA.
"""

import os
import sys
import json
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "elderly_care.db")
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "training_data")
FEW_SHOT_FILE = os.path.join(TRAINING_DATA_DIR, "few_shot_examples.json")
RLHF_DATASET_FILE = os.path.join(TRAINING_DATA_DIR, "rlhf_dataset.json")


class RLHFService:
    """Quản lý Huấn luyện Tăng cường, Phản hồi Người dùng & Xuất Dataset."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Khởi tạo bảng rlhf_feedback."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rlhf_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    rating INTEGER NOT NULL,          -- +1 (Hài lòng) hoặc -1 (Chưa hài lòng)
                    corrected_response TEXT,         -- Câu trả lời chuẩn do người dùng sửa
                    feedback_text TEXT,              -- Ghi chú góp ý
                    source TEXT DEFAULT 'web_chat',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()

    def record_feedback(
        self,
        prompt: str,
        response: str,
        rating: int,
        corrected_response: str = "",
        feedback_text: str = "",
        source: str = "web_chat"
    ) -> Dict[str, Any]:
        """
        Ghi nhận phản hồi người dùng (+1 hoặc -1).
        Nếu rating == +1 hoặc có corrected_response -> đưa vào tập tri thức Few-Shot để tăng cường ngay.
        """
        clean_prompt = prompt.strip()
        clean_resp = response.strip()
        clean_corrected = corrected_response.strip()

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rlhf_feedback (prompt, response, rating, corrected_response, feedback_text, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (clean_prompt, clean_resp, rating, clean_corrected, feedback_text.strip(), source))
            feedback_id = cursor.lastrowid
            conn.commit()
            conn.close()

        # Cập nhật file Few-Shot và Dataset
        self.export_datasets()

        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Đã ghi nhận phản hồi tăng cường thành công!"
        }

    def get_dynamic_few_shot_examples(self, limit: int = 4) -> List[Dict[str, str]]:
        """
        Lấy các mẫu Q&A có điểm thưởng cao nhất (hoặc đã được người dùng chỉnh sửa chuẩn)
        để chèn trực tiếp vào System Prompt của RAG Engine (Few-Shot Prompting).
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prompt, COALESCE(NULLIF(corrected_response, ''), response) as best_response
                FROM rlhf_feedback
                WHERE rating = 1 OR (corrected_response IS NOT NULL AND corrected_response != '')
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            conn.close()

        examples = []
        for r in rows:
            examples.append({
                "question": r["prompt"],
                "answer": r["best_response"]
            })
        return examples

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Thống kê chỉ số phản hồi và hiệu quả tăng cường."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rlhf_feedback")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM rlhf_feedback WHERE rating = 1")
            positive = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM rlhf_feedback WHERE rating = -1")
            negative = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM rlhf_feedback WHERE corrected_response IS NOT NULL AND corrected_response != ''")
            corrected = cursor.fetchone()[0]
            conn.close()

        satisfaction_rate = round((positive / total * 100), 1) if total > 0 else 100.0

        return {
            "total_feedbacks": total,
            "positive_likes": positive,
            "negative_dislikes": negative,
            "user_corrected_count": corrected,
            "satisfaction_rate_percent": satisfaction_rate
        }

    def export_datasets(self):
        """Xuất dữ liệu phản hồi ra các định dạng chuẩn DPO và Alpaca Fine-Tuning."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rlhf_feedback ORDER BY id ASC")
            rows = cursor.fetchall()
            conn.close()

        alpaca_dataset = []
        dpo_dataset = []
        few_shots = []

        for r in rows:
            prompt = r["prompt"]
            resp = r["response"]
            rating = r["rating"]
            corrected = r["corrected_response"]

            # Alpaca Instruction Format
            gold_output = corrected if corrected else resp
            if rating == 1 or corrected:
                alpaca_dataset.append({
                    "instruction": "Bạn là Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi. Hãy trả lời ngắn gọn, chuẩn xác.",
                    "input": prompt,
                    "output": gold_output
                })
                few_shots.append({
                    "question": prompt,
                    "answer": gold_output
                })

            # DPO Format (Prompt, Chosen, Rejected)
            if rating == -1 and corrected:
                dpo_dataset.append({
                    "prompt": prompt,
                    "chosen": corrected,
                    "rejected": resp
                })

        try:
            with open(RLHF_DATASET_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "alpaca_fine_tuning": alpaca_dataset,
                    "dpo_preference_pairs": dpo_dataset,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)

            with open(FEW_SHOT_FILE, "w", encoding="utf-8") as f:
                json.dump(few_shots[-10:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[RLHF EXPORT ERROR]: {e}", flush=True)


# Singleton
rlhf_service_instance = None


def get_rlhf_service() -> RLHFService:
    global rlhf_service_instance
    if rlhf_service_instance is None:
        rlhf_service_instance = RLHFService()
    return rlhf_service_instance
