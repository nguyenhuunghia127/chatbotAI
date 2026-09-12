# -*- coding: utf-8 -*-
"""
SCRIPT HUẤN LUYỆN TĂNG CƯỜNG DỮ LIỆU CÂU HỎI THƯỜNG NGÀY & ĐỜI SỐNG CHO CHATBOT (AuraCare AI)
Kết nối xKiro API tạo sinh dữ liệu, nạp vào CSDL SQLite, Vector Store & RAG Engine.
"""

import os
import sys
import time
import json

# Tương thích UTF-8 console Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.llm_synthetic_generator import get_llm_generator, GENERATION_TOPICS
from src.rag.rag_engine import get_rag_engine

DAILY_TOPICS = [
    "daily_routine_schedule",
    "daily_sleep_insomnia",
    "daily_hygiene_denture_bath",
    "daily_drinking_tea_coffee",
    "daily_elderly_psychology_empathy",
    "daily_chit_chat_companion"
]


def train_daily_questions():
    print("=" * 80)
    print("🌟 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN DỮ LIỆU ĐỜI SỐNG & CÂU HỎI THƯỜNG NGÀY CHO CHATBOT")
    print("=" * 80)

    gen = get_llm_generator()
    all_generated_samples = []

    for idx, topic_key in enumerate(DAILY_TOPICS, 1):
        topic_meta = GENERATION_TOPICS.get(topic_key, {})
        title = topic_meta.get("title", topic_key)
        print(f"\n[{idx}/{len(DAILY_TOPICS)}] 🚀 Đang tạo sinh dữ liệu chủ đề: '{title}'...")

        try:
            samples = gen.generate_qa_pairs(topic_key=topic_key, num_samples=3, generate_dpo=True)
            if samples:
                for s in samples:
                    s["topic"] = topic_key
                    s["category"] = "daily_routine"
                all_generated_samples.extend(samples)
                print(f"   -> Thu được {len(samples)} mẫu chuẩn y khoa & đời sống.")
            else:
                print(f"   ⚠️ Không lấy được mẫu từ API cho chủ đề {topic_key}.")
        except Exception as e:
            print(f"   ❌ Lỗi khi tạo sinh chủ đề {topic_key}: {e}")

        time.sleep(1)

    print(f"\n📊 Tổng số mẫu dữ liệu thường ngày đã tạo sinh: {len(all_generated_samples)} mẫu.")

    # Tích hợp vào hệ thống: Few-Shot Memory, CSDL SQLite, và rlhf_dataset.json
    print("\n💾 Đang tích hợp dữ liệu vào CSDL SQLite và Bộ nhớ Few-Shot RAG...")
    integration_res = gen.integrate_into_knowledge_base(all_generated_samples)
    print(f"✅ Kết quả tích hợp: {integration_res}")

    # Làm mới và huấn luyện nạp vào Vector Database của RAG Engine
    print("\n🧠 Đang tái tạo chỉ mục Vector Embeddings (Vector Retriever Indexing)...")
    from src.rag.local_vector_retriever import get_local_retriever
    local_ret = get_local_retriever()
    local_ret.rebuild_index()
    print(f"✅ Đã nạp và lập chỉ mục véc-tơ thành công cho {len(local_ret.chunks)} khối tri thức y khoa & đời sống!")

    rag = get_rag_engine()
    rag._load_local_documents_cache()
    if hasattr(rag, "_init_vector_store"):
        try:
            rag._init_vector_store()
            print("✅ ChromaDB Store đã được tái huấn luyện và đồng bộ thành công!")
        except Exception as e:
            pass

    print("\n🎉 HOÀN TẤT HUẤN LUYỆN! Bắt đầu kiểm thử suy luận của AI với các câu hỏi thực tế...")


def verify_inference():
    print("\n" + "=" * 80)
    print("🧪 KIỂM THỬ SUY LUẬN TỰ ĐỘNG CỦA CHATBOT SAU KHI HỌC:")
    print("=" * 80)

    rag = get_rag_engine()
    test_queries = [
        "chào em, hôm nay trời mưa lạnh cụ An có nên ra ngoài sân đi dạo không?",
        "cụ An thích uống trà đặc buổi sáng có sao không?",
        "cụ trằn trọc khó ngủ về đêm thì làm thế nào mà không dùng thuốc ngủ?",
        "cụ An lười tắm vì sợ lạnh thì tôi nên làm thế nào?",
        "cụ hay quên chìa khóa và kính lão, dạo này cụ hay buồn thì nên trò chuyện thế nào?"
    ]

    for q in test_queries:
        print(f"\n👉 CÂU HỎI: \"{q}\"")
        res = rag.query(q)
        print(f"⚙️  ENGINE: {res.get('engine')}")
        print(f"📚 NGUỒN TRI THỨC: {res.get('sources')}")
        ans = res.get('answer', '')
        # Hiển thị 250 ký tự đầu của câu trả lời
        preview = ans[:300] + "..." if len(ans) > 300 else ans
        print(f"💬 TRẢ LỜI:\n{preview}\n")


if __name__ == "__main__":
    train_daily_questions()
    verify_inference()
