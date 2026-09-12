# -*- coding: utf-8 -*-
"""
Script tạo sinh dữ liệu câu hỏi hay gặp (FAQ) qua xKiro LLM API
"""
import sys
import os

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

from src.training.llm_synthetic_generator import get_llm_generator

FAQ_TOPICS = [
    "nutrition_diet_restriction",
    "bathing_hygiene_safety",
    "emotional_mental_sleep",
    "exercise_physiotherapy",
    "device_sos_support"
]

def main():
    gen = get_llm_generator()
    all_new = []
    for topic in FAQ_TOPICS:
        print(f"\n🚀 Đang tạo sinh dữ liệu cho chủ đề FAQ: {topic}...")
        samples = gen.generate_qa_pairs(topic_key=topic, num_samples=3)
        print(f"-> Thu được {len(samples)} mẫu.")
        for s in samples:
            s["topic"] = topic
        all_new.extend(samples)

    print(f"\n📊 Tổng số mẫu FAQ mới: {len(all_new)}. Bắt đầu đồng bộ vào hệ thống...")
    res = gen.integrate_into_knowledge_base(all_new)
    print("Kết quả đồng bộ:", res)

if __name__ == "__main__":
    main()
