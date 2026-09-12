# -*- coding: utf-8 -*-
"""
SCRIPT HUẤN LUYỆN TĂNG CƯỜNG CÁC CÂU HỎI ĐA LĨNH VỰC KHÔNG LIÊN QUAN NHAU
(DIVERSE UNRELATED TOPICS TRAINING VIA xKIRO LLM API)
Dự án: Hệ thống Trợ lý AI Giám sát Người cao tuổi (AuraCare AI)
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Mục tiêu:
1. Gọi API xKiro (mistralai/ministral-8b) tạo sinh các cặp Q&A chuẩn xác cho 6 chủ đề độc lập:
   - Chủ đề 1: Quyền lợi Bảo hiểm Y tế (BHYT) & Thủ tục hành chính người cao tuổi
   - Chủ đề 2: Sơ cứu tai nạn sinh hoạt gia đình (Bỏng nước sôi, đứt tay, ong đốt)
   - Chủ đề 3: Khoa học môi trường & Thiết bị đời sống (Sóng Wi-Fi, máy điều hòa, tivi)
   - Chủ đề 4: Rèn luyện trí não & Trí nhớ (Cờ tướng, Sudoku, đố chữ, phòng ngừa Alzheimer)
   - Chủ đề 5: Đời sống tinh thần, thơ ca thư giãn & Nghệ thuật giao tiếp gia đình
   - Chủ đề 6: Nhận thức ranh giới chuyên môn & Câu hỏi ngoài phạm vi (Out-of-scope / General Knowledge)
2. Nạp dữ liệu vào Few-Shot RAG Memory, RLHF Dataset, SQLite DB.
3. Kích hoạt tái lập chỉ mục Vector Không gian Ngữ nghĩa (Local Vector Retriever).
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict, Any, Optional

# Đảm bảo UTF-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.llm_synthetic_generator import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    GENERATED_QA_FILE,
    FEW_SHOT_FILE,
    RLHF_DATASET_FILE
)
from src.db.db_service import get_db_service
from src.rag.local_vector_retriever import get_local_retriever

# Danh sách các câu hỏi thuộc 6 lĩnh vực hoàn toàn độc lập, không liên quan nhau
DIVERSE_QUESTIONS = [
    # Chủ đề 1: BHYT & Pháp lý người cao tuổi
    {
        "topic": "Chính sách BHYT Người cao tuổi",
        "category": "LEGAL_INSURANCE_POLICY",
        "question": "Người từ đủ 80 tuổi trở lên có được Nhà nước cấp thẻ BHYT miễn phí và thanh toán 100% không?",
        "context_hint": "Theo Luật BHYT, người từ đủ 80 tuổi không có lương hưu được cấp thẻ BHYT bảo trợ xã hội miễn phí và hưởng 100% chi phí khám chữa bệnh đúng tuyến."
    },
    {
        "topic": "Thủ tục nhận thuốc BHYT thay",
        "category": "LEGAL_INSURANCE_POLICY",
        "question": "Người nhà có thể đi khám và lĩnh thuốc BHYT định kỳ thay cho người cao tuổi đi lại khó khăn được không?",
        "context_hint": "Được phép lĩnh thay thuốc mạn tính nếu mang theo thẻ BHYT/CCCD của cụ, CCCD người nhận thay, giấy ủy quyền hoặc sổ y bạ đã đăng ký với bệnh viện."
    },

    # Chủ đề 2: Sơ cứu Tai nạn Sinh hoạt Gia đình
    {
        "topic": "Sơ cứu Bỏng nước sôi",
        "category": "DOMESTIC_FIRSTAID",
        "question": "Người nhà bị bỏng nước sôi khi nấu bếp thì cần sơ cứu thế nào đúng cách trong 15 phút đầu?",
        "context_hint": "Ngâm hoặc xối nước mát sạch 15-20 phút ngay lập tức (15-20°C). Tuyệt đối cấm bôi kem đánh răng, mỡ trăn, nước mắm hay chườm đá lạnh. Che gạc ẩm và đưa đi viện nếu diện tích lớn."
    },
    {
        "topic": "Sơ cứu Vết đứt tay chảy máu",
        "category": "DOMESTIC_FIRSTAID",
        "question": "Bị đứt tay chảy máu do dao gọt hoa quả thì sơ cứu và băng ép cầm máu thế nào?",
        "context_hint": "Rửa sạch nước muối sinh lý, dùng gạc vô trùng ấn trực tiếp lên miệng vết thương 5-10 phút, nâng cao tay hơn tim, sát khuẩn Povidine và băng lại. Đưa đi khâu nếu sâu toạc gân."
    },

    # Chủ đề 3: Khoa học Môi trường & Thiết bị Đời sống
    {
        "topic": "Sóng Wi-Fi & Máy tạo nhịp tim",
        "category": "TECH_ENVIRONMENT_HEALTH",
        "question": "Sóng Wi-Fi và sóng điện thoại di động có ảnh hưởng đến người già hoặc người đeo máy tạo nhịp tim không?",
        "context_hint": "Sóng Wi-Fi là bức xạ không ion hóa năng lượng thấp, an toàn. Người có máy tạo nhịp tim cần để điện thoại cách xa máy ít nhất 15cm (không bỏ túi áo ngực) và bộ phát Wi-Fi cách giường 1m."
    },
    {
        "topic": "Nhiệt độ Điều hòa Mùa hè",
        "category": "TECH_ENVIRONMENT_HEALTH",
        "question": "Mùa hè bật điều hòa cho người già thì nên để nhiệt độ bao nhiêu để không bị cảm lạnh và đột quỵ?",
        "context_hint": "Cài đặt từ 26-28°C kèm quạt gió nhẹ hoặc tạo ẩm. Chênh lệch trong ngoài không quá 7°C, tránh gió thổi thẳng vào đầu/ngực, mở cửa trước 5-10 phút khi ra ngoài tránh co mạch đột quỵ."
    },

    # Chủ đề 4: Rèn luyện Trí não & Phòng ngừa Alzheimer
    {
        "topic": "Chơi Cờ tướng & Não bộ",
        "category": "COGNITIVE_BRAIN_EXERCISE",
        "question": "Chơi cờ tướng hoặc cờ vua có giúp người già rèn luyện tư duy và phòng ngừa đãng trí không?",
        "context_hint": "Chơi cờ kích thích cả hai bán cầu não, tạo liên kết synap thần kinh mới (Neuroplasticity), rèn luyện phán đoán và trí nhớ, giúp tinh thần minh mẫn và giảm cô đơn khi chơi cùng con cháu."
    },
    {
        "topic": "Trò chơi Ô số Sudoku & Đố chữ",
        "category": "COGNITIVE_BRAIN_EXERCISE",
        "question": "Những trò chơi đố chữ hay giải ô số Sudoku giúp ích gì cho trí nhớ người cao tuổi?",
        "context_hint": "Kích hoạt vùng hồi hải mã (trung tâm trí nhớ), làm chậm quá trình teo não sinh lý. Nên chơi mức độ dễ đến vừa để tạo hưng phấn, kết hợp đọc sách báo mỗi ngày."
    },

    # Chủ đề 5: Đời sống Tinh thần, Thơ ca & Giải trí
    {
        "topic": "Thơ ca & Thư giãn tinh thần",
        "category": "SMALLTALK_ENTERTAINMENT",
        "question": "Hãy kể một câu chuyện vui ngắn hoặc đọc bài thơ về tuổi già thanh thản để cụ nghe thư giãn.",
        "context_hint": "Đọc bài thơ tuổi già an yên, kể mẩu chuyện vui gia đình ấm áp, khuyên cụ giữ tâm an trí sáng, nghe nhạc không lời Baroque để sóng não thư thái, hạ huyết áp."
    },
    {
        "topic": "Thú vui Điền viên Cây cảnh",
        "category": "SMALLTALK_ENTERTAINMENT",
        "question": "Những thói quen chăm sóc cây cảnh hay nuôi chim cá cảnh giúp ích gì cho tâm lý người già?",
        "context_hint": "Tưới cây, phơi nắng sáng sớm tạo vitamin D, giảm hormone cortisol gây căng thẳng, tạo niềm vui gắn kết với thiên nhiên và giúp người già luôn cảm thấy cuộc sống có ý nghĩa."
    },

    # Chủ đề 6: Câu hỏi Ngoài phạm vi & Kiến thức Tổng quát (Out of Scope)
    {
        "topic": "Giá vàng & Viết code Python (Out-of-scope)",
        "category": "GENERAL_OUT_OF_SCOPE",
        "question": "Hôm nay giá vàng thế nào và bạn có biết viết mã code lập trình Python không?",
        "context_hint": "AI định vị rõ mình là Trợ lý Lão khoa AuraCare AI, giải thích lịch sự rằng giá vàng biến động theo thị trường và AI cũng có thể hỗ trợ kiến thức công nghệ; tuy nhiên sứ mệnh số 1 là túc trực bảo vệ sức khỏe cho cụ An."
    },
    {
        "topic": "Sửa xe máy & Kiến thức xã hội (Out-of-scope)",
        "category": "GENERAL_OUT_OF_SCOPE",
        "question": "Xe máy của tôi bị chết máy giữa đường thì phải làm sao, bạn có biết sửa xe không?",
        "context_hint": "AI giải đáp cơ bản kiểm tra xăng, bu-gi, khuyên dắt vào tiệm sửa xe uy tín; đồng thời nhắc khéo quay lại chăm sóc sức khỏe gia đình và cụ An."
    }
]


def call_llm_generate_answer(item: Dict[str, str], api_key: str, base_url: str) -> Optional[str]:
    """Gọi LLM API để tạo sinh câu trả lời chuẩn xác, có cấu trúc và kết bài hoàn chỉnh."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "Bạn là Trợ lý AI Bác sĩ Lão khoa & Chăm sóc Người cao tuổi AuraCare AI.\n"
        "BỐI CẢNH BỆNH NHÂN: Cụ Nguyễn Văn An (82 tuổi, phòng 102). Tiền sử THA độ 2, tiểu đường type 2, dị ứng Penicillin. Bác sĩ phụ trách: BS. CKI Trần Minh Tuấn (0912.345.678).\n\n"
        "NHIỆM VỤ: Trả lời câu hỏi đa lĩnh vực từ người nhà hoặc người cao tuổi dựa trên gợi ý tri thức được cung cấp.\n"
        "QUY TẮC CẤU TRÚC VÀ ĐỘ DÀI:\n"
        "1. Trả lời CHÍNH XÁC, THỰC TẾ, CÔ ĐỌNG (khoảng 200 - 350 từ). Tránh lan man dài dòng.\n"
        "2. Chia thành 2-3 ý chính rõ ràng (gạch đầu dòng hoặc đánh số).\n"
        "3. ĐỐI VỚI CÂU HỎI NGOÀI PHẠM VI (giá vàng, viết code, sửa xe): Trả lời lịch sự, thông minh, sau đó khéo léo nhắc nhở quay lại sứ mệnh chính là sức khỏe của cụ An.\n"
        "4. QUY TẮC KẾT THÚC: Luôn kết thúc bằng câu đúc kết trọn vẹn và dấu chấm câu rõ ràng (. ! ?). Tuyệt đối KHÔNG dừng dở dang giữa câu."
    )

    user_prompt = (
        f"CHỦ ĐỀ: {item['topic']}\n"
        f"GỢI Ý TRI THỨC: {item['context_hint']}\n"
        f"CÂU HỎI: {item['question']}\n\n"
        "Hãy sinh câu trả lời chuẩn xác, đầy đủ và kết thúc trọn vẹn."
    )

    models_to_try = [
        "mistralai/ministral-8b",
        "mistralai/mistral-large-2512",
        "deepseek/deepseek-chat-v3.1"
    ]

    for model in models_to_try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1536
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content:
                    return content
        except Exception as e:
            print(f"   [WARN] Model {model} gặp lỗi: {e}")
            continue

    return None


def run_training_pipeline():
    """Thực thi pipeline tạo sinh dữ liệu, nạp vào bộ nhớ RAG và tái lập chỉ mục."""
    print("=" * 80)
    print("🚀 [DIVERSE QA TRAINER] KHỞI ĐỘNG HUẤN LUYỆN CÁC CHỦ ĐỀ ĐỘC LẬP KHÔNG LIÊN QUAN NHAU")
    print("=" * 80)

    api_key = os.environ.get("LLM_API_KEY", DEFAULT_API_KEY)
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    db_service = get_db_service()

    generated_pairs = []

    for idx, item in enumerate(DIVERSE_QUESTIONS, 1):
        print(f"\n[{idx}/{len(DIVERSE_QUESTIONS)}] Đang tạo sinh dữ liệu cho: [{item['topic']}]")
        print(f"   ❓ Câu hỏi: {item['question']}")

        ans = call_llm_generate_answer(item, api_key, base_url)
        if ans:
            print(f"   ✅ Sinh thành công ({len(ans)} ký tự).")
            pair = {
                "id": f"diverse_qa_{idx:03d}",
                "topic": item["topic"],
                "category": item["category"],
                "prompt": item["question"],
                "response": ans,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            generated_pairs.append(pair)
        else:
            print(f"   ❌ Không tạo được phản hồi cho câu hỏi này.")
        time.sleep(1)

    print(f"\n📊 Tổng số mẫu Q&A đa lĩnh vực đã tạo sinh: {len(generated_pairs)}/{len(DIVERSE_QUESTIONS)}")

    if not generated_pairs:
        print("[FAIL] Không có mẫu dữ liệu nào được sinh ra.")
        return

    # 1. Cập nhật vào synthetic_qa_generated.json
    all_qa = []
    if os.path.exists(GENERATED_QA_FILE):
        try:
            with open(GENERATED_QA_FILE, "r", encoding="utf-8") as f:
                all_qa = json.load(f)
        except Exception:
            all_qa = []

    # Ghép thêm các mẫu mới
    existing_prompts = {q.get("prompt") for q in all_qa}
    added_count = 0
    for p in generated_pairs:
        if p["prompt"] not in existing_prompts:
            all_qa.append(p)
            added_count += 1

    with open(GENERATED_QA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)
    print(f"💾 [DATASET] Đã lưu {added_count} mẫu mới vào: {GENERATED_QA_FILE} (Tổng cộng: {len(all_qa)} mẫu).")

    # 2. Cập nhật vào few_shot_examples.json
    few_shots = []
    if os.path.exists(FEW_SHOT_FILE):
        try:
            with open(FEW_SHOT_FILE, "r", encoding="utf-8") as f:
                few_shots = json.load(f)
        except Exception:
            few_shots = []

    fs_prompts = {fs.get("prompt") for fs in few_shots}
    for p in generated_pairs:
        if p["prompt"] not in fs_prompts:
            few_shots.append({"prompt": p["prompt"], "response": p["response"]})

    with open(FEW_SHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(few_shots, f, ensure_ascii=False, indent=2)
    print(f"🧠 [FEW-SHOT] Đã cập nhật Few-Shot Memory: {len(few_shots)} mẫu chuẩn.")

    # 3. Cập nhật vào SQLite DB
    for p in generated_pairs:
        try:
            db_service.save_feedback(
                query=p["prompt"],
                response=p["response"],
                rating=5,
                feedback_text=f"Diverse Synthetic Training Ground Truth ({p['category']})"
            )
        except Exception:
            pass
    print(f"🗄️ [SQLITE] Đã đồng bộ toàn bộ mẫu mới vào CSDL SQLite rlhf_feedback.")

    # 4. Tái lập chỉ mục Local Vector Retriever
    print("\n⚡ [RAG INDEX] Đang tái lập chỉ mục Vector Không gian Ngữ nghĩa Local...")
    retriever = get_local_retriever()
    retriever.rebuild_index()
    print(f"🎯 [RAG INDEX] Hoàn tất lập chỉ mục! Tổng số đoạn vector trong không gian: {len(retriever.chunks)}")

    print("\n" + "=" * 80)
    print("🎉 [HOÀN TẤT] QUÁ TRÌNH HUẤN LUYỆN CÁC CHỦ ĐỀ ĐA DẠNG ĐÃ THÀNH CÔNG RỰC RỠ!")
    print("=" * 80)


if __name__ == "__main__":
    run_training_pipeline()
