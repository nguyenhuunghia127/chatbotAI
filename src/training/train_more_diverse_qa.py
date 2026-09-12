# -*- coding: utf-8 -*-
"""
SCRIPT HUẤN LUYỆN TĂNG CƯỜNG ĐỢT 2: CÁC CHỦ ĐỀ ĐA LĨNH VỰC MỞ RỘNG (MORE DIVERSE TOPICS)
Dự án: Hệ thống Trợ lý AI Giám sát Người cao tuổi (AuraCare AI)
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

6 Chủ đề mới:
- Chủ đề 7: Dược lý & Tương tác Thảo dược Đông y với Thuốc Tây (Sâm, Tam thất, Lá vối)
- Chủ đề 8: Chăm sóc Vết loét Tì đè & Người già nằm lâu (Lật trở người, Đệm hơi)
- Chủ đề 9: Thị lực, Đục thủy tinh thể & Thiết bị Máy trợ thính
- Chủ đề 10: Đau nhức Xương khớp khi Trời lạnh & Phòng ngừa Co mạch Mùa đông
- Chủ đề 11: Dinh dưỡng Răng yếu & Chăm sóc Răng giả tháo lắp
- Chủ đề 12: Kỹ năng Ứng xử Gia đình & Câu đố dân gian rèn luyện Trí tuệ
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict, Any, Optional

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

NEW_DIVERSE_QUESTIONS = [
    # Chủ đề 7: Thảo dược Đông y & Thuốc Tây
    {
        "topic": "Nhân sâm & Bệnh huyết áp cao",
        "category": "HERBAL_DRUG_INTERACTION",
        "question": "Người đang uống thuốc hạ huyết áp và tiểu đường có được uống thêm nhân sâm hay tam thất không?",
        "context_hint": "Nhân sâm kích thích thần kinh và co mạch làm tăng huyết áp kịch phát. Tam thất hoạt huyết mạnh tăng nguy cơ chảy máu nếu đang dùng thuốc chống đông. Cần hỏi ý kiến BS Tuấn trước khi dùng."
    },
    {
        "topic": "Uống nước lá vối hàng ngày",
        "category": "HERBAL_DRUG_INTERACTION",
        "question": "Uống nước lá vối hoặc trà atiso hàng ngày thay nước lọc có làm tụt huyết áp hoặc hại thận người già không?",
        "context_hint": "Lá vối lợi tiểu nhẹ, uống quá đặc hoặc thay hoàn toàn nước lọc làm mất điện giải và hạ huyết áp quá mức. Nên uống loãng ban ngày, không uống sau 19h tránh tiểu đêm."
    },

    # Chủ đề 8: Loét tì đè & Nằm lâu
    {
        "topic": "Phòng ngừa Loét tì đè",
        "category": "BEDRIDDEN_PRESSURE_ULCER",
        "question": "Người già nằm một chỗ bị đỏ rát vùng mông và xương cùng thì phòng ngừa và xử lý lở loét tì đè thế nào?",
        "context_hint": "Lật trở người tối thiểu 2 giờ/lần, nằm nghiêng 30 độ có gối chèn. Dùng đệm hơi chống loét. Giữ da khô thoáng, bôi kem kẽm oxyd, không rắc bột kháng sinh lên vết loét."
    },
    {
        "topic": "Chăm sóc da và đệm hơi",
        "category": "BEDRIDDEN_PRESSURE_ULCER",
        "question": "Bao nhiêu tiếng cần lật trở người một lần và cách dùng đệm hơi chống loét cho người già nằm liệt ra sao?",
        "context_hint": "Lật trở người mỗi 2 tiếng (ngửa, nghiêng trái, nghiêng phải). Đệm hơi tự động bơm xả phân tán áp lực liên tục, vệ sinh da nhẹ nhàng bằng nước ấm sau mỗi lần tiêu tiểu."
    },

    # Chủ đề 9: Thị lực & Thính lực
    {
        "topic": "Đục thủy tinh thể người già",
        "category": "VISION_HEARING_CARE",
        "question": "Cụ già mắt nhìn mờ như có màn sương che, hay bị chói mắt thì có phải bị đục thủy tinh thể không và khi nào cần mổ?",
        "context_hint": "Đó là triệu chứng điển hình của đục thủy tinh thể (cườm khô). Phẫu thuật Phaco thay thủy tinh thể nhân tạo là giải pháp tối ưu khi thị lực cản trở sinh hoạt. Tăng độ sáng phòng ngừa té ngã."
    },
    {
        "topic": "Sử dụng máy trợ thính",
        "category": "VISION_HEARING_CARE",
        "question": "Cách chọn và bảo quản máy trợ thính cho người cao tuổi bị lãng tai nghễnh ngãng thế nào?",
        "context_hint": "Tháo máy và cất hộp hút ẩm ban đêm, lau sạch ráy tai ở núm tai nghe, không để dính nước. Khi nói chuyện đứng đối diện, nói chậm rãi rõ từ, không hét lớn vào tai."
    },

    # Chủ đề 10: Xương khớp trời lạnh
    {
        "topic": "Đau khớp khi trời lạnh",
        "category": "WEATHER_JOINT_SEASONAL",
        "question": "Tại sao mỗi khi trời trở lạnh hoặc sắp mưa thì các khớp gối của người già lại đau nhức buốt hơn?",
        "context_hint": "Do áp suất khí quyển giảm và nhiệt độ hạ thấp làm dịch khớp quánh đặc, mạch máu co lại giảm nuôi dưỡng khớp. Chườm ấm 15-20 phút, xoa bóp nhẹ, mặc quần ấm và đi tất len giữ ấm chân."
    },
    {
        "topic": "Phòng đột quỵ mùa đông",
        "category": "WEATHER_JOINT_SEASONAL",
        "question": "Mùa đông giữ ấm chân và phòng ngừa đột quỵ khi thức dậy nửa đêm cho người già thế nào?",
        "context_hint": "Không bước chân trần xuống nền nhà lạnh; ngồi mép giường 1-2 phút, xỏ dép có quai, mặc áo ấm trước khi đi vệ sinh. Rửa mặt bằng nước ấm, cấm nước lạnh buốt gây co mạch đột quỵ."
    },

    # Chủ đề 11: Răng yếu & Răng giả
    {
        "topic": "Chế biến đồ ăn cho răng yếu",
        "category": "DENTAL_DENTURE_NUTRITION",
        "question": "Cụ bị rụng nhiều răng, đeo hàm giả nhai khó thì chế biến thức ăn thế nào để vẫn đủ chất đạm và chất xơ?",
        "context_hint": "Băm nhuyễn thịt nấu cháo, hầm nhừ cá lọc xương, dùng đậu phụ mềm, trứng hấp, bí đỏ/khoai lang nghiền để cung cấp đủ protein chống teo cơ và chất xơ chống táo bón."
    },
    {
        "topic": "Tháo răng giả ban đêm",
        "category": "DENTAL_DENTURE_NUTRITION",
        "question": "Tại sao ban đêm bắt buộc phải tháo hàm răng giả và cách ngâm rửa hàm giả ra sao?",
        "context_hint": "Bắt buộc tháo hàm giả ban đêm để nướu nghỉ ngơi và phòng ngừa nuốt sặc hàm giả vào đường thở khi ngủ gây tử vong. Chải sạch bằng bàn chải mềm, ngâm vào cốc nước sạch hoặc dung dịch sát khuẩn."
    },

    # Chủ đề 12: Ứng xử gia đình & Đố vui dân gian
    {
        "topic": "Kỹ năng từ chối tự đi xe",
        "category": "SOCIAL_INTERACTIVE_RIDDLES",
        "question": "Làm thế nào khi người già nhất quyết đòi tự đi chợ bằng xe đạp cũ dù chân tay đã yếu?",
        "context_hint": "Không tranh cãi gay gắt hay giật chìa khóa. Dùng kỹ thuật chuyển hướng (Distraction): nói xe non hơi chưa bơm để chiều con chở đi, hoặc nhờ cụ việc nhẹ khác trong nhà để chuyển chú ý."
    },
    {
        "topic": "Câu đố dân gian kích thích trí não",
        "category": "SOCIAL_INTERACTIVE_RIDDLES",
        "question": "Kể cho tôi 3 câu đố vui dân gian nhẹ nhàng để ông bà cùng con cháu giải đố thư giãn vào buổi tối.",
        "context_hint": "Đố quả pháo (bằng quả cau làm cả làng vui), con trâu (bốn chân đạp đất hai sừng nhọn hoắt), tình mẹ mái nhà. Giúp khơi gợi ký ức xưa, rèn luyện não bộ và tạo tiếng cười sum vầy gia đình."
    }
]


def call_llm_generate(item: Dict[str, str], api_key: str, base_url: str) -> Optional[str]:
    """Gọi LLM API xKiro tạo sinh câu trả lời chuẩn xác và trọn vẹn."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "Bạn là Trợ lý AI Bác sĩ Lão khoa & Chăm sóc Người cao tuổi AuraCare AI.\n"
        "BỐI CẢNH BỆNH NHÂN: Cụ Nguyễn Văn An (82 tuổi, phòng 102). Tiền sử: THA độ 2, tiểu đường type 2, dị ứng Penicillin. Bác sĩ phụ trách: BS. CKI Trần Minh Tuấn (0912.345.678).\n\n"
        "NHIỆM VỤ: Trả lời câu hỏi đa lĩnh vực lão khoa dựa trên gợi ý tri thức được cung cấp.\n"
        "QUY TẮC CẤU TRÚC VÀ ĐỘ DÀI:\n"
        "1. Trả lời CHÍNH XÁC, THỰC TẾ, CÔ ĐỌNG (khoảng 200 - 350 từ).\n"
        "2. Chia thành 2-3 ý chính rõ ràng (gạch đầu dòng hoặc đánh số).\n"
        "3. LUÔN LUÔN kết thúc bằng câu đúc kết trọn vẹn và dấu chấm câu rõ ràng (. ! ?). Tuyệt đối KHÔNG dừng dở dang giữa câu."
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
        except Exception:
            continue

    return None


def run():
    print("=" * 80)
    print("🚀 [MORE DIVERSE QA TRAINER] KHỞI ĐỘNG HUẤN LUYỆN TĂNG CƯỜNG ĐỢT 2")
    print("=" * 80)

    api_key = os.environ.get("LLM_API_KEY", DEFAULT_API_KEY)
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    db_service = get_db_service()

    generated_pairs = []

    for idx, item in enumerate(NEW_DIVERSE_QUESTIONS, 1):
        print(f"\n[{idx}/{len(NEW_DIVERSE_QUESTIONS)}] Đang tạo sinh: [{item['topic']}]")
        print(f"   ❓ {item['question']}")

        ans = call_llm_generate(item, api_key, base_url)
        if ans:
            print(f"   ✅ Thành công ({len(ans)} ký tự).")
            pair = {
                "id": f"more_diverse_qa_{idx:03d}",
                "topic": item["topic"],
                "category": item["category"],
                "prompt": item["question"],
                "response": ans,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            generated_pairs.append(pair)
        else:
            print(f"   ❌ Lỗi sinh dữ liệu.")
        time.sleep(1)

    print(f"\n📊 Tổng số mẫu mới đã tạo: {len(generated_pairs)}/{len(NEW_DIVERSE_QUESTIONS)}")

    if not generated_pairs:
        print("[FAIL] Không có mẫu nào.")
        return

    # 1. Cập nhật synthetic_qa_generated.json
    all_qa = []
    if os.path.exists(GENERATED_QA_FILE):
        try:
            with open(GENERATED_QA_FILE, "r", encoding="utf-8") as f:
                all_qa = json.load(f)
        except Exception:
            all_qa = []

    existing_prompts = {q.get("prompt") for q in all_qa}
    added_count = 0
    for p in generated_pairs:
        if p["prompt"] not in existing_prompts:
            all_qa.append(p)
            added_count += 1

    with open(GENERATED_QA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)
    print(f"💾 [DATASET] Đã cập nhật {added_count} mẫu mới vào {GENERATED_QA_FILE} (Tổng cộng: {len(all_qa)} mẫu).")

    # 2. Cập nhật few_shot_examples.json
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
    print(f"🧠 [FEW-SHOT] Few-Shot Memory hiện có: {len(few_shots)} mẫu.")

    # 3. Đồng bộ SQLite DB
    for p in generated_pairs:
        try:
            db_service.save_feedback(
                query=p["prompt"],
                response=p["response"],
                rating=5,
                feedback_text=f"More Diverse Training Ground Truth ({p['category']})"
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
    print("🎉 [HOÀN TẤT] QUÁ TRÌNH TẠO SINH & NẠP TRI THỨC MỞ RỘNG THÀNH CÔNG!")
    print("=" * 80)


if __name__ == "__main__":
    run()
