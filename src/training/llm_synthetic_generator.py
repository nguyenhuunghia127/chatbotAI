# -*- coding: utf-8 -*-
"""
MODULE TẠO SINH DỮ LIỆU HUẤN LUYỆN TĂNG CƯỜNG BẰNG LLM API (LLM SYNTHETIC DATA GENERATOR)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi (AuraCare AI)
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Mục đích:
1. Kết nối với các API LLM (OpenAI, DeepSeek, OpenRouter, Groq, hoặc Proxy tương thích).
2. Tự động sinh dữ liệu đối thoại Y tế & Giám sát Lão khoa (Synthetic Prompt & Response).
3. Hỗ trợ tạo cả dữ liệu huấn luyện thông thường (Prompt - Response) và cặp DPO (Prompt - Chosen - Rejected).
4. Tự động nạp dữ liệu sinh ra vào:
   - Few-Shot Memory (`few_shot_examples.json`) -> Chatbot thông minh lên tức thì.
   - RLHF Training Dataset (`rlhf_dataset.json`) -> Dùng để Fine-tune LoRA/QLoRA.
   - Auto-Trainer Knowledge Base -> Huấn luyện ngầm tự động liên tục.
"""

import os
import sys
import re
import json
import time
import requests
from typing import List, Dict, Any, Optional

# Đảm bảo UTF-8
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
TRAINING_DIR = os.path.join(PROJECT_ROOT, "data", "training_data")
FEW_SHOT_FILE = os.path.join(TRAINING_DIR, "few_shot_examples.json")
RLHF_DATASET_FILE = os.path.join(TRAINING_DIR, "rlhf_dataset.json")
GENERATED_QA_FILE = os.path.join(TRAINING_DIR, "synthetic_qa_generated.json")

os.makedirs(TRAINING_DIR, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

# API Key & Base URL được tải an toàn từ biến môi trường hoặc file .env
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "")
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.xkiro.com/v1")
DEFAULT_MODEL = "mistralai/ministral-8b"

# Danh sách chủ đề y tế & giám sát người cao tuổi chuyên sâu
GENERATION_TOPICS = {
    "fall_emergency": {
        "title": "Sự cố Té ngã & Xử lý Khẩn cấp",
        "description": "Các tình huống người già bị ngã trong nhà tắm, phòng khách, cầu thang; cách nhận biết qua âm thanh va đập, góc nghiêng cơ thể; sơ cứu gãy xương hông và gọi cấp cứu 115."
    },
    "fever_infection": {
        "title": "Thân nhiệt Bất thường & Sốt cao ở Người già",
        "description": "Tình trạng sốt >38°C hoặc hạ thân nhiệt <35°C; nguy cơ nhiễm trùng tiết niệu, viêm phổi; phác đồ bù nước oresol, chườm ấm vùng trán/nách/bẹn."
    },
    "cardiac_anomalies": {
        "title": "Rối loạn Nhịp tim & Huyết áp Cấp tính",
        "description": "Nhịp tim tăng nhanh >120 BPM hoặc giảm chậm <50 BPM; tụt huyết áp tư thế khi đổi vị trí đột ngột; tư thế ngồi Fowler 45 độ; dấu hiệu nhồi máu cơ tim."
    },
    "stroke_fast_protocol": {
        "title": "Nhận diện & Sơ cứu Đột Quỵ Não (Phác đồ FAST)",
        "description": "Dấu hiệu FAST: Face (méo miệng/lệch nhân trung), Arms (yếu liệt tay chân một bên), Speech (nói đớ/khó nói/vô ngôn), Time (giờ vàng tiêu sợi huyết <4.5 giờ); tư thế nằm nghiêng an toàn, tuyệt đối không cạo gió hay chích lễ."
    },
    "hypoglycemia_emergency": {
        "title": "Xử trí Cơn Hạ Đường Huyết Cấp tính",
        "description": "Nhận biết hạ đường huyết ở bệnh nhân tiểu đường Type 2: vã mồ hôi lạnh, run tay chân, đói cồn cào, tim đập nhanh, lơ mơ; quy tắc 15-15 (uống 15g đường nhanh/nước ép, kiểm tra lại sau 15 phút)."
    },
    "choking_airway_obstruction": {
        "title": "Xử trí Sặc Nghẹn & Hóc Dị Vật Đường Thở",
        "description": "Người cao tuổi nghẹn thức ăn, sặc thuốc, khó thở tím tái, ôm cổ; thủ thuật Heimlich tư thế ngồi/đứng hoặc vỗ lưng ấn ngực cho người già yếu; gọi 115 ngay khi bất tỉnh."
    },
    "drug_safety_penicillin_alert": {
        "title": "An toàn Dùng thuốc & Chống Dị ứng Kháng sinh Penicillin",
        "description": "Tuyệt đối không dùng kháng sinh nhóm Beta-lactam (Penicillin, Amoxicillin, Augmentin, Ampicillin); phối hợp an toàn thuốc huyết áp và thuốc hạ đường huyết; tránh dùng thuốc lúc đói."
    },
    "dementia_sundowning_agitation": {
        "title": "Sa sút Trí tuệ & Hội chứng Hoàng hôn (Sundowning)",
        "description": "Người già lú lẫn, kích động, đòi về nhà lúc chiều tối/ban đêm; kỹ thuật giao tiếp thấu cảm, bật đèn dịu nhẹ, hạn chế tiếng ồn; phòng ngừa đi lạc và khóa cửa an toàn."
    },
    "voice_distress_cries": {
        "title": "Phát hiện Tiếng Kêu cứu & Âm thanh Bất thường qua Micro",
        "description": "Micro thu được tiếng kêu 'Cứu tôi với', tiếng rên rỉ đau đớn, tiếng rơi vỡ chén đĩa; AI lập tức phân tích, kích hoạt còi báo động trạm và thông báo cho người giám hộ."
    },
    "identity_patient_profile": {
        "title": "Danh tính Người hỏi & Tra cứu Bệnh án Cụ già",
        "description": "Xác định vai trò người bảo hộ (Nguyễn Hữu Nghĩa - Con trai trưởng); tra cứu tiền sử Cụ Nguyễn Văn An (82 tuổi, phòng 102, tiểu đường, huyết áp, dị ứng Penicillin)."
    },
    "daily_care_safety": {
        "title": "Chăm sóc Sinh hoạt Hàng ngày & Phòng ngừa Tai nạn",
        "description": "Lắp thanh vịn nhà vệ sinh, đèn cảm biến ban đêm, thảm chống trượt; chế độ dinh dưỡng giảm muối cho người huyết áp cao; nhắc nhở uống thuốc đúng giờ."
    },
    "hypertensive_crisis": {
        "title": "Cơn Tăng Huyết Áp Kịch Phát & Đau Đầu Dữ Dội",
        "description": "Huyết áp vọt cao >=180/120 mmHg, đau đầu sau gáy, hoa mắt, tức ngực; tư thế ngồi nghỉ 45 độ, không tự uống liều hạ áp cực mạnh đột ngột, gọi 115 nếu có tổn thương cơ quan đích."
    },
    "dehydration_heatstroke": {
        "title": "Say Nắng, Sốc Nhiệt & Bù Nước Chống Mất Nước",
        "description": "Người già giảm cảm giác khát, da khô nhăn, tiểu vàng sẫm, tụt huyết áp tư thế; phác đồ bù oresol, nước ấm chia nhỏ ngụm mỗi 1-2 giờ, tránh sốc nhiệt ngày hè."
    },
    "bedsores_ulcer_prevention": {
        "title": "Dự Phòng & Chăm Sóc Loét Tì Đè Người Nằm Liệt Giường",
        "description": "Loét xương cùng cụt, gót chân ở người già nằm một chỗ; quy tắc xoay trở tư thế mỗi 2 giờ, đệm hơi đảo áp lực, cấm xoa bóp vùng da đỏ, bổ sung đạm và vitamin C tái tạo da."
    },
    "polypharmacy_medication_compliance": {
        "title": "Quản Lý Đa Thuốc & Xử Trí Quên Liều / Uống Nhầm",
        "description": "Hộp chia thuốc 4 buổi/ngày, tránh tương tác nguy hiểm giữa thuốc hạ áp và thuốc tiểu đường; xử trí khi quên liều (tuyệt đối không uống gấp đôi liều bù)."
    },
    "cpr_resuscitation": {
        "title": "Hồi Sinh Tim Phổi (CPR) & Cấp Cứu Ngừng Tuần Hoàn",
        "description": "Nhận diện người cao tuổi bất tỉnh, ngừng thở; gọi 115 bật loa ngoài, đặt nằm sàn cứng, ép tim 100-120 lần/phút ở giữa ngực sâu 5cm, tỷ lệ 30 ép tim : 2 thổi ngạt."
    },
    "confusion_delirium": {
        "title": "Mê Sảng Cấp Tính (Delirium) & Phân Biệt Sa Sút Trí Tuệ",
        "description": "Suy giảm nhận thức đột ngột trong vài giờ, mất tập trung, nói lảm nhảm, ảo thị; nhận diện cấp cứu tiềm ẩn do nhiễm trùng, thiếu oxy, hạ đường huyết hoặc mất nước."
    },
    "pain_management_safety": {
        "title": "Đánh Giá Thang Điểm Đau & Dùng Thuốc Giảm Đau An Toàn",
        "description": "Thang điểm VAS 0-10, giới hạn Paracetamol dưới 3000mg/ngày để bảo vệ gan, cảnh báo nguy cơ xuất huyết dạ dày và suy thận khi lạm dụng thuốc NSAID."
    },
    "parkinson_tremors_mobility": {
        "title": "Bệnh Parkinson, Run Tay & Đông Cứng Dáng Đi",
        "description": "Run khi nghỉ kiểu vê thuốc lào, cứng đờ cơ bắp; hiện tượng đông cứng dáng đi (dính chân xuống sàn) và kỹ thuật bước qua vạch kẻ ảo, cấm kéo giật tay cụ."
    },
    "urinary_tract_infection": {
        "title": "Nhiễm Trùng Tiết Niệu Kín Đáo (UTI) Gây Lú Lẫn",
        "description": "Nhiễm trùng tiểu ở người già không sốt cao rầm rộ mà biểu hiện đột ngột lú lẫn, tiểu đục mùi khai nồng, té ngã không rõ nguyên nhân; cần xét nghiệm nước tiểu ngay."
    },
    "nutrition_dysphagia": {
        "title": "Chăm Sóc Khó Nuốt & Phòng Ngừa Viêm Phổi Hít",
        "description": "Người già ho sặc khi ăn uống, giọng ướt đục; quy tắc ngồi thẳng lưng 90 độ khi ăn và duy trì 30 phút sau ăn, thức ăn xay nhuyễn sánh sệt, đút từng thìa nhỏ."
    },
    "nutrition_diet_restriction": {
        "title": "Chế Độ Dinh Dưỡng & Kiêng Khem (Tiểu Đường & Huyết Áp)",
        "description": "Tiết chế muối dưới 5g/ngày, kiêng đồ ngọt và hoa quả nhiều đường, chia nhỏ 4-5 bữa ăn, bổ sung đạm lành mạnh (cá, ức gà, đậu phụ) và rau củ mềm."
    },
    "bathing_hygiene_safety": {
        "title": "An Toàn Tắm Rửa & Phòng Ngừa Đột Quỵ Trong Nhà Tắm",
        "description": "Cấm tuyệt đối tắm đêm sau 19h; tắm nước ấm 37-38°C dội từ chân lên thân mình, không xối đột ngột lên đầu; lót thảm cao su chống trơn và lắp thanh vịn Inox."
    },
    "emotional_mental_sleep": {
        "title": "Tâm Lý Tuổi Già, Giấc Ngủ & Kỹ Năng Dỗ Cụ Cáu Gắt",
        "description": "Không tự ý dùng thuốc ngủ an thần gây té ngã; biện pháp ngâm chân nước ấm thảo dược; kỹ năng đồng cảm và chuyển hướng chú ý (Validation & Distraction) khi cụ từ chối uống thuốc hoặc đòi đi lang thang."
    },
    "exercise_physiotherapy": {
        "title": "Vận Động Dưỡng Sinh & Phục Hồi Chức Năng Tuổi Già",
        "description": "Đi bộ nhẹ nhàng 15-30 phút/ngày với giày chống trượt, bài tập vẩy tay Dịch Cân Kinh, xoay khớp cổ tay cổ chân buổi sáng chống cứng khớp."
    },
    "device_sos_support": {
        "title": "Hướng Dẫn Thiết Bị Giám Sát AuraCare & Nút Bấm SOS",
        "description": "Cách sử dụng nút bấm SOS đeo tay và đầu giường/nhà tắm; khả năng hoạt động 100% ngoại tuyến khi mất Internet; thông tin hotline BS. CKI Trần Minh Tuấn (0912.345.678) và cấp cứu 115."
    },
    "daily_routine_schedule": {
        "title": "Lịch Trình Sinh Hoạt & Nhịp Sống Thường Ngày",
        "description": "Thời gian biểu chuẩn cho cụ già: thức dậy 6h làm ấm khớp, ăn sáng uống thuốc 7h, phơi nắng 7h30, ngủ trưa 30-45 phút, ăn chiều trước 19h, chuẩn bị ngủ lúc 21h30."
    },
    "daily_sleep_insomnia": {
        "title": "Giấc Ngủ Thường Nhật & Khắc Phục Mất Ngủ Không Dùng Thuốc",
        "description": "Cụ trằn trọc khó ngủ, hay thức giấc lúc 2-3h sáng, đi tiểu đêm; ngâm chân nước ấm gừng muối, uống trà tâm sen loãng hoặc sữa ấm không đường, tránh uống nước nhiều sau 20h, giữ phòng tối và thoáng."
    },
    "daily_hygiene_denture_bath": {
        "title": "Vệ Sinh Cá Nhân, Tắm Rửa & Chăm Sóc Răng Giả Hàng Ngày",
        "description": "Cách dỗ cụ khi cụ lười tắm/sợ lạnh; tắm bằng ghế ngồi tắm có tựa lưng, lau người bằng khăn ấm vào ngày lạnh; tháo và ngâm rửa răng giả bằng dung dịch chuyên dụng sau mỗi bữa ăn."
    },
    "daily_drinking_tea_coffee": {
        "title": "Thói Quen Uống Nước, Trà & Cà Phê Hàng Ngày",
        "description": "Nhắc cụ uống đủ 1.5 - 2 lít nước chia nhỏ trong ngày; lưu ý cụ thích uống trà đặc (cần pha loãng, không uống lúc đói hoặc sau 16h vì gây tăng huyết áp và mất ngủ); kiêng cà phê đặc."
    },
    "daily_elderly_psychology_empathy": {
        "title": "Tâm Lý Thường Ngày, Sự Đãng Trí & Đồng Cảm Khi Cụ Cáu Kỉnh",
        "description": "Xử lý khi cụ hay quên kính lão, chìa khóa, đồ đạc; dỗ dành khi cụ cáu gắt, đòi về quê, cảm thấy cô đơn; kỹ thuật trò chuyện tôn trọng, kiên nhẫn, khuyến khích cụ kể chuyện xưa."
    },
    "daily_chit_chat_companion": {
        "title": "Chào Hỏi Lễ Phép & Trò Chuyện Bầu Bạn Thường Nhật",
        "description": "Các câu chào hỏi sớm tinh mơ, chúc ngủ ngon, hỏi thăm tâm trạng, dặn dò thời tiết (mặc áo ấm khi trời trở lạnh, tránh gió lùa), nhắc cụ uống nước ấm và vận động nhẹ."
    }
}


class LLMSyntheticGenerator:
    """Điều phối kết nối LLM API để tạo sinh dữ liệu huấn luyện tăng cường (RLHF / Few-Shot)."""

    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def set_config(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        """Cập nhật cấu hình kết nối API."""
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url.rstrip("/")
        if model:
            self.model = model

    def generate_qa_pairs(
        self,
        topic_key: str = "fall_emergency",
        num_samples: int = 5,
        generate_dpo: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Gọi API LLM để tạo sinh danh sách các cặp câu hỏi - câu trả lời chất lượng cao.
        Hỗ trợ tạo cả dữ liệu DPO: (prompt, chosen, rejected).
        """
        topic_info = GENERATION_TOPICS.get(topic_key, {
            "title": topic_key,
            "description": "Tình huống chăm sóc và theo dõi sức khỏe người cao tuổi."
        })

        system_prompt = (
            "Bạn là Chuyên gia Y tế Lão khoa & Kỹ sư Cấp cứu Y tế cao cấp.\n"
            "Nhiệm vụ của bạn là tạo sinh dữ liệu chuẩn (Ground Truth) phục vụ Huấn luyện Tăng cường (RLHF / DPO) "
            "cho Chatbot AI Giám sát Người cao tuổi tại gia đình (Hệ thống AuraCare AI).\n\n"
            "BỐI CẢNH DỰ ÁN:\n"
            "- Người cao tuổi: Cụ Nguyễn Văn An (82 tuổi, phòng 102 - Tầng 1). Tiền sử: Tăng huyết áp độ 2, Tiểu đường Type 2, Đau khớp, từng té ngã năm 2024. Dị ứng tuyệt đối: Penicillin. Bác sĩ phụ trách: BS. CKI Trần Minh Tuấn (0912.345.678).\n"
            "- Người bảo hộ chính đang hỏi: Nguyễn Hữu Nghĩa (Con trai trưởng, SĐT: 0908.123.456).\n"
            "- Cảm biến theo dõi: Camera (góc nghiêng, vận tốc rơi), Micro (tiếng va đập dB), Thân nhiệt (°C), Nhịp tim BLE (BPM).\n\n"
            "YÊU CẦU ĐẦU RA:\n"
            f"Hãy tạo đúng {num_samples} mẫu câu hỏi và phản hồi bằng Tiếng Việt chuẩn y tế về chủ đề: '{topic_info['title']}' ({topic_info['description']}).\n"
            "Mỗi mẫu phải là 1 đối tượng JSON chứa:\n"
            "- 'prompt': Câu hỏi thực tế của người nhà hoặc cụ già (ngắn gọn, tự nhiên, đa dạng cách hỏi đời thường).\n"
            "- 'chosen': Câu trả lời CHUẨN XÁC, chuẩn y khoa, bình tĩnh, hướng dẫn hành động cụ thể, nhắc nhở 115 khi cần (viết trên 1 dòng đơn hoặc dùng \\n chuẩn JSON, không ngắt dòng thô).\n"
            + ("- 'rejected': Câu trả lời SAI HOẶC NGUY HIỂM (ngắn gọn 1-2 câu trên 1 dòng đơn).\n" if generate_dpo else "")
            + "Định dạng kết quả trả về BẮT BUỘC là 1 mảng JSON thuần túy (không kèm giải thích ngoài JSON): [ {...}, {...} ]"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Nếu là OpenRouter, bổ sung thêm headers yêu cầu
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "https://auracare.local"
            headers["X-Title"] = "AuraCare AI"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Hãy tạo {num_samples} cặp dữ liệu Q&A chất lượng cao cho chủ đề '{topic_info['title']}'."}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }

        candidate_models = [self.model]
        for fb in ["mistralai/ministral-8b", "mistralai/mistral-large-2512", "mistralai/mistral-medium-3.5", "mistralai/mistral-small-2603", "deepseek/deepseek-chat-v3.1"]:
            if fb not in candidate_models:
                candidate_models.append(fb)

        url = f"{self.base_url}/chat/completions"

        for cand_model in candidate_models:
            payload["model"] = cand_model
            print(f"📡 [LLM GENERATOR] Đang gửi yêu cầu tới: {url} (Model: {cand_model})...", flush=True)

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code != 200:
                    print(f"⚠️ [LLM GENERATOR] Lỗi API ({resp.status_code}) với {cand_model}: {resp.text[:200]}", file=sys.stderr, flush=True)
                    continue

                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"].strip()

                parsed_samples = self._parse_json_samples(raw_content)
                if isinstance(parsed_samples, list) and len(parsed_samples) > 0:
                    print(f"✅ [LLM GENERATOR] Tạo sinh thành công {len(parsed_samples)} mẫu dữ liệu với mô hình {cand_model}!", flush=True)
                    return parsed_samples
            except Exception as e:
                print(f"⚠️ [LLM GENERATOR] Ngoại lệ khi gọi {cand_model}: {e}", file=sys.stderr, flush=True)

        print("❌ [LLM GENERATOR] Không thể tạo sinh sau khi đã thử tất cả mô hình dự phòng.", file=sys.stderr, flush=True)
        return []

    def _parse_json_samples(self, raw_content: str) -> List[Dict[str, Any]]:
        """Phân tích JSON bền vững, chịu lỗi khi LLM sinh ký tự xuống dòng hoặc dấu phẩy thừa."""
        if not raw_content:
            return []

        # 1. Tách khối code markdown nếu có
        clean_text = raw_content
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        # 2. Tìm mảng JSON [ ... ]
        start = clean_text.find('[')
        end = clean_text.rfind(']')
        json_target = clean_text[start:end+1] if (start != -1 and end != -1 and end > start) else clean_text.strip()

        # Thử parse trực tiếp với strict=False
        try:
            res = json.loads(json_target, strict=False)
            if isinstance(res, list) and len(res) > 0:
                return res
        except Exception:
            pass

        # Thử xử lý dấu phẩy thừa trước ] hoặc }
        try:
            no_trailing = re.sub(r',\s*([\]}])', r'\1', json_target)
            res = json.loads(no_trailing, strict=False)
            if isinstance(res, list) and len(res) > 0:
                return res
        except Exception:
            pass

        # 3. Trích xuất từng object {...} độc lập
        extracted_objs = []
        for block in re.finditer(r'\{[^{}]*?"prompt"[^{}]*?\}', clean_text, re.DOTALL):
            block_str = block.group(0)
            block_str = re.sub(r',\s*}', '}', block_str)
            try:
                obj = json.loads(block_str, strict=False)
                if "prompt" in obj and ("chosen" in obj or "response" in obj):
                    extracted_objs.append(obj)
            except Exception:
                continue

        if extracted_objs:
            return extracted_objs

        # 4. Fallback dùng regex trích xuất trường prompt, chosen
        prompts = re.findall(r'"prompt"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', clean_text)
        chosens = re.findall(r'"chosen"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', clean_text)
        if prompts and chosens:
            fallback = []
            for p, c in zip(prompts, chosens):
                fallback.append({
                    "prompt": p.replace('\\"', '"').replace('\\n', '\n'),
                    "chosen": c.replace('\\"', '"').replace('\\n', '\n'),
                    "rejected": "Chăm sóc sai quy chuẩn y tế."
                })
            return fallback

        return []

    def _stringify_field(self, val: Any) -> str:
        """Đảm bảo trường dữ liệu luôn là chuỗi ký tự UTF-8, không gây lỗi khi LLM sinh dict/list."""
        if val is None:
            return ""
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, (dict, list)):
            try:
                # Nếu là dict có các trường giải pháp, format thành chuỗi dễ đọc
                if isinstance(val, dict):
                    parts = []
                    for k, v in val.items():
                        if isinstance(v, list):
                            parts.append(f"• {k}:")
                            for item in v:
                                if isinstance(item, dict):
                                    item_str = ", ".join(f"{ik}: {iv}" for ik, iv in item.items())
                                    parts.append(f"  - {item_str}")
                                else:
                                    parts.append(f"  - {item}")
                        else:
                            parts.append(f"• {k}: {v}")
                    return "\n".join(parts)
                return json.dumps(val, ensure_ascii=False, indent=2)
            except Exception:
                return str(val)
        return str(val).strip()

    def integrate_into_knowledge_base(self, samples: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Tích hợp các mẫu dữ liệu vừa tạo vào hệ thống:
        1. Lưu vào file `synthetic_qa_generated.json`
        2. Bổ sung vào `few_shot_examples.json` để RAG học tức thì
        3. Bổ sung vào `rlhf_dataset.json` cho Fine-tuning DPO
        """
        if not samples:
            return {"few_shot_added": 0, "dpo_added": 0, "total_saved": 0}

        # 1. Lưu kho tổng
        existing_all: List[Dict[str, Any]] = []
        if os.path.exists(GENERATED_QA_FILE):
            try:
                with open(GENERATED_QA_FILE, "r", encoding="utf-8") as f:
                    existing_all = json.load(f)
            except Exception:
                existing_all = []

        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        for s in samples:
            s_record = {
                "prompt": self._stringify_field(s.get("prompt")),
                "chosen": self._stringify_field(s.get("chosen", s.get("response"))),
                "rejected": self._stringify_field(s.get("rejected")),
                "topic": s.get("topic", "general_care"),
                "category": s.get("category", "medical"),
                "generated_at": now_str,
                "model": self.model
            }
            existing_all.append(s_record)

        with open(GENERATED_QA_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_all, f, ensure_ascii=False, indent=2)

        # 2. Bổ sung vào Few-Shot Memory
        few_shot_data: List[Dict[str, Any]] = []
        if os.path.exists(FEW_SHOT_FILE):
            try:
                with open(FEW_SHOT_FILE, "r", encoding="utf-8") as f:
                    few_shot_data = json.load(f)
            except Exception:
                few_shot_data = []

        existing_prompts = {item.get("prompt", "").lower() for item in few_shot_data}
        added_few_shot = 0

        for s in samples:
            prompt = self._stringify_field(s.get("prompt"))
            chosen = self._stringify_field(s.get("chosen", s.get("response")))
            if prompt and chosen and prompt.lower() not in existing_prompts:
                few_shot_data.append({
                    "prompt": prompt,
                    "response": chosen,
                    "reward_score": 1.0,
                    "added_at": now_str,
                    "source": f"llm_synthetic_{self.model}"
                })
                existing_prompts.add(prompt.lower())
                added_few_shot += 1

        with open(FEW_SHOT_FILE, "w", encoding="utf-8") as f:
            json.dump(few_shot_data, f, ensure_ascii=False, indent=2)

        # 3. Đồng bộ trực tiếp vào CSDL SQLite rlhf_feedback & Xuất Dataset
        from src.training.rlhf_service import get_rlhf_service
        rlhf_svc = get_rlhf_service()

        added_dpo = 0
        for s in samples:
            prompt = self._stringify_field(s.get("prompt"))
            chosen = self._stringify_field(s.get("chosen", s.get("response")))
            rejected = self._stringify_field(s.get("rejected"))
            if prompt and chosen:
                if rejected:
                    rlhf_svc.record_feedback(
                        prompt=prompt,
                        response=rejected,
                        rating=-1,
                        corrected_response=chosen,
                        feedback_text=f"Auto-Synthesized DPO pair by {self.model}",
                        source="llm_synthetic"
                    )
                else:
                    rlhf_svc.record_feedback(
                        prompt=prompt,
                        response=chosen,
                        rating=1,
                        corrected_response=chosen,
                        feedback_text=f"Auto-Synthesized QA pair by {self.model}",
                        source="llm_synthetic"
                    )
                added_dpo += 1

        # Tự động xuất cập nhật lại rlhf_dataset.json và few_shot_examples.json
        rlhf_svc.export_datasets()

        print(f"🎯 [INTEGRATION] Đã nạp thành công: {added_few_shot} mẫu Few-Shot, {added_dpo} mẫu DPO/QA vào CSDL SQLite & Dataset.", flush=True)
        return {
            "few_shot_added": added_few_shot,
            "dpo_added": added_dpo,
            "total_saved": len(existing_all)
        }

    def generate_all_topics(self, samples_per_topic: int = 3) -> Dict[str, Any]:
        """Tạo sinh dữ liệu toàn diện cho tất cả 5 chủ đề y tế lão khoa."""
        total_samples: List[Dict[str, Any]] = []
        for topic_key in GENERATION_TOPICS.keys():
            print(f"\n--- [CHỦ ĐỀ: {topic_key}] ---", flush=True)
            samples = self.generate_qa_pairs(topic_key=topic_key, num_samples=samples_per_topic)
            total_samples.extend(samples)
            time.sleep(1)  # Giãn cách tránh rate limit

        res = self.integrate_into_knowledge_base(total_samples)
        res["total_generated"] = len(total_samples)
        return res


# Singleton helper
_generator_instance = None


def get_llm_generator() -> LLMSyntheticGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = LLMSyntheticGenerator()
    return _generator_instance


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tạo sinh dữ liệu huấn luyện tăng cường bằng LLM API")
    parser.add_argument("--key", type=str, default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL của API")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Tên model")
    parser.add_argument("--topic", type=str, default="all", help="Chủ đề (fall_emergency, fever_infection, v.v. hoặc 'all')")
    parser.add_argument("--count", type=int, default=3, help="Số mẫu trên mỗi chủ đề")
    args = parser.parse_args()

    gen = LLMSyntheticGenerator(api_key=args.key, base_url=args.base_url, model=args.model)
    print("=" * 70, flush=True)
    print("🤖 [LLM SYNTHETIC GENERATOR] KHỞI CHẠY TẠO SINH DỮ LIỆU HUẤN LUYỆN TĂNG CƯỜNG", flush=True)
    print(f"🌐 Base URL: {args.base_url}", flush=True)
    print(f"🧠 Model: {args.model}", flush=True)
    print(f"📋 Chủ đề: {args.topic}", flush=True)
    print("=" * 70, flush=True)

    if args.topic == "all":
        stats = gen.generate_all_topics(samples_per_topic=args.count)
    else:
        samples = gen.generate_qa_pairs(topic_key=args.topic, num_samples=args.count)
        stats = gen.integrate_into_knowledge_base(samples)

    print(f"\n🎉 HOÀN THÀNH! Kết quả: {stats}", flush=True)
