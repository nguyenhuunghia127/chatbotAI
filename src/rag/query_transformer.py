# -*- coding: utf-8 -*-
"""
MODULE QUERY TRANSFORMATION RAG (MULTI-QUERY & CHUẨN HÓA LÂM SÀNG)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Kiến trúc: Compound MedRAG & Query Transformation

Chức năng:
1. Ánh xạ khẩu ngữ đời thường / câu hỏi micro STT thành thuật ngữ y khoa chuẩn (Clinical Normalizer).
   - Ví dụ: "cụ bị đớ lưỡi" -> "đột quỵ não tai biến mạch máu não phác đồ FAST"
   - Ví dụ: "ngã cái rầm đau hông" -> "té ngã chấn thương gãy xương hông cấp cứu 115"
2. Sinh đa truy vấn song song (Multi-Query Expansion):
   - Mở rộng 1 câu hỏi thành 2-3 truy vấn con chuyên biệt giúp Vector DB và TF-IDF quét phủ rộng,
     không bị sót tài liệu do khác biệt cách diễn đạt.
3. Hoàn toàn 100% Offline, không phụ thuộc mô hình ngoài.
"""

import os
import sys
import re
from typing import List, Dict, Any, Tuple

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ClinicalQueryTransformer:
    """Bộ chuyển đổi và mở rộng truy vấn lâm sàng chuyên biệt cho Lão khoa."""

    # Bảng ánh xạ khẩu ngữ / triệu chứng dân gian -> Thuật ngữ & Phác đồ y tế chuẩn
    COLLOQUIAL_CLINICAL_MAP = [
        # Đột quỵ & Thần kinh
        (r'\b(đớ lưỡi|nói ngọng|méo mồm|méo miệng|lệch mặt|rớt tay|liệt tay|không giơ tay được)\b',
         "đột quỵ não tai biến mạch máu não phác đồ fast giờ vàng cấp cứu 115"),
        
        # Té ngã & Chấn thương
        (r'\b(ngã cái rầm|té ngã|ngã đập mông|đập sàn|nằm bẹp|kêu đau hông|gãy xương hông)\b',
         "té ngã chấn thương sàn nhà va đập bất tỉnh gọi 115 kiểm tra tri giác gãy xương"),
        
        # Tim mạch & Huyết áp
        (r'\b(tim đập nhanh|đánh trống ngực|mạch nhanh|140 bpm|145 bpm|huyết áp vọt|180|185|đau tức ngực)\b',
         "cơn tăng huyết áp kịch phát loạn nhịp tim nhồi máu cơ tim nghỉ tựa 45 độ gọi 115"),
        
        # Hạ đường huyết
        (r'\b(run tay|vã mồ hôi lạnh|đói cồn cào|bủn rủn|hoa mắt|tụt đường)\b',
         "hạ đường huyết cấp tính đái tháo đường quy tắc 15-15 nước cam mật ong kẹo ngọt"),
        
        # Hóc sặc nghẹn
        (r'\b(hóc cháo|sặc nghẹn|tím tái|nghẹn thức ăn|ngạt thở|không thở được)\b',
         "sặc nghẹn hóc dị vật đường thở thủ thuật heimlich ấn ngực sơ cứu khẩn cấp"),
        
        # Dị ứng & Kháng sinh
        (r'\b(amoxicillin|augmentin|penicillin|ampicillin|kháng sinh)\b',
         "dị ứng tuyệt đối kháng sinh nhóm penicillin sốc phản vệ cấm tuyệt đối bác sĩ trần minh tuấn"),
        
        # Đau nhức & Thuốc giảm đau
        (r'\b(đau nhức xương|uống thuốc giảm đau|paracetamol|panadol|ibuprofen|nsaid|thang điểm đau)\b',
         "giảm đau an toàn paracetamol tối đa 3000mg cấm nsaid xuất huyết dạ dày suy thận"),
        
        # Nhiễm trùng tiểu & Lú lẫn
        (r'\b(nước tiểu khai nồng|tiểu buốt|tiểu rắt|vẩn đục|lú lẫn đột ngột|mê sảng)\b',
         "nhiễm trùng đường tiết niệu kín đáo uti mê sảng delirium xét nghiệm nước tiểu urosepsis"),
        
        # Hội chứng hoàng hôn & Sa sút trí tuệ
        (r'\b(chập tối đòi về|hoàng hôn|sundowning|quậy phá ban đêm|nhìn thấy người lạ)\b',
         "hội chứng hoàng hôn sundowning sa sút trí tuệ bật đèn sáng ấm âm nhạc êm dịu"),
        
        # Dinh dưỡng & Ăn kiêng
        (r'\b(ăn mặn|nhiều muối|bao nhiêu muối|kiêng muối|bớt muối|tiểu đường ăn gì|kiêng ăn gì)\b',
         "chế độ dinh dưỡng ăn kiêng giảm muối dưới 5g kiểm soát đường bột chia nhỏ 4-5 bữa"),
        
        # Tắm rửa
        (r'\b(tắm đêm|sau 19h|nước tắm|dội nước|nhà tắm)\b',
         "an toàn tắm rửa người cao tuổi cấm tắm đêm sau 19h nước ấm 37 độ thảm chống trượt"),
        
        # Giấc ngủ & Tâm lý
        (r'\b(mất ngủ|khó ngủ|thuốc ngủ|thuốc an thần|cáu gắt|dỗi|không chịu uống thuốc)\b',
         "chăm sóc giấc ngủ cấm tự ý dùng thuốc an thần ngâm chân nước ấm kỹ thuật thấu cảm validation")
    ]

    def __init__(self):
        pass

    def normalize_and_expand(self, query: str) -> Dict[str, Any]:
        """
        Phân tích câu hỏi, chuẩn hóa khẩu ngữ và sinh Multi-Query mở rộng.
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()

        expanded_terms = []
        matched_categories = []

        for pattern, clinical_terms in self.COLLOQUIAL_CLINICAL_MAP:
            if re.search(pattern, q_lower):
                expanded_terms.append(clinical_terms)
                matched_categories.append(clinical_terms.split()[0])

        # Sinh 2 - 3 truy vấn con (Multi-Queries) cho tầng Retrieval
        multi_queries = [q_clean]

        if expanded_terms:
            # Truy vấn 2: Câu hỏi gốc kết hợp với thuật ngữ y khoa mở rộng
            combined_query = f"{q_clean} {' '.join(expanded_terms[:2])}"
            multi_queries.append(combined_query)

            # Truy vấn 3: Chỉ tập trung vào phác đồ lâm sàng trọng tâm
            protocol_query = f"phác đồ sơ cứu hướng dẫn điều trị {' '.join(expanded_terms[0].split()[:6])}"
            multi_queries.append(protocol_query)
        else:
            # Nếu không khớp mẫu khẩu ngữ đặc biệt, sinh biến thể tìm kiếm theo từ khóa chính
            words = [w for w in q_lower.replace("?", "").replace(",", "").split() if len(w) > 2]
            if len(words) >= 3:
                multi_queries.append(f"hướng dẫn y tế người cao tuổi {' '.join(words[:5])}")

        return {
            "original_query": q_clean,
            "matched_categories": matched_categories,
            "expanded_terms": expanded_terms,
            "multi_queries": list(dict.fromkeys(multi_queries)) # Giữ thứ tự và loại trùng lặp
        }


# Singleton
clinical_query_transformer_instance = None


def get_query_transformer() -> ClinicalQueryTransformer:
    global clinical_query_transformer_instance
    if clinical_query_transformer_instance is None:
        clinical_query_transformer_instance = ClinicalQueryTransformer()
    return clinical_query_transformer_instance


if __name__ == "__main__":
    transformer = get_query_transformer()
    print("=" * 70)
    print("KIỂM THỬ MODULE QUERY TRANSFORMATION RAG (MULTI-QUERY)")
    print("=" * 70)

    test_queries = [
        "cụ tự nhiên bị đớ lưỡi một bên mặt rồi ngã nằm bẹp",
        "cụ đòi uống augmentin hoặc amoxicillin để hạ sốt được không",
        "trời lạnh cụ có được tắm đêm sau 19h không"
    ]

    for q in test_queries:
        print(f"\n❓ [CÂU GỐC]: {q}")
        res = transformer.normalize_and_expand(q)
        print(f"🔍 [DANH MỤC KHỚP]: {res['matched_categories']}")
        print("⚡ [MULTI-QUERIES ĐÃ SINH]:")
        for i, mq in enumerate(res["multi_queries"], 1):
            print(f"   {i}. {mq}")
