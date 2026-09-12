# -*- coding: utf-8 -*-
"""
MODULE HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI Ý ĐỊNH Y TẾ & CẤP CỨU (INTENT CLASSIFIER TRAINER)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Công nghệ:
1. Feature Extraction: TfidfVectorizer (Word n-grams 1-2, sublinear_tf=True)
2. Classification Model: CalibratedClassifierCV với LinearSVC hoặc LogisticRegression (class_weight='balanced')
3. Data Sources:
   - Dữ liệu tạo sinh thực tế qua LLM API (synthetic_qa_generated.json)
   - Bộ dữ liệu phản hồi người dùng RLHF / DPO (rlhf_feedback & rlhf_dataset.json)
   - Bộ dữ liệu hạt giống lâm sàng chuẩn (Clinical Seed Dataset) bao phủ 18 nhãn
   - Kỹ thuật Data Augmentation câu hỏi Tiếng Việt
4. Lưu trữ: models/intent_classifier.joblib
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

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

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "intent_classifier.joblib")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "data", "training_data")

os.makedirs(MODELS_DIR, exist_ok=True)

# Bộ dữ liệu mẫu hạt giống chuẩn lâm sàng (Clinical Seed Corpus)
CLINICAL_SEED_CORPUS = {
    "EMERGENCY_STROKE": [
        "cụ bị đột quỵ rồi phải làm sao",
        "dấu hiệu đột quỵ ở người già",
        "miệng cụ bị méo lệch sang một bên",
        "tay cụ bị yếu liệt không nhấc lên được",
        "cụ nói đớ nói ngọng không rõ chữ",
        "nhận diện quy tắc fast đột quỵ",
        "giờ vàng cấp cứu đột quỵ là bao lâu",
        "cụ có bị tai biến không",
        "nghi ngờ cụ bị tai biến mạch máu não",
        "mặt cụ bị xệ một bên và rơi bát cơm",
        "cụ ngã rồi nói ngọng líu lưỡi",
        "có nên cạo gió khi cụ bị méo miệng không",
        "tại sao không được chích lể đầu ngón tay khi đột quỵ"
    ],
    "EMERGENCY_FALL": [
        "sáng nay cụ có bị ngã không",
        "cụ vừa bị ngã trong nhà tắm",
        "lúc 16h cụ có ngã không",
        "cần làm gì ngay khi người già bị té ngã",
        "tại sao không được vội đỡ người già dậy khi bị ngã",
        "làm thế nào để người già tự đứng dậy an toàn sau khi ngã",
        "cụ bị trượt chân ngã nằm sàn",
        "góc nghiêng cơ thể 85 độ va đập mạnh",
        "tiếng va đập lớn nghi ngờ cụ té",
        "cụ ngã đập mông xuống đất kêu đau hông",
        "cụ bị ngã gãy xương hông",
        "kiểm tra camera xem cụ có ngã không",
        "cụ nằm bất động dưới sàn nhà"
    ],
    "EMERGENCY_CARDIAC": [
        "nhịp tim của cụ hôm nay thế nào",
        "nhịp tim trên 120 bpm ở người cao tuổi phải làm gì",
        "dấu hiệu cảnh báo nhồi máu cơ tim ở người già",
        "cụ bị đau thắt ngực dữ dội",
        "cụ kêu nặng ngực lan lên vai và cánh tay trái",
        "cụ bị khó thở vã mồ hôi ôm ngực",
        "nhịp tim cụ tăng vọt 145 bpm lúc nghỉ",
        "tư thế ngồi fowler cho bệnh nhân tim mạch",
        "mạch của cụ đập nhanh bất thường",
        "loạn nhịp tim ở người già có nguy hiểm không",
        "cụ kêu hồi hộp đánh trống ngực"
    ],
    "EMERGENCY_HYPERTENSION": [
        "huyết áp của cụ tăng vọt lên 185 thì làm sao",
        "cụ bị tăng huyết áp kịch phát 190/110",
        "cụ kêu đau đầu dữ dội vùng sau gáy và hoa mắt",
        "huyết áp tâm thu trên 180 có nguy cơ gì",
        "có nên cho cụ ngậm adalat nhỏ dưới lưỡi khi huyết áp cao không",
        "xử trí cơn tăng huyết áp cấp cứu tại nhà",
        "cụ bị chảy máu cam và huyết áp tăng cao",
        "huyết áp cụ đo được 180/120 mmhg phải xử trí thế nào",
        "tư thế nghỉ ngơi cho người huyết áp cao vọt"
    ],
    "EMERGENCY_HYPOGLYCEMIA": [
        "cụ bị hạ đường huyết phải làm sao",
        "cụ bị tụt đường huyết run tay chân vã mồ hôi lạnh",
        "triệu chứng đói cồn cào hoa mắt ở người tiểu đường",
        "quy tắc 15-15 hạ đường huyết là gì",
        "cho cụ uống nước cam hay kẹo khi tụt đường",
        "uống bao nhiêu thìa đường khi bị hạ đường huyết",
        "cụ lơ mơ ngất xỉu do đói thuốc tiểu đường",
        "đường huyết dưới 70 mg dl cần xử trí cấp cứu ra sao",
        "sau khi uống nước đường bao lâu thì đo lại huyết đường"
    ],
    "EMERGENCY_CHOKING": [
        "cụ bị sặc nghẹn thức ăn phải làm sao",
        "cụ bị hóc dị vật tím tái mặt mày",
        "thủ thuật heimlich cho người già thực hiện thế nào",
        "cụ đang ăn cháo thì ho sặc sụa ôm cổ họng",
        "cách sơ cứu hóc dị vật khi cụ ngồi xe lăn",
        "cụ bị ngạt thở do mắc nghẹn viên thuốc",
        "ấn bụng hình chữ j đẩy dị vật đường thở",
        "cụ nghẹn không thở được và bất tỉnh",
        "sơ cứu sặc đường thở ở người cao tuổi"
    ],
    "ALLERGY_PENICILLIN": [
        "cụ bị dị ứng thuốc gì",
        "cụ có được uống amoxicillin không",
        "cụ có uống được augmentin không",
        "tại sao cấm tuyệt đối penicillin cho cụ an",
        "dị ứng kháng sinh nhóm beta lactam nguy hiểm thế nào",
        "nguy cơ sốc phản vệ khi dùng penicillin ở người già",
        "bác sĩ có kê đơn ampicillin cho cụ được không",
        "cụ bị sốt có được mua kháng sinh penicillin uống không",
        "thông báo dị ứng thuốc cho bác sĩ phụ trách"
    ],
    "FIRST_AID_CPR": [
        "cách ép tim cpr cho người già",
        "hồi sinh tim phổi cơ bản khi cụ ngừng thở",
        "tỷ lệ ép tim và thổi ngạt là bao nhiêu",
        "tần số ép tim 100 đến 120 lần một phút",
        "vị trí đặt tay ép tim ở đâu",
        "cụ bất tỉnh không bắt được mạch và không thở",
        "ép tim hands-only cpr cho người già",
        "gọi 115 và ép tim liên tục"
    ],
    "DISTRESS_VOICE": [
        "cứu tôi với",
        "cứu cụ với",
        "ai cứu tôi với đau quá",
        "tiếng rên rỉ đau đớn trong phòng",
        "tiếng bát đĩa rơi vỡ trong bếp",
        "cụ kêu đau chân không dậy được",
        "tiếng kêu cứu phát ra từ phòng 102",
        "alo có ai không cứu"
    ],
    "FEVER_INFECTION": [
        "hôm nay cụ có bị sốt không",
        "sốt ở người già có nguy hiểm không",
        "cách hạ sốt đúng cho người cao tuổi",
        "thân nhiệt cụ 38.9 độ c lúc trưa",
        "chườm khăn ấm hạ sốt ở trán nách bẹn",
        "bù nước oresol cho cụ bị sốt",
        "nhiễm trùng tiết niệu gây sốt ở người cao tuổi",
        "nhiệt độ hiện tại của cụ là bao nhiêu"
    ],
    "SUNDOWNING_DEMENTIA": [
        "hội chứng hoàng hôn là gì",
        "tại sao cứ chập tối là cụ đòi về nhà",
        "cụ bị lú lẫn quậy phá khi trời tối",
        "cách xử lý hội chứng sundowning ở người già",
        "bật đèn ấm chống ảo giác hoàng hôn",
        "cụ đòi mở cửa đi ra ngoài lúc nửa đêm",
        "sa sút trí tuệ và kích động ban đêm",
        "kỹ năng giao tiếp thấu cảm cho người sa sút trí tuệ"
    ],
    "DEHYDRATION_CARE": [
        "trời nắng nóng cụ lười uống nước có sao không",
        "dấu hiệu mất nước ở người cao tuổi",
        "tại sao người già không cảm thấy khát nước",
        "da khô nhăn và nước tiểu sẫm màu có phải thiếu nước",
        "cần cho cụ uống bao nhiêu lít nước mỗi ngày",
        "bù nước oresol từng ngụm nhỏ",
        "phòng ngừa sốc nhiệt say nắng cho cụ già mùa hè",
        "tụt huyết áp tư thế do thiếu nước"
    ],
    "BEDSORES_CARE": [
        "làm sao để cụ nằm một chỗ không bị loét lưng",
        "cách phòng ngừa loét tì đè cho người nằm liệt",
        "mỗi mấy tiếng phải xoay trở tư thế nằm cho cụ",
        "vị trí nào dễ bị loét nhất khi nằm lâu",
        "có nên đấm bóp vùng da bị đỏ ửng không",
        "đệm hơi chống loét có tác dụng gì",
        "chế độ ăn giàu đạm giúp lành vết loét da",
        "vệ sinh da chống loét xương cùng cụt"
    ],
    "MEDICATION_CARE": [
        "nếu cụ quên uống một liều thuốc huyết áp thì làm sao",
        "nếu cụ lỡ quên một liều thuốc huyết áp thì có uống gấp đôi không",
        "uống gấp đôi liều thuốc huyết áp khi lỡ quên",
        "có được uống gấp đôi liều thuốc khi lỡ quên không",
        "cách sắp xếp hộp chia thuốc 7 ngày 4 buổi",
        "uống thuốc huyết áp và tiểu đường cùng lúc được không",
        "nhắc nhở cụ uống thuốc đúng giờ",
        "xử trí khi lỡ uống nhầm thuốc",
        "quản lý đa thuốc polypharmacy cho người cao tuổi"
    ],
    "IDENTITY_USER_PROFILE": [
        "tôi là ai",
        "tôi tên gì",
        "tôi tên là gì",
        "bạn biết tôi là ai không",
        "thông tin của tôi",
        "hồ sơ của tôi",
        "tôi có quyền gì trong hệ thống",
        "vai trò của tôi là gì",
        "ai đang hỏi câu này"
    ],
    "IDENTITY_PATIENT_PROFILE": [
        "cụ là ai",
        "ông là ai",
        "bà là ai",
        "người bệnh là ai",
        "thông tin của cụ",
        "hồ sơ bệnh án của cụ",
        "tiền sử bệnh của cụ nguyễn văn an",
        "bác sĩ phụ trách của cụ là ai",
        "cụ nằm ở phòng nào",
        "năm nay cụ bao nhiêu tuổi"
    ],
    "GENERAL_HEALTH_CHECK": [
        "tình hình sức khỏe của cụ hôm nay thế nào",
        "sức khỏe của cụ ra sao rồi",
        "cụ khỏe không",
        "báo cáo tổng quan sức khỏe trong ngày",
        "hôm nay có sự cố gì không",
        "tổng kết tình trạng người bệnh",
        "chỉ số sinh tồn của cụ ổn định không"
    ],
    "GREETING_SYSTEM_TEST": [
        "alo 1 2 3 4",
        "kiểm tra micro",
        "thử micro",
        "nghe rõ không",
        "xin chào",
        "chào bạn",
        "hello",
        "trợ lý này có những tính năng gì",
        "bạn làm được những gì"
    ],
    "CONFUSION_DELIRIUM": [
        "cụ bị lú lẫn đột ngột mê sảng có phải cấp cứu không",
        "phân biệt mê sảng cấp tính và sa sút trí tuệ",
        "mê sảng delirium ở người cao tuổi có hồi phục được không",
        "tại sao cụ tự nhiên nhìn thấy ảo giác và kích động lú lẫn",
        "cụ bị lẫn lộn ngày đêm bứt rứt không yên",
        "nguyên nhân gây mê sảng cấp tính ở người già",
        "cụ đột nhiên không nhận ra người thân trong vài giờ",
        "cần đưa cụ đi viện ngay khi bị mê sảng cấp không"
    ],
    "PAIN_MANAGEMENT": [
        "thang điểm đau vas từ 0 đến 10 đánh giá thế nào",
        "cụ kêu đau khớp dữ dội mức 7 trên 10",
        "uống bao nhiêu paracetamol một ngày là an toàn cho người già",
        "liều paracetamol tối đa 3000mg trên ngày",
        "tại sao không nên tự ý dùng thuốc giảm đau nsaid kháng viêm cho người già",
        "thuốc giảm đau nào gây xuất huyết dạ dày và suy thận ở người cao tuổi",
        "cụ bị đau lưng có được uống diclofenac hay meloxicam không",
        "biện pháp giảm đau không dùng thuốc cho người già như chườm ấm xoa bóp"
    ],
    "PARKINSON_MOBILITY": [
        "cụ bị bệnh parkinson run tay khi nghỉ",
        "hiện tượng đông cứng dáng đi freezing of gait là gì",
        "chân cụ bị dính chặt xuống sàn nhà không bước đi được",
        "kỹ thuật bước qua vạch kẻ ảo cho người bị parkinson",
        "làm sao để cụ bị parkinson không bị ngã khi xoay người",
        "nguyên tắc đi thẳng không quay ngoắt người cho người parkinson",
        "triệu chứng run tay run chân và cứng đờ cơ bắp ở người già",
        "hướng dẫn tập phục hồi chức năng vận động cho bệnh nhân parkinson"
    ],
    "UTI_INFECTION": [
        "nhiễm trùng đường tiết niệu kín đáo ở người già nguy hiểm thế nào",
        "tại sao cụ bị nhiễm trùng tiểu uti lại không sốt mà chỉ lú lẫn",
        "nước tiểu của cụ có mùi khai nồng vẩn đục và tiểu rắt",
        "cụ tiểu buốt tiểu són và thay đổi tri giác đột ngột",
        "xét nghiệm tổng phân tích nước tiểu cho người cao tuổi",
        "cụ bị tiểu không tự chủ và sốt nhẹ nghi viêm đường tiết niệu",
        "biến chứng sốc nhiễm khuẩn niệu urosepsis ở người cao tuổi",
        "cho cụ uống đủ nước để phòng ngừa nhiễm trùng đường tiểu"
    ],
    "DYSPHAGIA_NUTRITION": [
        "chứng khó nuốt dysphagia ở người cao tuổi",
        "cụ bị nghẹn sặc khi uống nước lọc phải làm sao",
        "tư thế ngồi thẳng 90 độ khi ăn để chống sặc thức ăn",
        "làm sao phòng ngừa viêm phổi hít do sặc thức ăn ở người già",
        "chất làm đặc nước cho người già khó nuốt",
        "chế độ ăn mềm xay nhuyễn cho cụ bị nghẹn nuốt khó",
        "không được cho cụ nằm ngay sau khi ăn ít nhất 30 phút",
        "dấu hiệu nghẹn hóc thức ăn và viêm phổi hít ở người nằm liệt"
    ],
    "NUTRITION_DIET": [
        "chế độ ăn uống dinh dưỡng cho người cao tuổi tăng huyết áp tiểu đường",
        "cụ an cần kiêng ăn những gì",
        "mỗi ngày cụ được ăn tối đa bao nhiêu muối",
        "người già bị tiểu đường có nên ăn hoa quả ngọt không",
        "thực đơn ăn uống chuẩn cho người già bị huyết áp cao",
        "tại sao phải hạn chế muối dưới 5 gam một ngày",
        "cụ bị tiểu đường và huyết áp nên chia bao nhiêu bữa ăn một ngày",
        "thực phẩm tốt cho tim mạch và đường huyết của người già",
        "chế độ dinh dưỡng khoa học cho cụ an",
        "bệnh nhân của tôi ăn nhiều muối được không",
        "người bệnh có được ăn mặn không",
        "ăn bao nhiêu muối mỗi ngày là đủ",
        "cụ an có được ăn dưa cà muối không",
        "tiểu đường và cao huyết áp có ăn bánh ngọt được không"
    ],
    "BATHING_HYGIENE": [
        "người cao tuổi có nên tắm đêm sau 19h không",
        "tại sao cấm tuyệt đối tắm muộn sau 19 giờ ở người già",
        "nhiệt độ nước tắm an toàn cho người cao tuổi là bao nhiêu",
        "quy trình tắm đúng cách cho cụ già dội nước từ chân lên",
        "phòng ngừa trượt ngã đột quỵ trong phòng tắm",
        "thời gian tắm tối đa cho người già là mấy phút",
        "cần chuẩn bị thảm chống trượt và tay vịn nhà tắm thế nào",
        "hướng dẫn tắm rửa vệ sinh an toàn cho cụ an",
        "nguy cơ co mạch tai biến khi tắm nước lạnh hoặc tắm đêm"
    ],
    "EMOTIONAL_MENTAL": [
        "cụ bị khó ngủ mất ngủ trằn trọc ban đêm phải làm sao",
        "có nên tự mua thuốc ngủ hoặc thuốc an thần cho cụ uống không",
        "tác hại nguy hiểm của thuốc an thần gây té ngã và lú lẫn ở người già",
        "cụ an cáu gắt khó tính dỗi không chịu uống thuốc",
        "kỹ thuật lắng nghe validation và đánh lạc hướng distraction",
        "làm sao khi cụ bướng bỉnh từ chối ăn uống và uống thuốc",
        "tâm lý người cao tuổi cô đơn trầm cảm cần chăm sóc thế nào",
        "biện pháp giúp cụ ngủ ngon không dùng thuốc tây",
        "cách dỗ cụ uống thuốc khi cụ khó chịu cáu giận"
    ],
    "EXERCISE_PHYSIOTHERAPY": [
        "bài tập thể dục dưỡng sinh an toàn cho người cao tuổi",
        "cụ bị thoái hóa khớp có nên đi bộ hàng ngày không",
        "mỗi ngày người già nên đi bộ bao nhiêu phút",
        "hướng dẫn bài tập vẩy tay dịch cân kinh cho người già",
        "cụ bị cứng khớp buổi sáng cần xoa bóp và khởi động thế nào",
        "bài tập vận động nhẹ nhàng phòng ngừa té ngã",
        "người cao tuổi tập thể dục cần đi giày chống trượt ra sao",
        "thời điểm tập thể dục dưỡng sinh tốt nhất cho người già",
        "các bài tập phục hồi chức năng vận động cho cụ"
    ],
    "DEVICE_SOS_SUPPORT": [
        "hướng dẫn sử dụng nút bấm khẩn cấp sos",
        "nút bấm sos đeo tay và gắn tường hoạt động thế nào",
        "khi mất mạng wifi hoặc cúp điện nút sos có gửi cảnh báo được không",
        "hệ thống nút sos có hoạt động offline ngoại tuyến không",
        "số điện thoại liên hệ khẩn cấp của bác sĩ tuấn là gì",
        "gọi cấp cứu 115 và liên hệ bác sĩ cki trần minh tuấn",
        "bác sĩ phụ trách cụ an có số điện thoại nào",
        "cách bấm chuông báo động sos khi cụ gặp sự cố",
        "danh bạ y tế khẩn cấp của cụ an"
    ],
    "LEGAL_INSURANCE_POLICY": [
        "người cao tuổi trên 80 tuổi có được cấp thẻ bhyt miễn phí không",
        "thẻ bhyt người già được quỹ bảo hiểm thanh toán bao nhiêu phần trăm",
        "quyền lợi bảo hiểm y tế cho người cao tuổi từ đủ 80 tuổi",
        "thủ tục lĩnh thuốc bảo hiểm y tế thay cho bố mẹ già",
        "giấy ủy quyền nhận thuốc bhyt định kỳ cho người cao tuổi",
        "thủ tục chuyển tuyến khám chữa bệnh bhyt cho người cao tuổi",
        "xin giấy chuyển viện bảo hiểm y tế cho cụ",
        "chính sách bồi hoàn và chi trả 100% bhyt cho người già",
        "con cái có được đi lấy thuốc huyết áp tiểu đường thay cho bố mẹ không"
    ],
    "DOMESTIC_FIRSTAID": [
        "bị bỏng nước sôi khi nấu ăn thì sơ cứu thế nào",
        "sơ cứu bỏng nước nóng cho người già",
        "bỏng dầu mỡ nóng khi nấu bếp xử lý 15 phút đầu ra sao",
        "bị đứt tay chảy máu khi gọt hoa quả",
        "cách băng ép cầm máu vết thương rách da chảy máu",
        "bị ong vò vẽ đốt xử trí thế nào",
        "kiến ba khoang cắn hoặc dính độc kiến ba khoang",
        "sơ cứu tai nạn sinh hoạt thông thường trong gia đình",
        "tại sao không được bôi kem đánh răng hay nước mắm lên vết bỏng"
    ],
    "TECH_ENVIRONMENT_HEALTH": [
        "sóng wifi và sóng điện thoại có ảnh hưởng sức khỏe người già không",
        "người đeo máy tạo nhịp tim có được dùng điện thoại di động không",
        "khoảng cách an toàn giữa điện thoại và máy tạo nhịp tim pacemaker",
        "mùa hè bật điều hòa máy lạnh cho người già bao nhiêu độ",
        "nhiệt độ điều hòa thích hợp để người già không bị cảm lạnh đột quỵ",
        "tác hại của ánh sáng xanh màn hình tivi điện thoại trước khi ngủ",
        "có nên bật đèn ngủ ánh sáng vàng cho người già không",
        "phòng ngủ của cụ nên để nhiệt độ bao nhiêu độ c"
    ],
    "COGNITIVE_BRAIN_EXERCISE": [
        "chơi cờ tướng cờ vua có giúp người già rèn luyện trí nhớ không",
        "các bài tập trí não phòng ngừa đãng trí alzheimer",
        "trò chơi đố chữ ô số sudoku cho người cao tuổi",
        "làm sao để rèn luyện não bộ duy trì sự minh mẫn cho người già",
        "đọc sách báo kích thích thần kinh và chống teo não",
        "phương pháp rèn luyện trí nhớ brain gym cho người cao tuổi",
        "chơi trò chơi gì để ông bà không bị lẫn và suy giảm nhận thức"
    ],
    "SMALLTALK_ENTERTAINMENT": [
        "kể cho tôi một câu chuyện vui ngắn cho cụ già",
        "bài thơ về tuổi già thanh thản an yên",
        "thú vui trồng cây cảnh nuôi chim thư giãn cho người già",
        "làm sao để tạo không khí sum vầy vui vẻ trong gia đình",
        "nghe nhạc gì giúp người già thư giãn hạ huyết áp",
        "kể một mẩu chuyện vui giải trí nhẹ nhàng",
        "những câu danh ngôn hay về sự hiếu thảo và tuổi già",
        "chăm sóc cây cảnh hoa cỏ ngoài ban công giúp ích gì cho tâm lý"
    ],
    "GENERAL_OUT_OF_SCOPE": [
        "giá vàng hôm nay bao nhiêu một lượng",
        "bạn có biết viết code lập trình python không",
        "xe máy bị chết máy giữa đường thì sửa thế nào",
        "thời tiết hôm nay ở hà nội ra sao",
        "bạn có biết sửa chữa đồ điện tử gia dụng không",
        "tổng thống mỹ hiện nay là ai",
        "cổ phiếu chứng khoán hôm nay tăng hay giảm",
        "hướng dẫn nấu món bún bò huế",
        "bạn làm thơ tình yêu được không"
    ],
    "HERBAL_DRUG_INTERACTION": [
        "người đang uống thuốc huyết áp có được uống thêm nhân sâm không",
        "uống tam thất và đông trùng hạ thảo có tương tác với thuốc tiểu đường không",
        "uống sâm có làm tăng vọt huyết áp không",
        "uống nước lá vối hàng ngày có bị tụt huyết áp hay hại thận không",
        "trà atiso và giảo cổ lam uống thay nước lọc có tốt cho người già không",
        "tương tác giữa thuốc tây y và thảo dược đông y",
        "người cao tuổi có nên tự ý dùng thuốc nam thuốc bắc bồi bổ không"
    ],
    "BEDRIDDEN_PRESSURE_ULCER": [
        "người già nằm một chỗ bị đỏ rát vùng mông và xương cùng",
        "cách phòng ngừa và chăm sóc loét tì đè cho người già nằm liệt",
        "bao nhiêu tiếng cần lật trở người một lần cho bệnh nhân nằm lâu",
        "hướng dẫn sử dụng đệm hơi chống loét cho người già",
        "xử lý vết loét tì đè hoại tử rỉ dịch ở người cao tuổi",
        "tại sao không được rắc bột kháng sinh lên vết loét tì đè",
        "vệ sinh và chăm sóc da chống lở loét cho người già nằm liệt giường"
    ],
    "VISION_HEARING_CARE": [
        "cụ già mắt nhìn mờ như có màn sương che có phải đục thủy tinh thể không",
        "khi nào người cao tuổi cần mổ phaco thay thủy tinh thể nhân tạo",
        "mắt cụ bị chói lóa nhìn một thành hai hình",
        "cách chọn và bảo quản máy trợ thính cho người cao tuổi",
        "vệ sinh núm tai nghe máy trợ thính chống ráy tai và ẩm mốc",
        "kỹ năng giao tiếp nói chuyện với người già bị lãng tai nghễnh ngãng",
        "tăng độ sáng đèn trong nhà để phòng ngừa té ngã do mắt kém"
    ],
    "WEATHER_JOINT_SEASONAL": [
        "tại sao mỗi khi trời trở lạnh sắp mưa thì khớp gối lại đau buốt",
        "cơ chế đau nhức xương khớp khi thời tiết thay đổi độ ẩm và áp suất",
        "cách chườm ấm và xoa bóp khớp gối mùa đông cho người già",
        "phòng ngừa đột quỵ do co mạch khi thức dậy ban đêm mùa đông",
        "tại sao không được bước chân trần xuống nền nhà lạnh ban đêm",
        "người già nên đi tất len và mặc quần ấm khi ngủ mùa đông ra sao",
        "rửa mặt đánh răng bằng nước ấm chống co giật mạch máu mùa lạnh"
    ],
    "DENTAL_DENTURE_NUTRITION": [
        "chế biến thức ăn cho người cao tuổi răng yếu rụng nhiều răng",
        "làm sao để người già nhai khó vẫn đủ chất đạm protein và chất xơ",
        "thực đơn cháo súp xay nhuyễn dinh dưỡng cho cụ răng yếu",
        "tại sao ban đêm khi đi ngủ bắt buộc phải tháo hàm răng giả",
        "nguy cơ sặc nuốt hàm răng giả vào đường thở gây tử vong khi ngủ",
        "cách vệ sinh cọ rửa và ngâm hàm răng giả tháo lắp đúng chuẩn",
        "dung dịch ngâm rửa hàm giả chuyên dụng cho người già"
    ],
    "SOCIAL_INTERACTIVE_RIDDLES": [
        "làm sao khi người già nhất quyết đòi tự đi chợ bằng xe đạp xe máy cũ",
        "kỹ thuật chuyển hướng distraction khi cụ bướng bỉnh đòi đi xe",
        "kể 3 câu đố vui dân gian cho ông bà và con cháu cùng giải đố",
        "câu đố dân gian kích thích trí não khơi gợi ký ức xưa",
        "trò chơi đố vui gắn kết tình cảm gia đình buổi tối",
        "cách ứng xử nhẹ nhàng khi người già suy giảm khả năng lái xe"
    ]
}

# Tiền tố/hậu tố để mở rộng dữ liệu tự động (Data Augmentation)
PREFIXES = [
    "", "xin hỏi ", "cho tôi hỏi ", "bác sĩ ơi ", "trợ lý ơi ", "alo ",
    "bạn cho tôi biết ", "hãy cho biết ", "hướng dẫn tôi ", "làm sao để "
]
SUFFIXES = [
    "", " ạ", " hả bạn", " thế nào", " được không", " giúp tôi với", " ngay bây giờ"
]


def load_augmented_training_data() -> Tuple[List[str], List[str]]:
    """Tổng hợp và tăng cường tập dữ liệu huấn luyện từ seed corpus + synthetic QA."""
    X: List[str] = []
    y: List[str] = []

    # 1. Nạp từ Seed Corpus với Augmentation
    for intent, sentences in CLINICAL_SEED_CORPUS.items():
        for s in sentences:
            # Câu gốc
            X.append(s)
            y.append(intent)

            # Sinh thêm 2-3 biến thể tự nhiên
            for _ in range(2):
                pref = random.choice(PREFIXES)
                suff = random.choice(SUFFIXES)
                aug = f"{pref}{s}{suff}".strip()
                if aug != s:
                    X.append(aug)
                    y.append(intent)

    # 2. Nạp thêm từ tệp synthetic_qa_generated.json nếu có
    syn_file = os.path.join(TRAINING_DIR, "synthetic_qa_generated.json")
    if os.path.exists(syn_file):
        try:
            with open(syn_file, "r", encoding="utf-8") as f:
                syn_data = json.load(f)
                for item in syn_data:
                    prompt = item.get("prompt", "").strip()
                    topic = item.get("topic", "").lower()
                    if not prompt:
                        continue

                    # Ánh xạ topic sang intent (Ưu tiên các chủ đề đặc hiệu trước từ khóa chung)
                    mapped_intent = None
                    if "stroke" in topic or "đột quỵ" in prompt.lower() or "fast" in prompt.lower():
                        mapped_intent = "EMERGENCY_STROKE"
                    elif "medicat" in topic or "quên liều" in prompt.lower() or "uống thuốc" in prompt.lower() or "hộp chia thuốc" in prompt.lower() or "gấp đôi" in prompt.lower() or "uống bù" in prompt.lower():
                        mapped_intent = "MEDICATION_CARE"
                    elif "penicillin" in topic or "dị ứng" in prompt.lower():
                        mapped_intent = "ALLERGY_PENICILLIN"
                    elif "pain" in topic or "vas" in prompt.lower() or "paracetamol" in prompt.lower():
                        mapped_intent = "PAIN_MANAGEMENT"
                    elif "delirium" in topic or "mê sảng" in prompt.lower() or "confusion" in topic:
                        mapped_intent = "CONFUSION_DELIRIUM"
                    elif "parkinson" in topic or "freezing" in prompt.lower() or "đông cứng" in prompt.lower():
                        mapped_intent = "PARKINSON_MOBILITY"
                    elif "urinary" in topic or "uti" in topic or "tiết niệu" in prompt.lower() or "nước tiểu" in prompt.lower():
                        mapped_intent = "UTI_INFECTION"
                    elif "dysphagia" in topic or "khó nuốt" in prompt.lower() or "viêm phổi hít" in prompt.lower():
                        mapped_intent = "DYSPHAGIA_NUTRITION"
                    elif "cpr" in topic or "ép tim" in prompt.lower():
                        mapped_intent = "FIRST_AID_CPR"
                    elif "choking" in topic or "nghẹn" in prompt.lower() or "hóc" in prompt.lower():
                        mapped_intent = "EMERGENCY_CHOKING"
                    elif "hypoglyc" in topic or "đường huyết" in prompt.lower():
                        mapped_intent = "EMERGENCY_HYPOGLYCEMIA"
                    elif "fall" in topic or "ngã" in prompt.lower():
                        mapped_intent = "EMERGENCY_FALL"
                    elif "cardiac" in topic or "tim" in prompt.lower():
                        mapped_intent = "EMERGENCY_CARDIAC"
                    elif "hypertens" in topic or "huyết áp" in prompt.lower():
                        mapped_intent = "EMERGENCY_HYPERTENSION"
                    elif "sundown" in topic or "hoàng hôn" in prompt.lower():
                        mapped_intent = "SUNDOWNING_DEMENTIA"
                    elif "dehydrat" in topic or "mất nước" in prompt.lower():
                        mapped_intent = "DEHYDRATION_CARE"
                    elif "bedsores" in topic or "loét" in prompt.lower():
                        mapped_intent = "BEDSORES_CARE"
                    elif "diet" in topic or "nutrition" in topic or "ăn kiêng" in prompt.lower() or "dinh dưỡng" in prompt.lower() or "giảm muối" in prompt.lower() or "kiêng ngọt" in prompt.lower():
                        mapped_intent = "NUTRITION_DIET"
                    elif "bath" in topic or "hygiene" in topic or "tắm" in prompt.lower() or "vệ sinh" in prompt.lower():
                        mapped_intent = "BATHING_HYGIENE"
                    elif "mental" in topic or "sleep" in topic or "emotional" in topic or "mất ngủ" in prompt.lower() or "thuốc ngủ" in prompt.lower() or "cáu gắt" in prompt.lower() or "từ chối uống thuốc" in prompt.lower():
                        mapped_intent = "EMOTIONAL_MENTAL"
                    elif "exercise" in topic or "physio" in topic or "vận động" in prompt.lower() or "dưỡng sinh" in prompt.lower() or "dịch cân kinh" in prompt.lower() or "vẩy tay" in prompt.lower() or "đi bộ" in prompt.lower():
                        mapped_intent = "EXERCISE_PHYSIOTHERAPY"
                    elif "sos" in topic or "device" in topic or "nút bấm" in prompt.lower() or "bác sĩ tuấn" in prompt.lower() or "ngoại tuyến" in prompt.lower() or "mất mạng" in prompt.lower():
                        mapped_intent = "DEVICE_SOS_SUPPORT"
                    elif "thuốc" in prompt.lower():
                        mapped_intent = "MEDICATION_CARE"

                    if mapped_intent:
                        X.append(prompt)
                        y.append(mapped_intent)
        except Exception as e:
            print(f"[TRAIN_INTENT] Cảnh báo đọc file synthetic: {e}")

    return X, y


def train_intent_model(save_path: str = MODEL_SAVE_PATH) -> Dict[str, Any]:
    """Huấn luyện và lưu mô hình Scikit-Learn TF-IDF Intent Classifier."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import joblib

    print("🚀 [TRAIN_INTENT] Chuẩn bị dữ liệu huấn luyện NLU Intent Classifier...")
    X, y = load_augmented_training_data()
    print(f"📊 [TRAIN_INTENT] Tổng số mẫu dữ liệu: {len(X)} mẫu trên {len(set(y))} nhãn lâm sàng.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=4000,
            token_pattern=r"(?u)\b\w+\b"
        )),
        ("clf", LogisticRegression(
            C=3.0,
            max_iter=600,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    # Đánh giá chéo 5-fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    mean_acc = round(scores.mean(), 4)
    print(f"📈 [TRAIN_INTENT] Độ chính xác Cross-Validation (5-Fold): {mean_acc * 100:.2f}% (Độ lệch chuẩn: {scores.std():.4f})")

    # Huấn luyện trên toàn bộ tập dữ liệu
    pipeline.fit(X, y)

    # Lưu mô hình
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"💾 [TRAIN_INTENT] Đã lưu mô hình thành công vào: {save_path}")

    return {
        "status": "success",
        "total_samples": len(X),
        "total_classes": len(set(y)),
        "cv_accuracy": mean_acc,
        "model_path": save_path
    }


if __name__ == "__main__":
    res = train_intent_model()
    print("Kết quả:", res)
