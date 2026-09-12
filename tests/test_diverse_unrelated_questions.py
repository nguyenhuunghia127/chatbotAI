# -*- coding: utf-8 -*-
"""
KIỂM THỬ THỰC TẾ 6 CÂU HỎI ĐA LĨNH VỰC KHÔNG LIÊN QUAN NHAU
Dự án: Hệ thống Trợ lý AI Giám sát Người cao tuổi AuraCare AI
"""

import sys
import time
import requests
import json

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

test_cases = [
    ("CHỦ ĐỀ 1 - BHYT & Pháp lý", "Người từ đủ 80 tuổi trở lên có được Nhà nước cấp thẻ BHYT miễn phí và thanh toán 100% không?"),
    ("CHỦ ĐỀ 2 - Sơ cứu Bỏng nước sôi", "Người nhà bị bỏng nước sôi khi nấu bếp thì cần sơ cứu thế nào đúng cách trong 15 phút đầu?"),
    ("CHỦ ĐỀ 3 - Sóng Wi-Fi & Máy tạo nhịp", "Sóng Wi-Fi và sóng điện thoại di động có ảnh hưởng đến người già hoặc người đeo máy tạo nhịp tim không?"),
    ("CHỦ ĐỀ 4 - Cờ tướng & Rèn trí não", "Chơi cờ tướng hoặc cờ vua có giúp người già rèn luyện tư duy và phòng ngừa đãng trí không?"),
    ("CHỦ ĐỀ 5 - Thơ ca & Giải trí", "Hãy kể một câu chuyện vui ngắn hoặc đọc bài thơ về tuổi già thanh thản để cụ nghe thư giãn."),
    ("CHỦ ĐỀ 6 - Ngoài phạm vi y tế", "Hôm nay giá vàng thế nào và bạn có biết viết mã code lập trình Python không?")
]

print("=" * 80)
print("🧪 BẮT ĐẦU KIỂM ĐỊNH 6 CÂU HỎI ĐA LĨNH VỰC KHÔNG LIÊN QUAN NHAU")
print("=" * 80)

for i, (topic, q) in enumerate(test_cases, 1):
    print(f"\n" + "=" * 60)
    print(f"--- TEST {i}: [{topic}] ---")
    print(f"❓ HỎI: {q}")
    t0 = time.time()
    try:
        res = requests.post("http://127.0.0.1:8001/api/chat", json={"message": q}, timeout=30).json()
        dt = round(time.time() - t0, 2)
        ans = res.get("answer", "")
        status = res.get("status")
        engine = res.get("engine")
        sources = res.get("sources", [])
        print(f"⚡ KẾT QUẢ: {status} | ĐỘ TRỄ: {dt}s | ĐỘ DÀI: {len(ans)} ký tự")
        print(f"🤖 ENGINE: {engine}")
        print(f"📚 SOURCES: {sources[:2]}")
        print("\n📖 ĐẦU TRÍCH ĐOẠN:\n" + ans[:300].strip() + "...")
        print("\n🎯 CÂU KẾT BÀI:\n" + ans[-220:].strip())
    except Exception as e:
        print(f"❌ LỖI TRUY VẤN: {e}")

print("\n" + "=" * 80)
print("🎉 HOÀN TẤT KIỂM ĐỊNH TẤT CẢ CÁC CHỦ ĐỀ!")
print("=" * 80)
