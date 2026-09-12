# -*- coding: utf-8 -*-
"""
KIỂM THỬ THỰC TẾ 6 CHỦ ĐỀ MỞ RỘNG ĐỢT 2
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
    ("CHỦ ĐỀ 7 - Dược lý Thảo dược & Thuốc Tây", "Người đang uống thuốc hạ huyết áp và tiểu đường có được uống thêm nhân sâm hay tam thất không?"),
    ("CHỦ ĐỀ 8 - Loét Tì Đè & Nằm Liệt", "Người già nằm một chỗ bị đỏ rát vùng mông và xương cùng thì phòng ngừa và xử lý lở loét tì đè thế nào?"),
    ("CHỦ ĐỀ 9 - Đục Thủy Tinh Thể & Thị Lực", "Cụ già mắt nhìn mờ như có màn sương che, hay bị chói mắt thì có phải bị đục thủy tinh thể không và khi nào cần mổ?"),
    ("CHỦ ĐỀ 10 - Khớp Khi Trời Lạnh", "Tại sao mỗi khi trời trở lạnh hoặc sắp mưa thì các khớp gối của người già lại đau nhức buốt hơn?"),
    ("CHỦ ĐỀ 11 - Răng Yếu & Dinh Dưỡng Hàm Giả", "Cụ bị rụng nhiều răng, đeo hàm giả nhai khó thì chế biến thức ăn thế nào để vẫn đủ chất đạm và chất xơ?"),
    ("CHỦ ĐỀ 12 - Kỹ Năng Ứng Xử Gia Đình", "Làm thế nào khi người già nhất quyết đòi tự đi chợ bằng xe đạp cũ dù chân tay đã yếu?")
]

print("=" * 80)
print("🧪 BẮT ĐẦU KIỂM ĐỊNH 6 CHỦ ĐỀ MỞ RỘNG ĐỢT 2")
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
        print("\n📖 ĐẦU TRÍCH ĐOẠN:\n" + ans[:250].strip() + "...")
        print("\n🎯 CÂU KẾT BÀI:\n" + ans[-200:].strip())
    except Exception as e:
        print(f"❌ LỖI TRUY VẤN: {e}")

print("\n" + "=" * 80)
print("🎉 HOÀN TẤT KIỂM ĐỊNH TẤT CẢ CÁC CHỦ ĐỀ ĐỢT 2!")
print("=" * 80)
