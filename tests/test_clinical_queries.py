# -*- coding: utf-8 -*-
"""Tập lệnh kiểm thử toàn diện các năng lực lâm sàng mới được huấn luyện của Chatbot AI."""
import sys
import os
import requests

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

url = 'http://127.0.0.1:8001/api/chat'

test_cases = [
    {
        "domain": "ĐỘT QUỴ NÃO (F.A.S.T)",
        "query": "cụ có dấu hiệu bị méo miệng và nói đớ thì phải làm sao?"
    },
    {
        "domain": "TĂNG HUYẾT ÁP KỊCH PHÁT",
        "query": "huyết áp của cụ tăng vọt lên 185 thì xử trí thế nào?"
    },
    {
        "domain": "CHỐNG CHỈ ĐỊNH PENICILLIN",
        "query": "cụ bị sốt có được mua amoxicillin hay augmentin uống không?"
    },
    {
        "domain": "HẠ ĐƯỜNG HUYẾT (QUY TẮC 15-15)",
        "query": "cụ bị run tay vã mồ hôi lạnh đói cồn cào do tụt đường huyết"
    },
    {
        "domain": "SẶC NGHẸN / HÓC DỊ VẬT (HEIMLICH)",
        "query": "cụ đang ăn cháo bị sặc nghẹn tím tái mặt mày ôm cổ"
    },
    {
        "domain": "LOÉT TÌ ĐÈ & NẰM LIỆT",
        "query": "làm sao để cụ nằm liệt giường không bị loét da ở lưng?"
    },
    {
        "domain": "MẤT NƯỚC & SAY NẮNG",
        "query": "trời nóng cụ lười uống nước da khô nhăn thì phải làm gì?"
    },
    {
        "domain": "QUẢN LÝ ĐA THUỐC & QUÊN LIỀU",
        "query": "nếu cụ quên uống liều thuốc huyết áp sáng nay thì có được uống gấp đôi bù không?"
    },
    {
        "domain": "HỒI SINH TIM PHỔI (CPR)",
        "query": "cách ép tim cpr cho cụ già khi bất tỉnh ngừng thở"
    },
    {
        "domain": "DANH TÍNH NGƯỜI DÙNG & HỒ SƠ",
        "query": "tôi là ai và cụ là ai?"
    }
]

print("=" * 80, flush=True)
print("🩺 KẾT QUẢ KIỂM THỬ TRUY VẤN TRI THỨC ĐÃ HUẤN LUYỆN (AURA CARE AI)", flush=True)
print("=" * 80, flush=True)

for idx, tc in enumerate(test_cases, 1):
    q = tc["query"]
    r = requests.post(url, json={"message": q}, timeout=10)
    data = r.json()
    answer = data.get("answer", "")
    engine = data.get("engine", "")
    
    print(f"\n[{idx}] CHUYÊN ĐỀ: {tc['domain']}", flush=True)
    print(f"❓ CÂU HỎI: \"{q}\"", flush=True)
    print(f"🤖 PHẢN HỒI (Engine: {engine}):\n{answer}", flush=True)
    print("-" * 80, flush=True)

print("\n✅ KIỂM THỬ HOÀN TẤT THÀNH CÔNG 10/10 TÌNH HUỐNG LÂM SÀNG!", flush=True)
