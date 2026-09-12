# -*- coding: utf-8 -*-
"""
SCRIPT CHẠY TIẾN TRÌNH HUẤN LUYỆN NGẦM TỰ ĐỘNG (STANDALONE AUTONOMOUS TRAINER)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi (Local 100%)

Cách chạy:
    1. Chạy ngầm định kỳ liên tục (mặc định mỗi 10 phút / 600s tự huấn luyện 1 lần):
       python run_auto_trainer.py

    2. Chạy với chu kỳ tùy chỉnh (ví dụ 60 giây / 1 phút):
       python run_auto_trainer.py --interval 60

    3. Chạy 1 chu kỳ duy nhất rồi thoát:
       python run_auto_trainer.py --once
"""

import os
import sys
import time
import argparse

# Đảm bảo UTF-8 cho console Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.training.auto_trainer import get_auto_trainer


def main():
    parser = argparse.ArgumentParser(description="Chạy tiến trình tự động huấn luyện ngầm cho AI Chatbot & ML Cảm biến")
    parser.add_argument("--interval", type=int, default=600, help="Khoảng cách giữa các chu kỳ huấn luyện (giây). Mặc định: 600s (10 phút)")
    parser.add_argument("--once", action="store_true", help="Chạy đúng 1 chu kỳ rồi kết thúc")
    args = parser.parse_args()

    trainer = get_auto_trainer()

    print("=" * 80, flush=True)
    print("🤖 [AUTONOMOUS TRAINER] KHỞI ĐỘNG TIẾN TRÌNH TỰ ĐỘNG HUẤN LUYỆN NGẦM", flush=True)
    print("🎯 Dự án: Chatbot AI & Hệ thống Giám sát Người cao tuổi Local", flush=True)
    print(f"⏱️ Chế độ: {'Chạy 1 lần (--once)' if args.once else f'Lặp định kỳ mỗi {args.interval} giây'}", flush=True)
    print("=" * 80, flush=True)

    if args.once:
        res = trainer.run_full_training_cycle()
        print(f"\n✅ Hoàn tất chu kỳ huấn luyện! Kết quả: {res}", flush=True)
        return

    # Chạy vòng lặp định kỳ
    trainer.start_background_daemon(interval_seconds=args.interval)
    print(f"🚀 Tiến trình daemon đã bắt đầu. Nhấn Ctrl+C để dừng lại bất kỳ lúc nào.\n", flush=True)

    try:
        while True:
            time.sleep(2)
            st = trainer.get_status()
            # print status summary periodic
    except KeyboardInterrupt:
        print("\n🛑 Nhận tín hiệu dừng từ người dùng. Đang tắt tiến trình ngầm...", flush=True)
        trainer.stop_background_daemon()
        print("✅ Đã dừng an toàn.", flush=True)


if __name__ == "__main__":
    main()
