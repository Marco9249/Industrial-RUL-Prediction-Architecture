import time
import os
import subprocess
from datetime import datetime

# المسارات
SOURCE_DIR = r"c:\Users\Mohammed26\Desktop\ابحاثي\5"
SCRIPT_TO_RUN = r"C:\Users\Mohammed26\.gemini\antigravity\brain\63b9c41a-6313-48c4-b748-9266cb1c6b1d\scratch\scan_proposal_v2.py"

def get_last_modified():
    # مراقبة كافة ملفات Word أو النص في المجلد
    files = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith(('.docx', '.py', '.txt'))]
    if not files: return 0
    return max(os.path.getmtime(f) for f in files)

print(f"[{datetime.now().strftime('%H:%M:%S')}] جاري مراقبة الملفات... سيتم التحديث تلقائياً عند حفظ أي تغيير.")

last_check = get_last_modified()

try:
    while True:
        current_modified = get_last_modified()
        if current_modified > last_check:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] تم رصد تغيير! جاري تحديث البحث...")
            subprocess.run(["python", SCRIPT_TO_RUN])
            print("------------------------------------------")
            last_check = current_modified
        time.sleep(2) # فحص كل ثانيتين
except KeyboardInterrupt:
    print("\nتوقف المراقب.")
