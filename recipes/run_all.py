"""批量运行全部 recipe demo，出图到 gallery/。失败的列出但不中断。"""
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
SKIP = {"run_all.py", "_common.py"}

failed = []
for f in sorted(HERE.glob("*.py")):
    if f.name in SKIP:
        continue
    r = subprocess.run([sys.executable, str(f)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    status = "OK " if r.returncode == 0 else "FAIL"
    print(f"[{status}] {f.name}")
    if r.returncode != 0:
        failed.append(f.name)
        tail = "\n".join((r.stderr or "").strip().splitlines()[-12:])
        print(tail)

print(f"\n{len(failed)} failed" if failed else "\nALL PASS")
sys.exit(1 if failed else 0)
