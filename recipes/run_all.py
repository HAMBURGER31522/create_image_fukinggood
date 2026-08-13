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
warns = []
for f in sorted(HERE.glob("*.py")):
    if f.name in SKIP:
        continue
    r = subprocess.run([sys.executable, str(f)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    status = "OK " if r.returncode == 0 else "FAIL"
    # 警告级项（注释框盖数据、草稿档等）不算失败，但必须可见，
    # 否则「ALL PASS」会把警告级回归一起吞掉
    hits = [ln.strip() for ln in (r.stdout or "").splitlines()
            if "WARN" in ln]
    print(f"[{status}] {f.name}" + (f"  ({len(hits)} warn)" if hits else ""))
    warns += [(f.name, h) for h in hits]
    if r.returncode != 0:
        failed.append(f.name)
        tail = "\n".join((r.stderr or "").strip().splitlines()[-12:])
        print(tail)

if warns:
    print(f"\n--- {len(warns)} warnings ---")
    for name, h in warns:
        print(f"  {name}: {h}")

print(f"\n{len(failed)} failed" if failed else "\nALL PASS")
sys.exit(1 if failed else 0)
