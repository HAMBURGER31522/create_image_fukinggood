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
    # 也捞 "未执行" / "未完成"：前者是检查自身抛异常被 qa 的 except 吞成
    # 一行 note（"这条检查已经失效"），后者是让位撞下限后的残留越界。
    # 只捞 WARN 会让「ALL PASS」把这两类一起吞掉。环境性的 note（如
    # "已跳过无 Regular 字面的字体"）是正确行为，不在此列。
    hits = [ln.strip() for ln in (r.stdout or "").splitlines()
            if "WARN" in ln or "未执行" in ln or "未完成" in ln]
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
