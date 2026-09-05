"""期刊档 × preset × 全部模板的实跑矩阵。

为什么需要它：库只有 `FF_PRESET` 一个环境变量入口，`recipes/run_all.py` 与
`audit.py check` 都无法在 `journal=` 档下跑一遍。于是「26 个模板在 ieee 档下
有 17 个跑不通」这件事，项目自身完全看不见——和第 16 轮「demo/ 十五轮没被
任何检查跑过」「nature 档 25 个模板全硬拒」是同一类：**检查范围盖不到的
功能，等于没有门禁**。

用法::

    python tools/journal_matrix.py                    # 全矩阵
    python tools/journal_matrix.py --journal ieee     # 只跑一档
    python tools/journal_matrix.py --save base.json   # 存成基线
    python tools/journal_matrix.py --baseline base.json   # 只报比基线变差的格

退出码：`--baseline` 模式下有格子变差则 1，否则 0；无基线时恒 0（矩阵是
观测工具，不是门禁——把它当门禁需要先把模板改成期刊档干净的，那是另一件事）。

**必须每个组合一个子进程**：模板走 `from core import apply_style`，
`core/__init__` 在 import 时就把原函数绑到了包命名空间上，进程内只 patch
`core.style.apply_style` 不生效（主窗口第一版 sweep 正是这样得到假的 26/26）。
`FF_GALLERY` 同理——`recipes/_common.py` 在 import 时把它解析成模块常量，
必须在进程启动**前**设好，否则出图会落进仓库的 `gallery/`。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"run_all.py", "_common.py"}
JOURNALS = (None, "nature", "cn", "ieee", "pnas")
PRESETS = ("cn", "nature")

# 子进程引导：注入 journal 后按 __main__ 跑模板。写成 -c 字符串而不是临时
# 文件，免得在仓库里留垃圾；journal 为空串表示不注入。
_BOOT = r"""
import os, sys, runpy
repo, recipe, journal = sys.argv[1], sys.argv[2], (sys.argv[3] or None)
sys.path.insert(0, os.path.join(repo, "demo"))
sys.path.insert(0, os.path.join(repo, "recipes"))
sys.path.insert(0, repo)
import matplotlib; matplotlib.use("Agg")
if journal:
    import core, core.style as S
    _o = S.apply_style
    def _patched(preset=None, base_size=None, draft=False, *, journal=None,
                 _j=journal, _o=_o):
        # 模板自己不传 journal；显式传了的（没有这种模板）以它为准
        return _o(preset, base_size, draft, journal=journal or _j)
    S.apply_style = _patched
    core.apply_style = _patched      # 模板走 from core import apply_style
runpy.run_path(recipe, run_name="__main__")
"""


def targets(root: Path) -> list[Path]:
    rec = sorted((root / "recipes").glob("*.py"))
    demo = sorted((root / "demo").glob("*.py"))
    return [f for f in rec + demo if f.name not in SKIP]


def _verdict(proc: subprocess.CompletedProcess) -> str:
    """把一次运行压成一格。OK / 建图期报错 / QA 拒绝。"""
    if proc.returncode == 0:
        return "OK"
    err = (proc.stderr or "") + "\n" + (proc.stdout or "")
    lines = [l.strip() for l in err.splitlines() if l.strip()]
    # QA 硬拒：AssertionError 后跟若干 "- xxx" 明细
    qa = [l for l in lines if l.startswith("- ")]
    if any("QA FAILED" in l for l in lines):
        head = qa[0][2:] if qa else ""
        return f"QA×{len(qa)}: {head[:60]}"
    for l in reversed(lines):
        if l.startswith(("ValueError", "RuntimeError", "TypeError",
                         "KeyError", "AssertionError")):
            return l[:72]
    return (lines[-1][:72] if lines else f"exit {proc.returncode}")


def run_cell(root: Path, recipe: Path, preset: str, journal: str | None,
             gallery: Path) -> str:
    env = dict(os.environ)
    env["FF_PRESET"] = preset
    env["FF_GALLERY"] = str(gallery)      # 必须在子进程启动前设好
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    gallery.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c", _BOOT, str(root), str(recipe), journal or ""],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, cwd=str(root))
    return _verdict(proc)


def run_matrix(root: Path, journals, presets, out_root: Path) -> dict:
    names = [f.name for f in targets(root)]
    result: dict[str, dict[str, str]] = {}
    for journal in journals:
        for preset in presets:
            combo = f"{preset}+{journal or 'none'}"
            cells = {}
            for f in targets(root):
                cells[f.name] = run_cell(
                    root, f, preset, journal, out_root / combo / f.stem)
            ok = sum(1 for v in cells.values() if v == "OK")
            print(f"[{combo:16}] {ok}/{len(names)} OK", flush=True)
            result[combo] = cells
    return result


def render(result: dict) -> str:
    combos = list(result)
    names = sorted({n for c in result.values() for n in c})
    w = max(len(n) for n in names) + 1
    head = "模板".ljust(w) + "".join(c.center(16) for c in combos)
    out = [head, "-" * len(head)]
    for n in names:
        cells = []
        for c in combos:
            v = result[c].get(n, "?")
            cells.append(("OK" if v == "OK" else "FAIL").center(16))
        out.append(n.ljust(w) + "".join(cells))
    out.append("-" * len(head))
    out.append("合计".ljust(w) + "".join(
        f"{sum(1 for v in result[c].values() if v == 'OK')}/{len(names)}"
        .center(16) for c in combos))
    out.append("")
    out.append("失败明细：")
    for c in combos:
        bad = {n: v for n, v in result[c].items() if v != "OK"}
        if not bad:
            continue
        out.append(f"  [{c}] {len(bad)} 条")
        for n, v in sorted(bad.items()):
            out.append(f"    {n:28} {v}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", action="append",
                    help="只跑这些档（可重复）；none 表示不注入")
    ap.add_argument("--preset", action="append", choices=list(PRESETS))
    ap.add_argument("--save", metavar="FILE", help="把结果存成基线 JSON")
    ap.add_argument("--baseline", metavar="FILE",
                    help="与基线比，只报变差的格（有变差退出码 1）")
    ap.add_argument("--out", metavar="DIR", help="出图目录（默认临时目录）")
    a = ap.parse_args()

    journals = ([None if j == "none" else j for j in a.journal]
                if a.journal else list(JOURNALS))
    presets = a.preset or list(PRESETS)

    tmp = None
    if a.out:
        out_root = Path(a.out)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="ff_journal_matrix_")
        out_root = Path(tmp.name)
    try:
        result = run_matrix(ROOT, journals, presets, out_root)
    finally:
        if tmp is not None:
            tmp.cleanup()

    print()
    print(render(result))

    if a.save:
        Path(a.save).write_text(
            json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")
        print(f"\n基线已存：{a.save}")

    if a.baseline:
        base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
        worse, better = [], []
        for combo, cells in sorted(result.items()):
            for name, now in sorted(cells.items()):
                was = base.get(combo, {}).get(name)
                if was is None:
                    continue
                if was == "OK" and now != "OK":
                    worse.append((combo, name, now))
                elif was != "OK" and now == "OK":
                    better.append((combo, name, was))
        print(f"\n对比基线 {a.baseline}：变好 {len(better)} 格 / "
              f"变差 {len(worse)} 格")
        for combo, name, was in better:
            print(f"  ✓ [{combo}] {name}  （原：{was[:50]}）")
        for combo, name, now in worse:
            print(f"  ✗ [{combo}] {name}  {now[:60]}")
        return 1 if worse else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
