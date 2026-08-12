"""联络表：把一批 PNG 拼成带文件名的网格图，agent 一次 Read 批量目测。

72h 赛时用法::

    python tools/contact_sheet.py gallery            # 全部图
    python tools/contact_sheet.py out/chap3 --cols 3 # 指定目录

输出 <目录>/_contact_sheet_N.png；对可疑图再单独 Read 原图复检。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

PER_SHEET = 12


def build(folder: str, cols: int = 3, per_sheet: int = PER_SHEET) -> list[str]:
    d = Path(folder)
    pngs = sorted(p for p in d.glob("*.png")
                  if not p.name.startswith("_contact_sheet"))
    if not pngs:
        print(f"目录 {d} 下没有 PNG")
        return []
    out_files = []
    for s in range(0, len(pngs), per_sheet):
        batch = pngs[s:s + per_sheet]
        rows = (len(batch) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 3.6))
        axes = list(getattr(axes, "flat", [axes]))
        for ax in axes:
            ax.set_axis_off()
        for ax, p in zip(axes, batch):
            ax.imshow(mpimg.imread(p))
            ax.set_title(p.name, fontsize=8, family="monospace")
        fig.tight_layout()
        out = d / f"_contact_sheet_{s // per_sheet + 1}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        out_files.append(str(out))
        print(f"[OK] {out}（{len(batch)} 张）")
    return out_files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="gallery")
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    build(args.folder, cols=args.cols)
