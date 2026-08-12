"""出图后自动自检：能自动查的自动查，查不了的留给 agent 目测。

用法（脚本末尾）::

    from core import run_qa
    run_qa(fig, expect_width=("onehalf",))   # 不合格直接抛 AssertionError
"""
from __future__ import annotations

import io

import numpy as np
import matplotlib.pyplot as plt

from .colors import BANNED_CMAPS
from .style import COLUMN_WIDTHS

MIN_FONT_PT = 5.5   # 印刷可读下限（注释小字档）
_TOL_MM = 3.0


def _all_texts(fig):
    texts = list(fig.texts)
    for ax in fig.get_axes():
        texts += ax.texts
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += ax.get_xticklabels() + ax.get_yticklabels()
        leg = ax.get_legend()
        if leg is not None:
            texts += leg.get_texts()
    return [t for t in texts if t.get_text().strip()]


def run_qa(fig, expect_width=None, strict: bool = True) -> list[str]:
    """返回问题列表；strict=True 时有问题直接抛错。"""
    problems: list[str] = []

    # 1. 字号下限
    small = [t for t in _all_texts(fig) if t.get_fontsize() < MIN_FONT_PT]
    if small:
        problems.append(
            f"{len(small)} 处文字字号 < {MIN_FONT_PT}pt，印刷不可读："
            f"{[t.get_text()[:12] for t in small[:3]]}")

    # 2. 色图黑名单
    for ax in fig.get_axes():
        for coll in list(ax.collections) + list(ax.images):
            cm = getattr(coll, "get_cmap", lambda: None)()
            if cm is not None and cm.name in BANNED_CMAPS:
                problems.append(f"使用了被禁色图 {cm.name}（jet/rainbow 族）")

    # 3. 画布宽度
    if expect_width:
        w_mm = fig.get_figwidth() * 25.4
        targets = [COLUMN_WIDTHS.get(w, w) for w in expect_width]
        if not any(abs(w_mm - t) <= _TOL_MM for t in targets):
            problems.append(
                f"画布宽 {w_mm:.0f}mm 不在目标档 {targets}（±{_TOL_MM}mm）")

    # 4a. 中文与 mathtext 混排（同串含 CJK 和 $..$ 整串走 mathtext → 豆腐块）
    def _has_cjk(s):
        return any("\u4e00" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef"
                   for ch in s)
    mixed = [t.get_text()[:20] for t in _all_texts(fig)
             if _has_cjk(t.get_text()) and t.get_text().count("$") >= 2]
    if mixed:
        problems.append(
            f"{len(mixed)} 处中文与 $mathtext$ 混排（会豆腐块），"
            f"改用 Unicode 数学字符（₀ ⁻¹ φ ε √ 等）：{mixed[:2]}")

    # 4b. 豆腐块：渲染一次，同时捕获 warnings 与 matplotlib logging 两条通道
    import logging
    import warnings
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logging.getLogger("matplotlib").addHandler(handler)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        fig.canvas.draw()
    logging.getLogger("matplotlib").removeHandler(handler)
    n_glyph = sum(1 for w in wlist if "missing from font" in str(w.message))
    n_glyph += stream.getvalue().count("does not have a glyph")
    if n_glyph:
        problems.append(f"{n_glyph} 处字形缺失（豆腐块），检查字体回退与混排")

    # 5. 图例遮挡数据（查曲线顶点落入图例 bbox 的数量，避免包围盒误报）
    try:
        for ax in fig.get_axes():
            leg = ax.get_legend()
            if leg is None:
                continue
            lb = leg.get_window_extent()
            for line in ax.lines:
                xy = line.get_xydata()
                if len(xy) < 2:
                    continue
                pts = ax.transData.transform(xy)
                inside = np.sum((pts[:, 0] > lb.x0) & (pts[:, 0] < lb.x1) &
                                (pts[:, 1] > lb.y0) & (pts[:, 1] < lb.y1))
                if inside >= max(6, 0.1 * len(pts)):
                    problems.append(
                        f"图例覆盖曲线 {inside} 个数据点，移动图例或改线端直标")
                    break
    except Exception:
        pass

    # Windows 控制台默认 GBK，避免中文输出乱码
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if problems and strict:
        raise AssertionError("QA FAILED:\n- " + "\n- ".join(problems))
    if problems:
        for p in problems:
            print(f"[QA WARN] {p}")
    else:
        print("[QA] PASS (auto checks)")
    return problems
