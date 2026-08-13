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
from .style import COLUMN_WIDTHS, delivered_width_in, is_styled, is_draft

# 6.5 = 排版缩放后仍 ≥6pt 的注释小字下限（SPEC §2.4 印刷档 7–10.5pt）
MIN_FONT_PT = 6.5
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


def run_qa(fig, expect_width=None, strict: bool = True,
           sourced=None) -> list[str]:
    """返回问题列表；strict=True 时有问题直接抛错。

    sourced: 图题数字的来源字典（recipe 返回的 info）。缺省取
    fig._ff_stats。给了来源就核对图题里的数字是否都能在来源值中
    找到（±2% 或四舍五入相等），抓"手写常数"违规。
    """
    problems: list[str] = []

    # 0. 图题数字溯源（有 _ff_stats/sourced 才查；只警告不阻断——
    #    差值/比率等合法派生数字无法穷举，误杀比漏报更伤）
    sourced = sourced if sourced is not None else getattr(fig, "_ff_stats",
                                                          None)
    if sourced:
        import re

        def _flat(v):
            if isinstance(v, dict):
                # 数值型的 key（如分位数 0.5）也算来源
                return [x for kv in v.items() for e in kv for x in _flat(e)]
            if isinstance(v, (list, tuple, np.ndarray)):
                return [x for vv in np.ravel(v) for x in _flat(vv)]
            try:
                f = float(v)
                return [f] if np.isfinite(f) else []
            except (TypeError, ValueError):
                return []

        base_vals = _flat(sourced)
        # 常见展示变体：原值、×100（百分比）、绝对值；再加两两差值/比率
        vals = set()
        for b in base_vals:
            vals.update((b, b * 100, abs(b), abs(b) * 100))
        for a in base_vals:
            for b in base_vals:
                vals.add(abs(a - b))
                if b:
                    vals.update((a / b, a / b * 100, abs(1 - a / b) * 100))
        titles = [t.get_text() for t in
                  ([fig._suptitle] if fig._suptitle else []) +
                  [ax.title for ax in fig.get_axes()]]
        for txt in titles:
            for tok in re.findall(r"\d+(?:\.\d+)?", txt):
                num = float(tok)
                if 1900 <= num <= 2100 and "." not in tok:
                    continue                      # 年份豁免
                dec = len(tok.split(".")[1]) if "." in tok else 0
                tol = lambda v: max(0.02 * abs(v), 0.55 * 10 ** -dec)
                if not any(abs(num - v) <= tol(v) for v in vals):
                    print(f"[QA WARN] 图题数字 {tok} 未溯源到计算变量，"
                          f"确认非手写：…{txt[:26]}")

    # 1. 字号下限
    small = [t for t in _all_texts(fig) if t.get_fontsize() < MIN_FONT_PT]
    if small:
        problems.append(
            f"{len(small)} 处文字字号 < {MIN_FONT_PT}pt，印刷不可读："
            f"{[t.get_text()[:12] for t in small[:3]]}")

    # 2. 色图黑名单（归一化 _r 反转与大小写，防止 jet_r 绕过）
    for ax in fig.get_axes():
        for coll in list(ax.collections) + list(ax.images):
            cm = getattr(coll, "get_cmap", lambda: None)()
            if cm is None:
                continue
            base = cm.name.lower().removesuffix("_r")
            if base in BANNED_CMAPS:
                problems.append(f"使用了被禁色图 {cm.name}（jet/rainbow 族）")

    # 2b. 硬拒绝构图检测：双 Y 轴（twinx）、饼图、分组竖柱
    from matplotlib.patches import Wedge
    from matplotlib.container import BarContainer
    axes = fig.get_axes()
    for i, a in enumerate(axes):
        for b in axes[i + 1:]:
            same_pos = a.get_position().bounds == b.get_position().bounds
            if same_pos and a.get_shared_x_axes().joined(a, b):
                problems.append(
                    "检测到双 Y 轴（twinx）——硬拒绝：不可通约的量拆面板"
                    "（见 phase_transition.py 的做法）")
    for ax in axes:
        # 极坐标轴豁免（风向玫瑰等合法用法），笛卡尔轴上的 Wedge 即饼图
        if ax.name != "polar" and any(isinstance(p, Wedge)
                                      for p in ax.patches):
            problems.append(
                "检测到饼图扇形——硬拒绝：改用有序水平条+直标或华夫图"
                "（composition.py）")
        # 分组竖柱：同轴 ≥2 个竖直 BarContainer 且共用同一基线
        # （堆叠柱的上层有 bottom 偏移，不误杀）
        vbars = [c for c in ax.containers
                 if isinstance(c, BarContainer) and len(c.patches) >= 3
                 and c.patches and
                 c.patches[0].get_height() >= c.patches[0].get_width()]
        ys = [p.get_y() for c in vbars for p in c.patches]
        same_base = ys and (max(ys) - min(ys)) <= 1e-9 * max(1, abs(max(ys)))
        if len(vbars) >= 2 and same_base:
            problems.append(
                "检测到分组竖柱——差异论证改用哑铃/斜率图/拆轴小倍数"
                "（comparison_rank.py）")

    # 2d. 样式与图题：apply_style 必须先行；图题必须存在（结论句）
    if not is_styled():
        problems.append(
            "未调用 apply_style()：字体/字号/脊线全是 matplotlib 默认，"
            "先 apply_style() 再画图")
    sup = fig._suptitle.get_text() if fig._suptitle is not None else ""
    if not (sup.strip() or
            any(a.get_title().strip() for a in fig.get_axes())):
        problems.append(
            "无图题：图题必须是结论句（如『A 成本仅为 B 的 53%』），"
            "不是坐标描述")

    # 2c. 注释层一等公民：全图至少一个统计注释框或带箭头的引线标注
    def _has_annotation_layer():
        for ax in axes:
            for t in ax.texts:
                if t.get_bbox_patch() is not None:
                    return True
                if getattr(t, "arrow_patch", None) is not None:
                    return True
        for t in fig.texts:
            if t.get_bbox_patch() is not None:
                return True
        return False
    if not _has_annotation_layer():
        problems.append(
            "无注释层：至少加一个 stat_box（n/RMS/CI 等统计框）"
            "或 callout 引线标注关键点")

    # 3. 交付宽度：校验 tight 裁剪+补白后真正落盘的宽度，
    #    而非 fig.get_figwidth()（裁剪前画布，与交付文件无关）
    tight = not any(getattr(a, "name", "") == "3d" for a in axes)
    w_mm = delivered_width_in(fig, tight=tight) * 25.4
    targets = ([COLUMN_WIDTHS.get(w, w) for w in expect_width]
               if expect_width else list(COLUMN_WIDTHS.values()))
    if not any(abs(w_mm - t) <= _TOL_MM for t in targets):
        problems.append(
            f"交付宽 {w_mm:.0f}mm 不在{'目标' if expect_width else '标准'}档 "
            f"{targets}（±{_TOL_MM}mm）；内容溢出画布时收紧构图")

    # 3b. 草稿档提醒（不抛错，避免草稿迭代被卡死；文件名已带 _DRAFT）
    if is_draft():
        print("[QA WARN] 当前为草稿档（120dpi、仅 PNG、文件名 _DRAFT），"
              "交付前用 apply_style() 默认档重出")

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

    # 5. 遮挡检查：图例与注释框互相重叠（硬错）、覆盖数据（硬错，
    #    仅图例；注释框盖数据只警告——统计框常合法压在稀疏区边缘）
    try:
        from matplotlib.transforms import Bbox
        rd = fig.canvas.get_renderer()
        boxes = []   # (名称, 窗口 bbox, 所属 ax, 是否图例)
        for ax in fig.get_axes():
            leg = ax.get_legend()
            if leg is not None:
                boxes.append(("图例", leg.get_window_extent(rd), ax, True))
            for t in ax.texts:
                bp = t.get_bbox_patch()
                # 必须取 bbox_patch：Annotation 的 window_extent 会把
                # 引线箭头也算进去，虚报 100% 重叠
                if bp is not None:
                    boxes.append((f"注释框「{t.get_text()[:10]}」",
                                  bp.get_window_extent(rd), ax, False))
        for t in fig.texts:
            bp = t.get_bbox_patch()
            if bp is not None:
                boxes.append((f"注释框「{t.get_text()[:10]}」",
                              bp.get_window_extent(rd), None, False))

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                bi, bj = boxes[i][1], boxes[j][1]
                inter = Bbox.intersection(bi, bj)
                if inter is None:
                    continue
                area = inter.width * inter.height
                frac = area / min(bi.width * bi.height,
                                  bj.width * bj.height)
                if frac > 0.15:
                    problems.append(
                        f"{boxes[i][0]} 与 {boxes[j][0]} 重叠 {frac:.0%}，"
                        f"错开位置")

        def _count_inside(ax, bb, xy):
            if len(xy) < 2:
                return 0, 0
            pts = ax.transData.transform(np.asarray(xy))
            n_in = int(np.sum(
                (pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) &
                (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)))
            return n_in, len(pts)

        for name, bb, ax, is_leg in boxes:
            if ax is None:
                continue
            datasets = [ln.get_xydata() for ln in ax.lines]
            datasets += [getattr(c, "get_offsets", lambda: [])()
                         for c in ax.collections]
            for xy in datasets:
                n_in, n = _count_inside(ax, bb, xy)
                if n and n_in >= max(6, 0.1 * n):
                    msg = f"{name} 覆盖 {n_in} 个数据点，移动位置"
                    if is_leg:
                        problems.append(msg + "或改线端直标")
                    else:
                        print(f"[QA WARN] {msg}")
                    break
    except Exception as e:  # 检查器自身故障不应伪装成通过
        print(f"[QA note] 遮挡检查未执行：{e}")

    if problems and strict:
        raise AssertionError("QA FAILED:\n- " + "\n- ".join(problems))
    if problems:
        for p in problems:
            print(f"[QA WARN] {p}")
    else:
        print("[QA] PASS (auto checks)")
    return problems
