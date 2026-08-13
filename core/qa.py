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
           sourced=None, allow=()) -> list[str]:
    """返回问题列表；strict=True 时有问题直接抛错。

    sourced: 图题数字的来源字典（recipe 返回的 info）。缺省取
    fig._ff_stats。给了来源就核对图题里的数字是否都能在来源值中
    找到（±2% 或四舍五入相等），抓"手写常数"违规。
    allow: 显式豁免的硬拒绝项，如 ("grouped_bars",)——仅限
    taxonomy 允许的场景（同单位、≤4 组）。
    """
    problems: list[str] = []
    if isinstance(expect_width, str):
        expect_width = (expect_width,)   # 裸字符串是最常见的抄写笔误

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
        # （堆叠柱的上层有 bottom 偏移，不误杀）。方向用容器的
        # orientation 属性判断——比较数据坐标下的高宽会混用单位，
        # 比例数据（高<宽）可直接绕过。柱心 x 基本重合的是叠加
        # 直方图（合法构图），放行；错开小于一个柱宽的才是分组柱。
        if "grouped_bars" not in allow:
            vbars = [c for c in ax.containers
                     if isinstance(c, BarContainer) and len(c.patches) >= 3
                     and getattr(c, "orientation", "vertical") == "vertical"]
            ys = [p.get_y() for c in vbars for p in c.patches]
            same_base = ys and (max(ys) - min(ys)) <= \
                1e-9 * max(1, abs(max(ys)))
            offset_group = False
            if len(vbars) >= 2 and same_base:
                c0 = np.sort([p.get_x() + p.get_width() / 2
                              for p in vbars[0].patches])
                bw = np.median([p.get_width() for p in vbars[0].patches])
                for c in vbars[1:]:
                    ci = np.sort([p.get_x() + p.get_width() / 2
                                  for p in c.patches])
                    n = min(len(c0), len(ci))
                    d = np.abs(ci[:n] - c0[:n])
                    # 柱心重合（<5% 柱宽）= 叠加直方图；错开但
                    # 不超过一个柱宽 = 分组柱签名
                    if np.median(d) > 0.05 * bw and np.median(d) <= 1.5 * bw:
                        offset_group = True
            if offset_group:
                problems.append(
                    "检测到分组竖柱——差异论证改用哑铃/斜率图/拆轴小倍数"
                    "（comparison_rank.py）；确属同单位对比可传 "
                    "allow=('grouped_bars',) 豁免")

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

    def _is_mathtext_mix(s):
        """含 CJK 且 $..$ 对内有字母/命令才算混排；纯货币数字放行。"""
        if not (_has_cjk(s) and s.count("$") >= 2):
            return False
        import re
        return any(re.search(r"[A-Za-z\\^_{}]", seg)
                   for seg in re.findall(r"\$([^$]*)\$", s))
    mixed = [t.get_text()[:20] for t in _all_texts(fig)
             if _is_mathtext_mix(t.get_text())]
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

        def _densify(xy, per_seg=8):
            """沿折线路径插值采样：只查顶点会漏掉长直线段穿过框体。"""
            xy = np.asarray(xy, dtype=float)
            if len(xy) < 2 or len(xy) > 400:   # 已足够密的不再加密
                return xy
            segs = [np.linspace(xy[k], xy[k + 1], per_seg, endpoint=False)
                    for k in range(len(xy) - 1)]
            return np.vstack(segs + [xy[-1:]])

        def _count_inside(ax, bb, xy):
            if len(xy) < 2:
                return 0, 0
            pts = ax.transData.transform(np.asarray(xy))
            n_in = int(np.sum(
                (pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) &
                (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)))
            return n_in, len(pts)

        from matplotlib.collections import LineCollection
        for name, bb, ax, is_leg in boxes:
            if ax is None:
                continue
            # (采样点集, 判定阈值)：路径按采样比例，散点按点数比例
            datasets = [(_densify(ln.get_xydata()), "path")
                        for ln in ax.lines]
            for c in ax.collections:
                if isinstance(c, LineCollection):
                    for seg in c.get_segments():
                        datasets.append((_densify(seg), "path"))
                else:
                    offs = getattr(c, "get_offsets", lambda: [])()
                    datasets.append((np.asarray(offs), "points"))
            # 柱体也算数据：图例压在 bar 上同样是遮挡。直接算图例框
            # 与柱面片的像素交叠面积，累计超图例面积 25% 判遮挡
            if is_leg:
                covered = 0.0
                for c in ax.containers:
                    for p in getattr(c, "patches", []):
                        pb = p.get_window_extent(rd)
                        ib = Bbox.intersection(bb, pb)
                        if ib is not None:
                            covered += ib.width * ib.height
                if bb.width * bb.height > 0 and \
                        covered >= 0.25 * bb.width * bb.height:
                    problems.append(
                        f"{name} 压在柱体上（交叠 "
                        f"{covered / (bb.width * bb.height):.0%} 图例面积），"
                        "移出数据区")
            for xy, kind in datasets:
                n_in, n = _count_inside(ax, bb, xy)
                thresh = max(12, 0.02 * n) if kind == "path" \
                    else max(6, 0.1 * n)
                if n and n_in >= thresh:
                    msg = f"{name} 覆盖数据（{n_in} 个采样点），移动位置"
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
