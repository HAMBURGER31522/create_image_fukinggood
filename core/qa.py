"""出图后自动自检：能自动查的自动查，查不了的留给 agent 目测。

用法（脚本末尾）::

    from core import run_qa
    run_qa(fig, expect_width=("onehalf",))   # 不合格直接抛 AssertionError
"""
from __future__ import annotations

import io
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .colors import BANNED_CMAPS, check_accessibility
from .style import (COLUMN_WIDTHS, MAX_HEIGHT_MM, delivered_width_in,
                    font_report, font_weights, is_styled, is_draft,
                    current_preset)
from .layout import MAX_PANELS
from . import _probe

# 字号下限随档走：nature 档 Nature 规定最小 5pt；cn 档排版缩放后仍需 ≥6pt
_MIN_FONT_BY_PRESET = {"nature": 5.0, "cn": 6.5}
MIN_FONT_PT = 6.5          # 兼容旧引用；实际判定用 _min_font()
_TOL_MM = 3.0


def _min_font() -> float:
    return _MIN_FONT_BY_PRESET.get(current_preset(), MIN_FONT_PT)


def _is_colored(color, tol: float = 0.06) -> bool:
    """是否为彩色文字（黑/白/中性灰不算）。nature 档禁彩色文字。"""
    try:
        r, g, b = mcolors.to_rgb(color)
    except (ValueError, TypeError):
        return False
    return (max(r, g, b) - min(r, g, b)) > tol


def _all_axes(fig):
    """fig 的全部坐标区，**含 inset**。

    `fig.get_axes()` 不包含 `ax.inset_axes()` 建出来的放大窗——它挂在
    父轴的 child_axes 上。只用 get_axes() 的后果是放大窗整体在 QA
    视野之外：它的标题被统计框压掉、刻度叠字、数据被遮，全都查不出来。
    """
    out, seen = [], set()

    def _walk(ax):
        if id(ax) in seen:
            return
        seen.add(id(ax))
        out.append(ax)
        for ch in getattr(ax, "child_axes", []) or []:
            _walk(ch)

    for a in fig.get_axes():
        _walk(a)
    return out


def _all_texts(fig):
    texts = list(fig.texts)
    # 图级图例与 3D 的 z 轴此前整体不在视野内，连带让**全部**文字类检查
    # （字号下限 / 豆腐块 / nature 禁彩色文字 / 对比度）对它们失明。
    for lg in getattr(fig, "legends", []):
        texts += list(lg.get_texts())
    for ax in _all_axes(fig):
        texts += ax.texts
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += ax.get_xticklabels() + ax.get_yticklabels()
        if getattr(ax, "name", "") == "3d":
            texts.append(ax.zaxis.label)
            texts += ax.get_zticklabels()
        leg = ax.get_legend()
        if leg is not None:
            texts += leg.get_texts()
    return [t for t in texts if t.get_text().strip()]


_ALLOW_CODES = frozenset({
    "grouped_bars", "unsourced", "overlap", "accessibility",
    "unexplained_band", "number_conflict", "duplicate_series",
    "clim_mismatch", "axis_slack", "sparse_line", "unit_axis_range",
    "incommensurable", "text_contrast",
})


def run_qa(fig, expect_width=None, strict: bool = True,
           sourced=None, allow=()) -> list[str]:
    """返回问题列表；strict=True 时有问题直接抛错。

    sourced: 图题数字的来源字典（recipe 返回的 info）。缺省取
    fig._ff_stats。给了来源就核对图题里的数字是否都能在来源值中
    找到（±2% 或四舍五入相等），抓"手写常数"违规。
    allow: 显式豁免的硬拒绝项，如 ("grouped_bars",)——仅限
    taxonomy 允许的场景（同单位、≤4 组）。可用码：
    grouped_bars / unsourced / overlap / accessibility /
    unexplained_band / number_conflict / duplicate_series /
    clim_mismatch / axis_slack / sparse_line / unit_axis_range /
    incommensurable / text_contrast。
    未知的码直接抛错（拼错时静默无效比报错更伤）。
    豁免会记入 fig._ff_qa_waived 并标进文件名。
    """
    unknown = set(allow) - _ALLOW_CODES
    if unknown:
        raise ValueError(
            f"未知的 allow 码 {sorted(unknown)}；可用：{sorted(_ALLOW_CODES)}"
            f"。拼错的码静默无效——图照样被拦，而你以为已经豁免了")
    def _is_canvas_like(ax):
        """流程图/示意图这类"画布轴"：坐标轴关掉、图上的框本身就是内容。
        分组柱检测对它没有意义。

        3D 轴的 `axison` 恒为 False 但它是真面板——与下面 `_is_canvas`
        的护栏保持一致，否则每个 3D 面板都会被当成画布。
        """
        if getattr(ax, "name", "") == "3d":
            return False
        return not ax.axison

    problems: list[str] = []
    _waived: list[str] = []

    def _hard(msg: str, code: str) -> None:
        """硬错，但留 allow 豁免位。

        豁免走 [QA WAIVED] 并记入 fig._ff_qa_waived，save_figure 会把
        文件名标成 _QAWAIVED——豁免留痕，不会悄悄混进交付物。
        """
        if code in allow:
            _waived.append(msg)
            print(f"[QA WAIVED] {msg}")
        else:
            problems.append(msg)

    # 先按最终版面重排"轴外"统计框：它们多半在 set_title 之前就放好了，
    # 那时量不到标题高度，位置只能靠猜
    try:
        from .annotate import reflow_outside
        reflow_outside(fig)
    except Exception as e:
        print(f"[QA note] 轴外重排未执行：{e}")
    if isinstance(expect_width, (str, int, float)):
        expect_width = (expect_width,)   # 裸标量是最常见的抄写笔误

    # 0. 图题数字溯源（有 _ff_stats/sourced 才查；只警告不阻断——
    #    差值/比率等合法派生数字无法穷举，误杀比漏报更伤）
    sourced = sourced if sourced is not None else getattr(fig, "_ff_stats",
                                                          None)
    if not sourced and "unsourced" not in allow:
        # 不给来源 = 整块溯源检查静默跳过，手写常数就是这样漏出去的。
        # 但"没启用检查"本身不是画错，只警告；**给了来源却对不上**才拦。
        print("[QA WARN] 未提供数字来源（fig._ff_stats 或 sourced=）："
              "图题与注释里的数字未经溯源，手写常数不会被发现。"
              "交付图建议补上")
    if sourced:
        # 这里曾有一句 `import re`：函数体内的 import 会把 re 变成**局部
        # 变量**，遮蔽模块级的 re，于是 sourced 为空时后面用到 re 的检查
        # 全部落进 except 静默跳过。模块顶层已 import，不要在这里再来一次。

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
        # 常见展示变体：原值、×100（百分比）、绝对值
        vals = set()
        for b in base_vals:
            vals.update((b, b * 100, abs(b), abs(b) * 100))
        # 两两派生：差、比、**积**。积是实测漏掉的一类——
        # 图题里写 "φ₉₀ = 0.874%，换口径后 1.545%" 时，后者往往是
        # p90 * ratio 算出来的，只查差与比会把它误判成手写常数。
        for a in base_vals:
            for b in base_vals:
                vals.add(abs(a - b))
                vals.add(a * b)
                vals.add(a * b * 100)
                if b:
                    vals.update((a / b, a / b * 100, abs(1 - a / b) * 100))
        # 数组的极值/长度：stat_box 里写 "扫 2.62–84.00" 这类区间时，
        # 两个端点是 arr.min()/arr.max()，它们不在逐元素展开里
        def _arrays(v):
            if isinstance(v, dict):
                for x in v.values():
                    yield from _arrays(x)
            elif isinstance(v, (list, tuple, np.ndarray)):
                arr = np.ravel(np.asarray(v, dtype=object))
                nums = []
                for e in arr:
                    try:
                        f = float(e)
                        if np.isfinite(f):
                            nums.append(f)
                    except (TypeError, ValueError):
                        pass
                if nums:
                    yield np.array(nums, dtype=float)

        for arr in _arrays(sourced):
            for f in (np.min, np.max, np.mean, np.ptp, np.sum):
                try:
                    v = float(f(arr))
                    vals.update((v, v * 100, abs(v), abs(v) * 100))
                except Exception:
                    pass
            vals.add(float(len(arr)))
            av = np.abs(arr)
            vals.update((float(av.max()), float(av.max()) * 100,
                         float(av.min()), float(av.min()) * 100))
        # 图题之外，stat_box / callout / 直标里的手写常数一样是事实错误级
        # 缺陷——独立评审就是在 stat_box 里查出 [0.87, 36.4] 这类手填值。
        scan = ([fig._suptitle] if fig._suptitle else []) +             [ax.title for ax in fig.get_axes()]
        for ax in fig.get_axes():
            scan += list(ax.texts)
        scan += list(fig.texts)
        titles = [t.get_text() for t in scan if t is not None]

        def _is_label(txt, tok):
            """结构性标签不是统计量，不该要求溯源。

            实测误报来源：路线图的途经序号（孤零零一个"5"）、分位线标签
            （"50%"、"90%"）、坐标刻度式短文本。它们由构图本身决定，
            不是从数据算出来的量。
            """
            t = txt.strip()
            # 整条文本就是这个数（可带 % / 单位）→ 序号或刻度标签
            if t.rstrip("%").strip() == tok:
                return True
            # 常见分位/置信水平
            if tok in ("50", "90", "95", "99", "0.5", "0.9", "0.95", "0.99"):
                return True
            # 约定常数：显著性水平、置信水平。它们由统计规范给定，
            # 不是从数据算出来的量。
            if tok in ("0.05", "0.01", "0.1", "1.96", "2.58"):
                return True
            return False

        for txt in titles:
            for tok in re.findall(r"\d+(?:\.\d+)?", txt):
                num = float(tok)
                if 1900 <= num <= 2100 and "." not in tok:
                    continue                      # 年份豁免
                dec = len(tok.split(".")[1]) if "." in tok else 0
                tol = lambda v: max(0.02 * abs(v), 0.55 * 10 ** -dec)
                if _is_label(txt, tok):
                    continue
                if not any(abs(num - v) <= tol(v) for v in vals):
                    msg = (f"数字 {tok} 未溯源到计算变量（疑似手写常数）："
                           f"…{txt[:26]}")
                    if "unsourced" in allow:
                        _waived.append(msg)
                        print(f"[QA WAIVED] {msg}")
                    else:
                        problems.append(msg)

    # 1. 字号下限（按档：nature 5pt / cn 6.5pt）
    _minpt = _min_font()
    small = [t for t in _all_texts(fig) if t.get_fontsize() < _minpt]
    if small:
        problems.append(
            f"{len(small)} 处文字字号 < {_minpt}pt，印刷不可读："
            f"{[t.get_text()[:12] for t in small[:3]]}")
    # nature 档还有上限：Nature 规定除面板标签(8pt)外正文最大 7pt
    if current_preset() == "nature":
        big = [t for t in _all_texts(fig)
               if t.get_fontsize() > 7.0 and t.get_text().strip() not in
               {chr(ord("a") + i) for i in range(MAX_PANELS)}]
        if big:
            problems.append(
                f"{len(big)} 处文字 > 7pt，超 Nature 正文字号上限："
                f"{[(t.get_text()[:10], t.get_fontsize()) for t in big[:3]]}")

    # 1b. 字体字重一致性——本 skill 历史上最伤观感的缺陷。
    #     思源系列常只装 Heavy(900) 一个字面，matplotlib 拿它当 Regular，
    #     中文全渲染成粗黑块紧挨 400 字重的拉丁数字，看起来像渲染坏了。
    rep = font_report()
    if rep:
        if rep.get("dropped"):
            print("[QA note] 已跳过无 Regular 字面的字体："
                  f"{[n for n, _ in rep['dropped']]}")
        if not rep.get("cjk"):
            problems.append(
                "字体栈里没有可用中文字体（含 Regular 字面的），"
                "中文会变豆腐块或被迫用粗黑字面")
        bad_w = [(n, font_weights(n)) for n in rep.get("latin", []) +
                 rep.get("cjk", []) if not any(350 <= w <= 550
                                               for w in font_weights(n))]
        if bad_w:
            problems.append(
                f"字体栈含无 Regular 字面的家族 {bad_w}，"
                f"中文/拉丁会出现字重不一致（一半粗一半细）")

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
        # 不再用 `if code not in allow` 直接跳过检测：那样豁免不留痕，
        # 而 docstring 承诺"豁免会记入 fig._ff_qa_waived 并标进文件名"。
        # 检测照跑，报告走 _hard()，由它统一处理豁免与留痕。
        if not _is_canvas_like(ax):
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
                _hard(
                    "检测到分组竖柱——差异论证改用哑铃/斜率图/拆轴小倍数"
                    "（comparison_rank.py）；多组分布对比改用 raincloud/"
                    "ridgeline（raincloud.py）；确属同单位对比可传 "
                    "allow=('grouped_bars',) 豁免", "grouped_bars")

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

    # 3c. 图高上限：Nature 明文 170mm（整页图含图注的可用高度）
    h_mm = fig.get_figheight() * 25.4
    if h_mm > MAX_HEIGHT_MM + _TOL_MM:
        problems.append(
            f"图高 {h_mm:.0f}mm 超上限 {MAX_HEIGHT_MM:.0f}mm，"
            f"减少行数或压缩面板高度")

    # 3d. 面板数：Nature 建议整页图 ≤6 个面板（小倍数网格是正当例外）
    n_panel = sum(1 for a in axes
                  if a.get_visible() and a.get_label() != "<colorbar>")
    if n_panel > MAX_PANELS:
        print(f"[QA WARN] {n_panel} 个面板超 Nature 建议上限 {MAX_PANELS}——"
              f"若非小倍数网格，拆成两张图")

    # 3e. 档规范：nature 档硬性对齐 Nature 官方要求
    if current_preset() == "nature":
        gridded = [a for a in axes
                   if any(ln.get_visible() for ln in
                          a.get_xgridlines() + a.get_ygridlines())]
        if gridded:
            problems.append(
                f"{len(gridded)} 个轴开着背景网格——Nature 明文 "
                f"'No background gridlines'，nature 档必须关掉")
        colored = [t.get_text()[:12] for t in _all_texts(fig)
                   if _is_colored(t.get_color())]
        if colored:
            problems.append(
                f"{len(colored)} 处彩色文字——Nature 明文 'Avoid coloured "
                f"text'，语义改走 keyline/key（框线、标记），文字用黑色："
                f"{colored[:3]}")
        heavy = [ln for a in axes for ln in a.lines
                 if ln.get_linewidth() > 1.0 + 1e-9]
        if heavy:
            problems.append(
                f"{len(heavy)} 条线宽 > 1pt，超 Nature 线宽区间 0.25–1pt")

    # 7. 可达性：分类色在三类色盲与灰度下是否仍可区分。
    #    若已用不同 marker/线型做冗余编码，灰度不可分降级为提示。
    try:
        used, marks, styles = [], set(), set()
        for ax in axes:
            for ln in ax.lines:
                if ln.get_visible() and len(ln.get_xydata()) > 1:
                    used.append(ln.get_color())
                    marks.add(str(ln.get_marker()))
                    styles.add(str(ln.get_linestyle()))
            for c in ax.containers:
                ps = getattr(c, "patches", [])
                if ps:
                    used.append(ps[0].get_facecolor())
        # 去掉 'None'/'none' 这类空标记后仍有多种形状或线型 = 有冗余编码
        real_marks = {m for m in marks if m.lower() not in ("none", "")}
        redundant = len(real_marks) > 1 or len(styles) > 1
        # 去重后只查前 6 个（更多分类色本身就该换构图）；跳过白与近灰
        uniq, seen_c = [], set()
        for c in used:
            rgb = mcolors.to_rgb(c)
            key = tuple(np.round(rgb, 3))
            if key in seen_c:
                continue
            seen_c.add(key)
            if max(rgb) > 0.97 and min(rgb) > 0.97:
                continue                       # 纯白（marker 描边等）
            if max(rgb) - min(rgb) < 0.04:
                continue                       # 中性灰：基线色，不参与分类
            uniq.append(mcolors.to_hex(rgb))
        _msgs = check_accessibility(uniq[:6], redundant=redundant)
        # redundant=True 只抑制灰度项，返回的即纯色盲色距问题。
        # SKILL.md §3d 给的判据（N=2→0.93 … N=6→0.05）说的正是色距，
        # 灰度可分性从未被声明为硬门槛——把灰度也升硬错会顶掉一批
        # 构图正当的图（实测 cluster_scatter / route_map / composition）。
        # 所以：色距不足且无形状/线型冗余 = 硬错；灰度不足 = 提示。
        _cvd = set(check_accessibility(uniq[:6], redundant=True))
        for msg in _msgs:
            full = f"可达性：{msg}"
            if msg in _cvd and not redundant:
                _hard(full, "accessibility")
            else:
                print(f"[QA WARN] {full}")
    except Exception as e:
        print(f"[QA note] 可达性检查未执行：{e}")

    # 8. 未解释的视觉编码：出现了填充带/第二组标记，却既无图例也无直标。
    #    这是"重画后信息量反而比原图少"的典型病灶——读者看到一条浅色带
    #    却无从知道它是 CI 还是范围还是分位区间。
    from matplotlib.collections import PolyCollection as _PolyC
    for ax in axes:
        # 精确类名比对在 matplotlib 3.10 上失效：fill_between 的返回类
        # 改名为 FillBetweenPolyCollection，这条检查因此一度完全不执行。
        # 用 isinstance 认子类；同时排除有 array 的（hexbin/色块那类由
        # 色标解释含义，不属于"无说明的填充带"）。
        polys = [c for c in ax.collections
                 if isinstance(c, _PolyC) and c.get_array() is None]
        # 只认"图例条目"或"**贴着这条带**的直标"。原先是"轴上有任何文字
        # 就算已解释"，而检查 2c 又强制每图必须有 stat_box（走 ax.text），
        # 于是凡是能过 QA 的图，这条检查必然被自己静音——喊都没喊过。
        has_key = bool(ax.get_legend_handles_labels()[1])
        # 只查**包住某条曲线**的填充：那才是"这条带是 CI 还是范围还是分位
        # 区间"的歧义所在。雨云图的密度云、几何示意图的多边形同样是
        # PolyCollection，但含义由图种与轴标题给定，不存在这种歧义
        # ——实测 raincloud / percolation_geometry 就是被这样误杀的。
        # 置信带的**上下界都在变**；"从基线填到某条曲线"（雨云的密度云、
        # ridgeline、面积图）下界恒定，它不是区间，没有"这是 CI 还是分位"
        # 的歧义——实测 raincloud 就是这样被误杀的。
        if polys:
            _keep = []
            for _pc in polys:
                try:
                    _v = np.vstack([q.vertices for q in _pc.get_paths()])
                    _v = _v[np.all(np.isfinite(_v), axis=1)]
                    if len(_v) < 4:
                        continue
                    _xs = np.round(_v[:, 0], 9)
                    _base = np.array([_v[_xs == u, 1].min()
                                      for u in np.unique(_xs)])
                    _span = np.ptp(_v[:, 1])
                    if _span > 0 and np.ptp(_base) / _span > 0.02:
                        _keep.append(_pc)      # 下界确实在变 = 真区间
                except Exception:
                    _keep.append(_pc)
            polys = _keep
        if polys:
            try:
                _lns = [ln for ln in ax.lines if ln.get_visible()
                        and len(ln.get_xydata()) >= 3]
                _wrap = []
                for _pc in polys:
                    _tr = _pc.get_transform()
                    _pp = [_tr.transform_path(q) for q in _pc.get_paths()]
                    for _ln in _lns:
                        pts = ax.transData.transform(
                            np.asarray(_ln.get_xydata(), dtype=float))
                        pts = pts[np.all(np.isfinite(pts), axis=1)]
                        if len(pts) < 3:
                            continue
                        ins = sum(any(q.contains_point((px, py))
                                      for q in _pp) for px, py in pts)
                        if ins / len(pts) >= 0.6:
                            _wrap.append(_pc)
                            break
                polys = _wrap
            except Exception as e:
                print(f"[QA note] 填充带归属判定未执行：{e}")
        if polys and not has_key:
            # 判据是"文字落在**带本身**上"，不是"落在带的包围盒里"：
            # 一条对角置信带的包围盒几乎等于整个坐标区，用包围盒判等于
            # 不判——左上角的 stat_box 也会被算成"解释了这条带"。
            try:
                rd0 = fig.canvas.get_renderer()
                for _pc in polys:
                    tr = _pc.get_transform()
                    paths = [tr.transform_path(pp) for pp in _pc.get_paths()]
                    for t in ax.texts:
                        if not t.get_text().strip():
                            continue
                        tb = t.get_window_extent(rd0)
                        cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
                        r = 0.5 * max(tb.width, tb.height)
                        if any(pp.contains_point((cx, cy), radius=r)
                               for pp in paths):
                            has_key = True
                            break
                    if has_key:
                        break
            except Exception:
                pass
        if polys and not has_key:
            _hard("存在填充区域（置信带/范围）但无图例也无直标，"
                  "读者无法得知其含义——补图例条目或就近直标",
                  "unexplained_band")
            break

    # 3b. 草稿档提醒（不抛错，避免草稿迭代被卡死；文件名已带 _DRAFT）
    if is_draft():
        print("[QA WARN] 当前为草稿档（120dpi、仅 PNG、文件名 _DRAFT），"
              "交付前用 apply_style() 默认档重出")

    # 4a. 中文与 mathtext 混排（同串含 CJK 和 $..$ 整串走 mathtext → 豆腐块）
    def _has_cjk(s):
        return any("\u4e00" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef"
                   for ch in s)

    def _is_mathtext_mix(s):
        """含 CJK 且有成对未转义 $ 即混排。

        matplotlib 是否走 mathtext 只取决于 $ 是否成对，与 $ 之间
        的内容无关——"成本 $1200 与 $3400" 同样整串进 mathtext，
        中文全变豆腐块。货币金额必须写 \\$（转义）或全角 ＄。
        """
        if not _has_cjk(s):
            return False
        import re
        return len(re.findall(r"(?<!\\)\$", s)) >= 2
    mixed = [t.get_text()[:20] for t in _all_texts(fig)
             if _is_mathtext_mix(t.get_text())]
    if mixed:
        problems.append(
            f"{len(mixed)} 处中文与 $mathtext$ 混排（会豆腐块），"
            f"改用 Unicode 数学字符（₀ ⁻¹ φ ε √ 等）；"
            f"货币金额写 \\$1200（反斜杠转义）或全角 ＄：{mixed[:2]}")

    # 4b. 豆腐块：渲染一次，同时捕获 warnings 与 matplotlib logging 两条通道。
    #     mathtext 解析结果带 lru_cache，前面的 delivered_width_in 已 draw 过
    #     一次，不清缓存则 mathtext 路径的字形警告永远抓不到
    import logging
    import warnings
    try:
        from matplotlib.mathtext import MathTextParser
        if hasattr(MathTextParser, "_parse_cached"):
            MathTextParser._parse_cached.cache_clear()
    except Exception:
        pass
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
        # 把缺失的字符本身报出来。只说"N 处字形缺失"等于让人再挖一遍，
        # 而 matplotlib 的警告里其实带着码位与 Unicode 名字。
        import re as _re
        miss = set()
        for w in wlist:
            m = _re.search(r"Glyph (\d+) \(\\N\{([^}]+)\}\)", str(w.message))
            if m:
                miss.add(f"U+{int(m.group(1)):04X}({m.group(2)})")
            elif "missing from font" in str(w.message):
                m2 = _re.search(r"Glyph (\d+)", str(w.message))
                if m2:
                    miss.add(f"U+{int(m2.group(1)):04X}")
        for m3 in _re.finditer(r"Glyph (\d+)", stream.getvalue()):
            miss.add(f"U+{int(m3.group(1)):04X}")
        detail = ("：" + "、".join(sorted(miss)[:4])) if miss else ""
        problems.append(
            f"{n_glyph} 处字形缺失（豆腐块），检查字体回退与混排{detail}")

    # 5. 遮挡检查：注释层压住数据 = 硬错。
    #    以前这里只 print WARN，结果是"喊了照样落盘"——独立评审在一批
    #    12 张图里查出 4 处数据被完全吞掉（整条系列不见、直标被切一半），
    #    每一处 QA 都喊过。喊而不拦等于没有检查。
    #    需要豁免时显式传 allow=("overlap",)，会打 [QA WAIVED] 并记录到
    #    fig._ff_qa_waived，save_figure 会把文件名标成 _QAWAIVED。
    FIELD_COVER_MAX = 0.25      # 框压住场类色块的面积比上限
    SERIES_SWALLOW_MAX = 0.8    # 单条系列被吞掉的采样点比例上限
    PATH_PTS_MAX = 6            # 路径采样点绝对阈值（原为 12，漏掉了 P0）
    CLOUD_RATIO_MAX = 0.05      # 散点云按比例判：密集云里几个点被压不算病
    # 像素兜底只负责抓"枚举漏掉的极端情况"（满铺的场底下 90%–100%），
    # 细粒度的压线/压点/压直标由上面三条各自负责，这里不重复报。
    PIXEL_INK_ERR = 0.20
    PIXEL_INK_WARN = 0.08

    def _hit(msg):
        if "overlap" in allow:
            _waived.append(msg)
            print(f"[QA WAIVED] {msg}")
        else:
            problems.append(msg)

    from matplotlib.transforms import Bbox

    def _is_canvas(ax):
        """该轴是不是"画布"而非"数据坐标区"。

        流程图/示意图（pipeline_diagram 这类）关掉了坐标轴，图上的方框
        本身就是内容——工序节点、泳道标签——不是压在数据上的注释。
        对这种轴跑遮挡检查必然全是误报。判据：坐标轴已关闭，或轴上
        根本没有数据系列（没数据就谈不上遮数据）。
        """
        if getattr(ax, "name", "") == "3d":
            return False          # Axes3D.axison 恒为 False，但它是真面板
        if not ax.axison:
            return True
        return not any(len(p) for _, p in _probe.series_samples(ax)) and             not _probe.has_field(ax)

    try:
        rd = fig.canvas.get_renderer()
        boxes = []       # (名称, 窗口 bbox, 所属 ax, 是否图例, artist)
        bare_texts = []  # (名称, bbox, ax, artist) 无边框直标
        for ax in _all_axes(fig):
            if _is_canvas(ax):
                continue
            leg = ax.get_legend()
            if leg is not None:
                boxes.append(("图例", leg.get_window_extent(rd), ax, True,
                              leg))
            for t in ax.texts:
                bp = t.get_bbox_patch()
                # 必须取 bbox_patch：Annotation 的 window_extent 会把
                # 引线箭头也算进去，虚报 100% 重叠
                if bp is not None:
                    boxes.append((f"注释框「{t.get_text()[:10]}」",
                                  bp.get_window_extent(rd), ax, False, t))
                elif t.get_text().strip():
                    bare_texts.append((f"直标「{t.get_text()[:12]}」",
                                       t.get_window_extent(rd), ax, t))
            # 排版文字同样会被压：把统计框移到"轴外"最容易压死的就是
            # 面板标题与图题——那往往是整张图的结论句。
            for art, nm in ((ax.title, "面板标题"),
                            (ax.xaxis.label, "x 轴标题"),
                            (ax.yaxis.label, "y 轴标题")):
                if art.get_text().strip():
                    bare_texts.append((f"{nm}「{art.get_text()[:14]}」",
                                       art.get_window_extent(rd), ax, art))
            for tk in ax.get_xticklabels() + ax.get_yticklabels():
                if tk.get_text().strip():
                    bare_texts.append((f"刻度「{tk.get_text()[:8]}」",
                                       tk.get_window_extent(rd), ax, tk))
        for t in fig.texts:
            bp = t.get_bbox_patch()
            if bp is not None:
                boxes.append((f"注释框「{t.get_text()[:10]}」",
                              bp.get_window_extent(rd), None, False, t))
        _sup = getattr(fig, "_suptitle", None)
        if _sup is not None and _sup.get_text().strip():
            bare_texts.append(("图题", _sup.get_window_extent(rd), None, _sup))

        # 5a. 注释层互相重叠
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                bi, bj = boxes[i][1], boxes[j][1]
                inter = Bbox.intersection(bi, bj)
                if inter is None:
                    continue
                area = inter.width * inter.height
                frac = area / max(1e-9, min(bi.width * bi.height,
                                            bj.width * bj.height))
                if frac > 0.15:
                    _hit(f"{boxes[i][0]} 与 {boxes[j][0]} 重叠 {frac:.0%}，"
                         f"错开位置")

        # 5b. 任何带框注释/图例压住任何无边框直标 —— 直标被切掉一半是
        #     从渲染图上最难看出、对读者伤害最大的一类缺陷
        for name, bb, ax_b, is_leg, art in boxes:
            for tname, tb, ax_t, tart in bare_texts:
                if tart is art or tb.width * tb.height <= 0:
                    continue
                inter = Bbox.intersection(bb, tb)
                if inter is None:
                    continue
                frac = (inter.width * inter.height) / (tb.width * tb.height)
                if "标题" in tname or tname == "图题":
                    lim = 0.05          # 标题少读几个字就丢结论
                elif tname.startswith("刻度"):
                    lim = 0.30          # 刻度多且密，蹭到一角不算病
                else:
                    lim = 0.15
                if frac > lim:
                    _hit(f"{name} 压住{tname}（{frac:.0%}），移动其一")

        # 5b2. 图题压面板标题 / 两个面板标题互压。5b 查的是"带框注释压
        #      直标"，两个都没有框的标题重叠就从缝里漏了——实测 q2_1 与
        #      q4_1 的 suptitle 正好压在面板标题上，结论句被切断，而 QA
        #      全绿放行。刻度不在此列（刻度重叠由 6b 专门管）。
        _titles = [t for t in bare_texts
                   if t[0] == "图题" or t[0].startswith("面板标题")]
        for _i in range(len(_titles)):
            for _j in range(_i + 1, len(_titles)):
                _ni, _bi = _titles[_i][0], _titles[_i][1]
                _nj, _bj = _titles[_j][0], _titles[_j][1]
                if _titles[_i][3] is _titles[_j][3]:
                    continue
                _it = Bbox.intersection(_bi, _bj)
                if _it is None:
                    continue
                _fr = (_it.width * _it.height) / max(
                    1e-9, min(_bi.width * _bi.height, _bj.width * _bj.height))
                if _fr > 0.10:
                    _kind = ("图题与面板标题重叠"
                             if "图题" in (_ni, _nj) else "两个面板标题重叠")
                    _hit(f"{_kind} {_fr:.0%}（{_ni} / {_nj}）——结论句被切断，"
                         f"调 suptitle 的 y、收紧面板标题，或用 tight_layout")

        # 5b3. 直标互压。5b 查"带框注释压直标"，5b2 查标题互压，两个**无框
        #      直标**互相叠字仍然漏网——实测 cohort 斜率图左侧两个高亮标签
        #      叠在一起，QA 全绿放行，而 SKILL.md 目测清单第 4 条明写
        #      "文字无重叠"。刻度之间的重叠由 6b 专管，这里不重复。
        # 刻度类由 6b 专管。标题**要留在池子里**：5b2 只做标题×标题，
        # 把标题整类剔除会让"无框直标压面板标题"变成零覆盖——而面板标题
        # 按本项目的规矩就是结论句。重复报告改为在配对时跳过标题×标题。
        _bare = [t for t in bare_texts if not t[0].startswith("刻度")]

        def _is_title(nm):
            return nm == "图题" or nm.startswith("面板标题")
        for _i in range(len(_bare)):
            for _j in range(_i + 1, len(_bare)):
                _ni, _bi2, _, _ai2 = _bare[_i]
                _nj, _bj2, _, _aj2 = _bare[_j]
                if _ai2 is _aj2:
                    continue
                if _is_title(_ni) and _is_title(_nj):
                    continue        # 标题×标题由 5b2 专管，别报两遍
                if min(_bi2.width * _bi2.height,
                       _bj2.width * _bj2.height) <= 0:
                    continue
                _it2 = Bbox.intersection(_bi2, _bj2)
                if _it2 is None:
                    continue
                _fr2 = (_it2.width * _it2.height) / max(
                    1e-9, min(_bi2.width * _bi2.height,
                              _bj2.width * _bj2.height))
                if _fr2 > 0.15:
                    _hit(f"直标互相重叠 {_fr2:.0%}（{_ni} / {_nj}）——"
                         f"错开位置或缩短其一")

        # 5c. 压数据：场按面积、曲线按吞没率与绝对点数
        for name, bb, ax, is_leg, art in boxes:
            if ax is None:
                continue
            # 3D 轴跳过"面积占比"判据：Poly3D / Line3DCollection 取不到自己
            # 的窗口包围盒，回退成整个坐标区后必然判成"框压满场"，而三维
            # 投影的四角本来就是空的。这类交给 5d 的像素兜底，它算得准。
            _fields = ([] if getattr(ax, "name", "") == "3d"
                       else list(ax.images) + list(ax.collections))
            for fa in _fields:
                if not fa.get_visible() or \
                        fa.__class__.__name__ not in _probe.FIELD_CLASSES:
                    continue
                try:
                    fb = fa.get_window_extent(rd)
                except Exception:
                    fb = ax.get_window_extent()
                inter = Bbox.intersection(bb, fb)
                if inter is None:
                    continue
                cover = (inter.width * inter.height) / \
                    max(1e-9, bb.width * bb.height)
                if cover > FIELD_COVER_MAX:
                    _hit(f"{name} 压在场图上（框底 {cover:.0%} 是色块），"
                         f"改 stat_box(outside='top') 放到坐标区外")
                    break
            for sname, pts in _probe.series_samples(ax):
                # patch 只用于自动选位，不参与吞没判定：装饰性小色块
                # （电极带、可行域底色）按外接框 4×4 采样后必然"整体落框内"，
                # 那不是数据系列被吞掉
                if len(pts) == 0 or sname.startswith("patch"):
                    continue
                inside = ((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) &
                          (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1))
                n_in = int(inside.sum())
                ratio = n_in / len(pts)
                if len(pts) >= 5 and ratio >= SERIES_SWALLOW_MAX:
                    _hit(f"{name} 吞掉整条数据系列（{ratio:.0%} 采样点在框内），"
                         f"读者会以为该系列不存在")
                elif len(pts) <= 4:
                    # 逐点 ax.plot(x, y, "o") 产生的是 1 点 Line2D，
                    # 上面三条规则（吞没率/比例/绝对数）对它全部失效，
                    # 而丢一个点往往就是丢一档结论
                    if n_in >= 1:
                        _hit(f"{name} 压住 {n_in}/{len(pts)} 个数据点"
                             f"（小样本系列，丢一个点就是丢一档结论）")
                elif len(pts) >= 50:
                    # 散点云/密集采样：绝对点数没有意义，看比例
                    if ratio >= CLOUD_RATIO_MAX:
                        _hit(f"{name} 覆盖 {ratio:.0%} 的数据点，移动位置")
                elif n_in >= PATH_PTS_MAX:
                    _hit(f"{name} 覆盖数据（{n_in} 个采样点），移动位置")

        # 5d. 像素级兜底：把注释层全隐藏后重渲，直接数框底下的非白像素。
        #     不依赖 artist 类型枚举——满铺的 pcolormesh、rasterized 的场、
        #     三角面片、以后新加的画法，这条都算得对。
        if boxes:
            rects = [(n, b) for n, b, _, _, _ in boxes]
            hidden = [a for _, _, _, _, a in boxes]
            inks = _probe.ink_under(fig, rects, hidden)
            for (n, _), ink in zip(rects, inks):
                if ink > PIXEL_INK_ERR:
                    _hit(f"{n} 底下有 {ink:.0%} 的图面内容（像素实测），"
                         f"该位置不是真空区")
                elif ink > PIXEL_INK_WARN:
                    print(f"[QA WARN] {n} 底下有 {ink:.0%} 的图面内容，"
                          f"能挪则挪")

        # 5e. 图例压柱顶（柱顶是读数位置，盖住即失效）
        for name, bb, ax, is_leg, art in boxes:
            if ax is None or not is_leg:
                continue
            top_hit = 0
            for c in ax.containers:
                for pch in getattr(c, "patches", []):
                    pb = pch.get_window_extent(rd)
                    if bb.x0 < (pb.x0 + pb.x1) / 2 < bb.x1 and \
                            bb.y0 < pb.y1 < bb.y1:
                        top_hit += 1
            if top_hit:
                _hit(f"{name} 压住 {top_hit} 根柱的柱顶（读数位置），"
                     f"移出数据区或改线端直标")
    except Exception as e:  # 检查器自身故障不应伪装成通过
        print(f"[QA note] 遮挡检查未执行：{e}")

    # 6. 数据画在视窗外：某面板有数据但几乎全部落在轴限之外。
    #    marginal_grid 的 sharey 陷阱就是这么产生一张空白面板的。
    try:
        for ax in fig.get_axes():
            if ax.get_label() == "<colorbar>" or not ax.get_visible():
                continue
            ab = ax.get_window_extent()
            tot = vis = 0
            for _, pts in _probe.series_samples(ax):
                tot += len(pts)
                vis += int(((pts[:, 0] >= ab.x0) & (pts[:, 0] <= ab.x1) &
                            (pts[:, 1] >= ab.y0) &
                            (pts[:, 1] <= ab.y1)).sum())
            if tot >= 3 and vis / tot < 0.05:
                problems.append(
                    f"面板有 {tot} 个数据点但仅 {vis / tot:.0%} 落在视窗内"
                    f"（轴限或共享轴设错），该面板对读者是空白")
    except Exception as e:
        print(f"[QA note] 视窗检查未执行：{e}")

    # 7. 刻度标签重叠：叠在一起的刻度等于没有刻度
    try:
        rd = fig.canvas.get_renderer()
        for ax in fig.get_axes():
            for axis, nm in ((ax.xaxis, "x"), (ax.yaxis, "y")):
                # 次刻度也要查：对数轴的次刻度标签最容易叠成一团乱码，
                # 而只查主刻度永远发现不了
                labs = [t for t in (axis.get_ticklabels() +
                                    axis.get_ticklabels(minor=True))
                        if t.get_visible() and t.get_text().strip()]
                labs.sort(key=lambda t: (t.get_window_extent(rd).x0,
                                         t.get_window_extent(rd).y0))
                bbs = [t.get_window_extent(rd) for t in labs]
                stop = False
                for i in range(len(bbs) - 1):
                    inter = Bbox.intersection(bbs[i], bbs[i + 1])
                    if inter is None:
                        continue
                    a = min(bbs[i].width * bbs[i].height,
                            bbs[i + 1].width * bbs[i + 1].height)
                    if a > 0 and (inter.width * inter.height) / a > 0.2:
                        problems.append(
                            f"{nm} 轴刻度标签重叠"
                            f"（「{labs[i].get_text()[:8]}」与"
                            f"「{labs[i + 1].get_text()[:8]}」），"
                            f"减少刻度数或旋转标签")
                        stop = True
                        break
                if stop:
                    break
    except Exception as e:
        print(f"[QA note] 刻度检查未执行：{e}")

    # 8. 信息密度：墨迹覆盖率。参考期刊图约 25%–45%，
    #    实测一批"注释规范但信息稀薄"的图只有 5%–7%——
    #    半张 A4 宽的画布上只摆了三五个数字。这条比任何主观规则都好落地。
    inks: list[float] = []
    try:
        for ax in fig.get_axes():
            if ax.get_label() == "<colorbar>" or not ax.get_visible():
                continue
            if getattr(ax, "name", "") == "3d" or not ax.axison:
                continue          # 3D 投影与示意图画布的留白是构图本身
            ink = _probe.panel_ink(fig, ax)
            bb = ax.get_window_extent()
            area_cm2 = (bb.width / fig.dpi * 2.54) * (bb.height / fig.dpi * 2.54)
            # 按面积分级：真正的问题是"整栏宽的面板只装几个数"，
            # 而不是小尺寸对比图本身稀疏。单栏哑铃图 ~55cm² 只告警，
            # 双栏面板 ~90cm² 还只有 5% 墨迹才是硬伤。
            inks.append(ink)
            # 逐面板只拦"真空图"：parity、哑铃这类构图本就稀疏而正确，
            # 按 12% 卡会把它们全部误杀。信息密度是**构图选择**问题，
            # 该由下面的图级均值与图种构成比来管，不是逐面板打磨。
            if ink < 0.045 and area_cm2 >= 12:
                problems.append(
                    f"面板 {area_cm2:.0f} cm² 却只有 {ink:.1%} 墨迹，"
                    f"接近空白——并入相邻面板，或让数据进正文表格")
            elif ink < 0.12:
                print(f"[QA WARN] 面板墨迹 {ink:.1%} 偏低"
                      f"（参考期刊图 25%–45%）")
            # 反作弊：大片单一平色能把 ink 顶上去而信息为零
            if _probe.has_field(ax):
                flat = _probe.flat_color_ratio(fig, ax)
                if flat > 0.50:
                    print(f"[QA WARN] 场图 {flat:.0%} 的面积是同一种颜色，"
                          f"墨迹高但信息低——收窄色标范围，或叠等值线/散点结构层")
    except Exception as e:
        print(f"[QA note] 密度检查未执行：{e}")

    # 8b. 轴限利用率：数据只占轴长一小截 = 轴限被撑大（多半为腾注释位）
    try:
        for ax in fig.get_axes():
            if ax.get_label() == "<colorbar>" or not ax.axison:
                continue
            if getattr(ax, "name", "") == "3d":
                continue
            series = [p for _, p in _probe.series_samples(ax) if len(p)]
            if not series:
                continue
            pts = np.vstack(series)
            inv = ax.transData.inverted()
            data = inv.transform(pts)
            for k, (get_lim, get_scale, nm) in enumerate((
                    (ax.get_xlim, ax.get_xscale, "x"),
                    (ax.get_ylim, ax.get_yscale, "y"))):
                lo, hi = get_lim()
                d = data[:, k]
                d = d[np.isfinite(d)]
                if d.size < 2:
                    continue
                # 对数轴上"线性差值 ÷ 线性跨度"没有意义：实测 facet_metrics
                # 的 log 面板数据铺满整条轴，线性口径却只算出 13%。占比必须
                # 在该轴自己的标度空间里量。
                if get_scale() == "log" and min(lo, hi) > 0 and d.min() > 0:
                    lo, hi, d = np.log10(lo), np.log10(hi), np.log10(d)
                span = abs(hi - lo)
                if span <= 0 or not np.isfinite(span):
                    continue
                used = (d.max() - d.min()) / span
                # 场图豁免：热力/等值线的轴限由图像本身定，叠在上面的
                # 几个标记点占多少轴长没有意义（实测 annotated_heatmap
                # 的标记点只占 20%，那是正确的）。
                # 比例/概率轴豁免：check 13 明确要求把它锁到 [0,1]，
                # 锁完数据只占其中一段是正常的（命中率 0.42–0.58 占 16%
                # 轴长），8b 却会诊断成"单个离群点把轴撑爆"——两条检查
                # 互相打架，且诊断词与实情相反。
                _lab8 = ax.get_xlabel() if nm == "x" else ax.get_ylabel()
                _prop8 = (d.min() >= 0 and d.max() <= 1 and d.max() > 0.2
                          and re.search(
                              r"概率|比例|占比|份额|覆盖率|准确率|召回率|"
                              r"命中率|通过率|合格率|达标率|probab|proportion|"
                              r"fraction|share|rate|ratio|accuracy|recall",
                              _lab8 or "", re.I))
                if used < 0.35 and not _probe.has_field(ax) and not _prop8:
                    _hard(f"{nm} 轴数据只占 {used:.0%} 轴长——其余数据被压成"
                          f"一条线（多半是单个离群点把轴撑爆）。改对数轴或"
                          f"断轴，或把离群点单独拆一格", "axis_slack")
                elif used < 0.60:
                    print(f"[QA WARN] {nm} 轴数据只占 {used:.0%} 轴长——"
                          f"轴限被撑大（多半为腾注释位），收紧轴限，"
                          f"注释交给 stat_box 自动选位")
    except Exception as e:
        print(f"[QA note] 轴限检查未执行：{e}")

    # 9. 图级信息密度与图种构成——管的是构图选择，不是单面板打磨。
    #    参考期刊图：全图墨迹约 38%、一维面板占比约 27%；
    #    实测一批"注释规范但全是折线"的图：11.5% / 78%。
    #    没有这条，下一批图还会退化成折线。
    # 只对多面板图生效：多面板图必须让每一格都配得上它占的版面；
    # 单/双面板的聚焦图（一条收敛曲线、一张 parity）本就可以稀疏。
    _sm = bool(getattr(fig, "_ff_small_multiples", False))
    if not _sm and len(inks) >= 3 and float(np.mean(inks)) < 0.13:
        problems.append(
            f"{len(inks)} 面板图的平均墨迹仅 {np.mean(inks):.1%} < 13%——"
            f"整张图信息稀薄，合并面板、改二维场/联合分布/三维几何，"
            f"或让数据进正文表格")
    try:
        allax = [a for a in fig.get_axes()
                 if a.get_label() != "<colorbar>" and a.get_visible()
                 and (a.axison or getattr(a, "name", "") == "3d")]

        def _is_inset(a):
            """完全落在另一个坐标区内部的轴是放大窗，不是独立面板。"""
            pa = a.get_position()
            for o in allax:
                if o is a:
                    continue
                po = o.get_position()
                if (po.x0 <= pa.x0 and po.x1 >= pa.x1 and
                        po.y0 <= pa.y0 and po.y1 >= pa.y1 and
                        po.width * po.height > pa.width * pa.height * 1.5):
                    return True
            return False

        cand = [a for a in allax if not _is_inset(a)]
        # 边缘直方、色标这类附属轴不是"面板"：按面积剔除，
        # 否则一个 jointplot 会被算成"1 个联合分布 + 2 个一维图"
        if cand:
            areas = [a.get_window_extent().width *
                     a.get_window_extent().height for a in cand]
            med = float(np.median(areas))
            cand = [a for a, ar in zip(cand, areas) if ar >= 0.40 * med]
        kinds = [_probe.panel_archetype(a) for a in cand]
        if len(kinds) >= 3 and not _sm:
            n1d = kinds.count("1d")
            frac = n1d / len(kinds)
            if frac > 0.67:
                problems.append(
                    f"{n1d}/{len(kinds)} 面板是一维构图（{frac:.0%}）——"
                    f"参考期刊图该比例约 27%。至少把一个换成二维场/三维几何/"
                    f"联合分布（contour_field / percolation_geometry / "
                    f"joint_marginal / phase_transition）")
    except Exception as e:
        print(f"[QA note] 构成比检查未执行：{e}")

    # 9. 图题与图内注释对同一个量给出矛盾数值。
    #    SKILL.md 目测清单第 2 条（"图题中的每个数字与图内统计框/标注
    #    一致"）此前只有人眼在管。图题写 RMS 5.07、统计框写 6.20 是
    #    事实错误级缺陷：读者按图题引用，正文与图内自相矛盾。
    #    只比"标题 vs 其作用域内的注释"，不跨面板比——小倍数各面板的
    #    n/占比本就该不同，跨面板比必然误报。
    try:
        import re as _re
        _pair = _re.compile(
            r"([A-Za-z\u4e00-\u9fff][^=＝:：\n]{0,15}?)\s*[=＝:：]\s*"
            r"(-?\d+(?:\.\d+)?)")

        def _pairs(texts):
            out = []
            for t in texts:
                for lab, tok in _pair.findall(t or ""):
                    key = lab.strip().lower().replace(" ", "")
                    if key:
                        out.append((key, tok))
            return out

        def _same_number(t1, t2):
            """只是精度不同就算一致：5.1 与 5.07 是同一个量的两种写法。"""
            dec = min(len(t1.split(".")[1]) if "." in t1 else 0,
                      len(t2.split(".")[1]) if "." in t2 else 0)
            return round(float(t1), dec) == round(float(t2), dec)

        _ax_all = _all_axes(fig)
        _sup = fig._suptitle.get_text() if fig._suptitle is not None else ""
        _body = ([t.get_text() for a in _ax_all for t in a.texts] +
                 [t.get_text() for t in fig.texts if t is not fig._suptitle] +
                 [a.title.get_text() for a in _ax_all])
        _scopes = []
        if _sup.strip():
            _scopes.append(("图题", [_sup], _body))
        for a in _ax_all:
            if a.title.get_text().strip():
                _scopes.append(("面板标题", [a.title.get_text()],
                                [t.get_text() for t in a.texts]))
        _seen = set()
        for _name, _head, _rest in _scopes:
            hp, rp = _pairs(_head), _pairs(_rest)
            # 只比**能唯一配对**的量名：同一个名字在一侧出现多次，多半是
            # 同一统计量在多个条件下各报一次（实测 q6 的两个 KS 分属正确/
            # 错误抽样），谁对谁无从判定，比了必然误报。
            from collections import Counter
            ch, cr = Counter(k for k, _ in hp), Counter(k for k, _ in rp)
            for k, v in hp:
                if ch[k] != 1 or cr.get(k, 0) != 1:
                    continue
                for k2, v2 in rp:
                    if k2 != k or _same_number(v, v2):
                        continue
                    m = (f"{_name}与图内注释对「{k}」给出两个不同数值"
                         f"（{v} / {v2}）——图题数字必须与统计框同源，"
                         f"用同一个变量 f-string 引用")
                    if m not in _seen:
                        _seen.add(m)
                        _hard(m, "number_conflict")
    except Exception as e:
        print(f"[QA note] 数值一致性检查未执行：{e}")

    # 10. 跨面板重复信息：同一条数据系列在两个面板各画一遍。
    #     Nature 明文"不重复信息"，SKILL.md 目测清单也列了，此前无检查。
    #     只查"有形状的曲线"：≥8 个采样点且非常量——水平基线/参考线在
    #     多个面板重复出现是正当的（那是刻度，不是数据）。放大窗豁免：
    #     inset 本来就是把同一条曲线放大再画一遍。
    try:
        def _inside_other(a, others):
            pa = a.get_position()
            for o in others:
                if o is a:
                    continue
                po = o.get_position()
                if (po.x0 <= pa.x0 and po.x1 >= pa.x1
                        and po.y0 <= pa.y0 and po.y1 >= pa.y1):
                    return True
            return False

        _all_ax = _all_axes(fig)
        _cand = [a for a in _all_ax if not _inside_other(a, _all_ax)]
        _sigs: dict[bytes, list[int]] = {}
        for a in _cand:
            for ln in a.lines:
                if not ln.get_visible():
                    continue
                xy = np.asarray(ln.get_xydata(), dtype=float)
                if len(xy) < 8 or not np.all(np.isfinite(xy)):
                    continue
                if np.ptp(xy[:, 1]) == 0:
                    continue          # 水平基线，重复出现是正当的
                if not _is_colored(ln.get_color()):
                    continue          # 中性色 = 参考几何/基线：各面板重复
                                      # 画同一个口径圆、同一条参考轮廓是
                                      # 正当构图（那是刻度，不是数据）
                _sigs.setdefault(xy.round(9).tobytes(), []).append(id(a))
        for _owners in _sigs.values():
            if len(set(_owners)) > 1:
                _hard("两个面板重复画了同一条数据系列——期刊图不重复信息，"
                      "删掉其一，或把第二处改画差值/残差以增加信息",
                      "duplicate_series")
                break
    except Exception as e:
        print(f"[QA note] 重复系列检查未执行：{e}")

    # 11. 小倍数各面板色标范围不一致。小倍数的立身之本是"同一编码施于
    #     不同切片"，clim 不同则面板间根本不可比，读者会把深浅当成量的
    #     差异——这是小倍数最致命的错。只对 small_multiples() 建的图查；
    #     不同量的多面板用不同 clim 是正当的，不在此列。
    if getattr(fig, "_ff_small_multiples", False):
        try:
            _clims: dict[str, set] = {}
            for a in _all_axes(fig):
                for sm in list(a.images) + list(a.collections):
                    if not hasattr(sm, "get_clim") or sm.get_array() is None:
                        continue
                    lo, hi = sm.get_clim()
                    if lo is None or hi is None:
                        continue
                    name = getattr(sm.get_cmap(), "name", "?")
                    _clims.setdefault(name, set()).add(
                        (round(float(lo), 9), round(float(hi), 9)))
            for _cm, _vals in _clims.items():
                if len(_vals) > 1:
                    _hard(f"小倍数各面板色标范围不一致（{_cm}: "
                          f"{sorted(_vals)}）——同一编码施于不同切片才可比，"
                          f"用 share_colorbar 并锁定 vmin/vmax",
                          "clim_mismatch")
                    break
        except Exception as e:
            print(f"[QA note] 小倍数色标检查未执行：{e}")

    # 11a. 文字对比度。语义色直接拿来写字在白底上往往不够：Okabe-Ito 的橙
    #      #E69F00 只有 2.25:1，屏幕上已经发虚、300dpi 印刷后更糟。
    #      core.annotate 的着色点已在 _text_color 里统一压暗，但 recipe 里
    #      裸 `ax.annotate(color=PALETTE[...])` 绕开了它——没有这条检查，
    #      这类缺陷结构上不可能被自动发现。WCAG 非文字/大字号下限 3:1。
    #
    #      **背景必须实测，不能靠几何猜**：早先版本用"文字中心落进任意实心
    #      图元的 window_extent"当"深底豁免"，两个方向都错——contourf 与
    #      Poly3D 的 window_extent 返回 Bbox(inf,...)，场图白字被硬拒；而
    #      白字落进任何**浅色**段都被无条件豁免，实测漏掉了 gallery 自己的
    #      7 处 1.64:1 缺陷。这里改为把全部文字藏起来渲染一次，直接采样每
    #      段文字底下的真实像素中位亮度。代价是每次 run_qa 多一次 draw。
    try:
        from .colors import luminance
        _texts = [t for t in _all_texts(fig)
                  if (t.get_text() or "").strip() and t.get_visible()]
        _bg = None
        if _texts:
            _vis0 = [t.get_visible() for t in _texts]
            for t in _texts:
                t.set_visible(False)
            try:
                fig.canvas.draw()
                _bg = np.asarray(fig.canvas.buffer_rgba(), dtype=float) / 255.0
            finally:
                for t, v in zip(_texts, _vis0):
                    t.set_visible(v)
                fig.canvas.draw()
        _rd3 = fig.canvas.get_renderer()
        _H = None if _bg is None else _bg.shape[0]
        _dim = []
        for t in _texts:
            # 白色描边光晕就是"文字压在场图上"的正解（annotate._halo 专为
            # 此写，比不透明白底方框更好——它不会在图上打一个洞）。文字
            # 有自己的浅色描边时，读者看到的局部背景就是描边而非底图。
            _pe = t.get_path_effects()
            if _pe and any(
                    luminance(mcolors.to_rgb(
                        getattr(e, "_gc", {}).get("foreground", "white")))
                    > 0.6 for e in _pe):
                continue
            col = mcolors.to_rgb(t.get_color())
            al = t.get_alpha()
            # 半透明文字先与其背景合成后再比
            bb = t.get_window_extent(_rd3)
            bg_rgb = (1.0, 1.0, 1.0)
            if _bg is not None and np.all(np.isfinite(
                    [bb.x0, bb.x1, bb.y0, bb.y1])):
                x0 = max(0, int(bb.x0)); x1 = min(_bg.shape[1], int(bb.x1) + 1)
                y0 = max(0, int(_H - bb.y1)); y1 = min(_H, int(_H - bb.y0) + 1)
                if x1 > x0 and y1 > y0:
                    patch = _bg[y0:y1, x0:x1, :3].reshape(-1, 3)
                    bg_rgb = tuple(np.median(patch, axis=0))
            # 文字自带底框时，**框才是背景**：藏文字会连框一起藏掉，采到的
            # 是框底下的东西。callout 在浅色场上正是用半透明白框（而非描边）
            # 保证可读，不算进来会把它误判成低对比度。
            _bp = t.get_bbox_patch()
            if _bp is not None:
                _fc = _bp.get_facecolor()
                if len(_fc) == 4 and _fc[3] > 0.15:
                    bg_rgb = tuple(_fc[3] * c + (1 - _fc[3]) * b
                                   for c, b in zip(_fc[:3], bg_rgb))
            if al is not None and al < 1.0:
                col = tuple(al * c + (1 - al) * b
                            for c, b in zip(col, bg_rgb))
            Lt, Lb = luminance(col), luminance(bg_rgb)
            ratio = (max(Lt, Lb) + 0.05) / (min(Lt, Lb) + 0.05)
            if ratio < 3.0:
                _dim.append((t.get_text().strip()[:12],
                             mcolors.to_hex(col), ratio))
        if _dim:
            _hard(f"{len(_dim)} 处文字与其实际背景对比度 < 3:1（印刷后发虚）："
                  f"{[(a, b, round(c, 2)) for a, b, c in _dim[:3]]}"
                  f"——语义色直接写字往往不够暗，走 annotate 的直标函数"
                  f"（会自动压暗）或 core.ink()；深底上的白字同理要够浅",
                  "text_contrast")
    except Exception as e:
        print(f"[QA note] 文字对比度检查未执行：{e}")

    # 11b. 同一根轴上叠不可通约的量。SPEC §2.2 把这条列为硬伤，但此前只
    #      查了 twinx——直接在同一根 y 轴上画"比例 0–1"和"成本 1200 元"
    #      一直漏网：小的那条被压成一条贴轴线，读者读不出任何变化。
    #      判据保守：线性轴、两条彩色系列的取值区间**完全不相交**、且量级
    #      差 >100×。对数轴豁免（跨数量级正是用 log 轴的理由）。
    try:
        for a in _all_axes(fig):
            if getattr(a, "name", "") == "3d" or not a.axison:
                continue
            if a.get_yscale() != "linear" or _probe.has_field(a):
                continue
            rng_ = []
            for ln in a.lines:
                if not ln.get_visible() or not _is_colored(ln.get_color()):
                    continue
                yv = np.asarray(ln.get_xydata(), dtype=float)[:, 1]
                yv = yv[np.isfinite(yv)]
                if yv.size < 3 or np.ptp(yv) == 0:
                    continue
                rng_.append((float(yv.min()), float(yv.max())))
            hit_pair = None
            for i2 in range(len(rng_)):
                for j2 in range(i2 + 1, len(rng_)):
                    lo1, hi1 = rng_[i2]
                    lo2, hi2 = rng_[j2]
                    if hi1 >= lo2 and hi2 >= lo1:
                        continue                      # 区间相交 = 同量纲
                    m1 = max(abs(lo1), abs(hi1))
                    m2 = max(abs(lo2), abs(hi2))
                    lo_m, hi_m = min(m1, m2), max(m1, m2)
                    if lo_m > 0 and hi_m / lo_m > 100:
                        hit_pair = (lo_m, hi_m)
                        break
                if hit_pair:
                    break
            if hit_pair:
                _hard(f"同一根线性 y 轴上叠了量级 {hit_pair[0]:.3g} 与 "
                      f"{hit_pair[1]:.3g}（相差 {hit_pair[1]/hit_pair[0]:.0f}×）"
                      f"且取值区间不相交的两条系列——不可通约或跨量级，"
                      f"小的那条被压成贴轴线。拆面板（见 phase_transition.py）"
                      f"；同单位跨量级改对数轴即可",
                      "incommensurable")
    except Exception as e:
        print(f"[QA note] 同轴量纲检查未执行：{e}")

    # 12. 少量离散点用折线连起来。折线宣称"点之间可以插值"，而报数档 /
    #     方案 / 策略这类离散量之间没有中间态，语义就错了。实测 q2_1
    #     面板 (a)：4 个档位连成一条线，占满整个面板还看不见置信区间。
    #     斜率图（每条线恰好 2 点）与稠密曲线都不在此列；轴上只要有更密
    #     的系列，说明稀疏线只是标记层，也放行。
    try:
        _sparse_hit = False
        for a in _all_axes(fig):
            if _sparse_hit:
                break
            _vis = [ln for ln in a.lines if ln.get_visible()]
            _npts = [len(np.asarray(ln.get_xydata())) for ln in _vis]
            if not _npts or max(_npts) > 6:
                continue
            _bundle = [ln for ln in _vis
                       if str(ln.get_linestyle()).lower().strip()
                       not in ("none", "")
                       and 3 <= len(np.asarray(ln.get_xydata())) <= 6]
            # 平行坐标是一整束这样的连线，连线正是它的构图本体；病灶
            # （q2_1 那种）是孤零零一两条。整束放行——**数上灰色背景线**：
            # 平行坐标常只强调 1–2 条，其余压成 #CCCCCC，只数彩色的会把
            # 它误判成病灶。
            if len(_bundle) > 2:
                continue
            _cand = [ln for ln in _bundle if _is_colored(ln.get_color())]
            # 已经有误差棒的轴放行：报错信息建议的替代就是"每档一行 +
            # 误差棒"，图里已经有了就别再拦。
            from matplotlib.container import ErrorbarContainer
            if any(isinstance(c, ErrorbarContainer) for c in a.containers):
                continue

            # x 是不是"类别轴"。**不要去解析渲染后的刻度文本**：log 轴是
            # mathtext `$\mathdefault{10^{0}}$`、日期轴是 "2026-01-05"、
            # 千分位是 "1,000"，它们 float() 全都解析不了，会被一律误判成
            # 类别轴，于是连续量扫描的标准画法反而挨拦。直接问 matplotlib
            # 这根轴是什么类型才可靠。
            from matplotlib.category import StrCategoryConverter
            from matplotlib.ticker import FixedFormatter
            from matplotlib.scale import LinearScale
            _xa = a.xaxis
            # `Axis.converter` 在 3.10 弃用、3.12 移除；`_scale` 是私有。
            # 用公开 API，否则升级后整条检查会被 except 静默吞掉。
            _conv = (_xa.get_converter() if hasattr(_xa, "get_converter")
                     else getattr(_xa, "converter", None))
            def _non_numeric(t):
                q = t.strip().replace("−", "-").rstrip("%").strip()
                if not q:
                    return False
                try:
                    float(q.replace(",", ""))
                    return False
                except ValueError:
                    return True

            # `set_xticklabels` 在 mpl 3.10 装的是 FuncFormatter 而非
            # FixedFormatter，只认后者等于没认。刻度**文本**非数值是可靠
            # 的补充信号——但只在**线性轴且无单位转换**时才看它，否则
            # log 的 mathtext、日期串会被误当类别（这正是上一轮的错法）。
            # 必须与 **FixedLocator** 合取：只有 `set_xticks()` 显式定位
            # 才装 FixedLocator，而 EngFormatter/千分位/带单位后缀这些常见
            # formatter 走的是 AutoLocator——不加这个合取，等距连续扫描会
            # 被一律误判成类别轴而硬拒（注释自己写着"不要解析刻度文本"）。
            from matplotlib.ticker import FixedLocator
            _plain = (a.get_xscale() == "linear" and _conv is None
                      and isinstance(_xa.get_major_locator(), FixedLocator))
            _cat_axis = (isinstance(_conv, StrCategoryConverter)
                         or isinstance(_xa.get_major_formatter(),
                                       FixedFormatter)
                         or (_plain and any(
                             _non_numeric(t.get_text())
                             for t in a.get_xticklabels())))
            # 日期/对数等非线性或有单位转换的轴 = 连续量，整轴放行
            _continuous_axis = (a.get_xscale() != "linear"
                                or (_conv is not None and not _cat_axis))
            if _continuous_axis:
                continue
            for ln in _cand:
                xy = np.asarray(ln.get_xydata(), dtype=float)
                n = len(xy)
                if not _cat_axis:
                    xs = xy[:, 0]
                    dx = np.abs(np.diff(xs))
                    dx = dx[np.isfinite(dx) & (dx > 0)]
                    # 等差 = 连续量扫描（D0 每 0.1 一档）；等比 = 几何扫描
                    # （网格加密 1,2,4,8,16 / 样本量倍增），两者插值都合法。
                    # 但"恰好落在 0..n-1 小整数上"是位置编码，不是采样：
                    # 连续量不会只在 0,1,2,3 上取 3–6 个样本。
                    arith = bool(dx.size) and dx.max() / dx.min() < 1.05
                    # 必须是 **0..n-1 连续整数**才算位置编码。"任何 <12
                    # 的整数"会把 k-means 肘部 k=2..6、迭代次数 1..5、
                    # 多项式阶数 1..4 这类小整数**连续量**全部硬挡——
                    # 它们插值完全合法，而 sparse_line 是硬错。
                    ints = np.allclose(xs, np.arange(len(xs)))
                    geo = False
                    if np.all(xs > 0) and len(xs) > 1:
                        r = xs[1:] / xs[:-1]
                        r = r[np.isfinite(r) & (r > 0)]
                        geo = bool(r.size) and r.max() / r.min() < 1.05
                    if (arith or geo) and not ints:
                        continue
                _hard(f"{n} 个点用折线连起来——折线宣称点之间可插值，"
                      f"而离散档位/方案/策略没有中间态。改点区间图"
                      f"（每档一行 + 误差棒 + 判据竖线）、有序条+直标"
                      f"或哑铃图", "sparse_line")
                _sparse_hit = True
                break
    except Exception as e:
        print(f"[QA note] 稀疏折线检查未执行：{e}")

    # 13. 比例/概率轴超出 [0,1]。负概率没有意义，>1 的比例也没有；实测
    #     q2_1 面板 (a) 把 y 轴放到 −0.25～1.12，上下白留两成，本就只有
    #     4 个点，于是更显孤零。判据只认"数据确实在 0–1 尺度上"（全部
    #     落在 [0,1] 且最大值 > 0.5），并留出 matplotlib 默认 5% 边距。
    try:
        import re as _re2
        _re_prop = _re2.compile(
            r"概率|比例|占比|份额|覆盖率|准确率|召回率|命中率|通过率|"
            r"合格率|达标率|(^|[^A-Za-z])(P|p)($|[^A-Za-z])|"
            r"probab|proportion|fraction|share|rate|ratio|accuracy|recall|"
            r"precision", _re2.I)
        for a in _all_axes(fig):
            if getattr(a, "name", "") == "3d" or not a.axison:
                continue
            # 必须用**原始顶点**，不能用 _probe.series_samples——后者会
            # 沿线 densify 插值，取值个数被插到几百，"位置类别轴只有两
            # 个取值"这个判据就失效了。
            _chunks = []
            for _ln in a.lines:
                if _ln.get_visible():
                    _xy = np.asarray(_ln.get_xydata(), dtype=float)
                    if _xy.size:
                        _chunks.append(_xy.reshape(-1, 2))
            for _c in a.collections:
                if (_c.__class__.__name__ in _probe.FIELD_CLASSES
                        or not _c.get_visible()):
                    continue
                _off = np.asarray(getattr(_c, "get_offsets", lambda: [])(),
                                  dtype=float)
                if _off.size:
                    _chunks.append(_off.reshape(-1, 2))
            if not _chunks:
                continue
            _dat = np.vstack(_chunks)
            for _k, (_getlim, _nm) in enumerate(((a.get_xlim, "x"),
                                                 (a.get_ylim, "y"))):
                # 必须轴标题里确实说了这是比例/概率：体积分数 φ = 0.5–1.0
                # **百分数**恰好落在 [0,1]，归一化坐标、物理量同理，它们
                # 都不是比例（实测 q2 的相变密度场 x 轴就被误判过）。
                _lab = (a.get_xlabel() if _nm == "x" else a.get_ylabel())
                if not _re_prop.search(_lab or ""):
                    continue
                # 无可读刻度的归一化位置轴（平行坐标的 y 就 set_yticks([])）
                # 谈不上"超出 [0,1]"：读者看不到刻度，也就无从误读，轴限
                # 留白是给两端的数值标注让位。
                _tl = (a.get_xticklabels() if _nm == "x"
                       else a.get_yticklabels())
                if not any(t.get_text().strip() for t in _tl):
                    continue
                d = _dat[:, _k]
                d = d[np.isfinite(d)]
                # 真比例轴上会散布多个不同取值；斜率图/哑铃图的 x 只有
                # 0 和 1 两个位置类别，轴限放宽是为两端直标留位，不是病。
                if (d.size < 3 or np.unique(np.round(d, 6)).size < 4
                        or d.min() < 0 or d.max() > 1 or d.max() <= 0.5):
                    continue
                lo, hi = _getlim()
                if lo < -0.10 or hi > 1.10:
                    _hard(f"{_nm} 是比例/概率轴（数据全落在 0–1）却把轴限"
                          f"放到 {lo:.2f}–{hi:.2f}：比例/概率轴超出 [0,1] "
                          f"没有意义，锁 set_{_nm}lim(0, 1)",
                          "unit_axis_range")
    except Exception as e:
        print(f"[QA note] 比例轴检查未执行：{e}")

    if _waived:
        fig._ff_qa_waived = list(_waived)

    if problems and strict:
        raise AssertionError("QA FAILED:\n- " + "\n- ".join(problems))
    if problems:
        for p in problems:
            print(f"[QA WARN] {p}")
    else:
        print("[QA] PASS (auto checks)")
    if not problems:
        from .style import mark_qa_passed
        mark_qa_passed(fig)          # save_figure 据此放行，坏图不落盘
    return problems
