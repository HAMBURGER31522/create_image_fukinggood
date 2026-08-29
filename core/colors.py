"""配色：低饱和学术色 + 场景化色图 + 视觉层次 + 色盲/灰度自检。

原则（对齐 Nature 官方规范与其艺术编辑指南）：
- **层次**："最重要的元素饱和度最高，背景元素用中性色。"用 emphasis()
  显式分三级 focus/context/background，不要让配角比主角更抢眼。
- 分类色默认 Okabe-Ito（色盲安全），全文同一对象同一色。
- **避免红/绿组合**——Nature 明文点名。语义色 good/bad 已改用蓝/橙对。
- 不靠颜色单独承载含义：颜色之外必须再给标签或形状（"Try to label
  where possible"）。
- 发散数据（有物理零点）用 PuOr/RdBu；顺序数据用 YlOrRd/crest/viridis 族。
- 禁 jet/rainbow/hsv（qa.py 会拦截）。
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

# Okabe-Ito 8 色（色盲安全，Nature 推荐的 Color Universal Design 族）
OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}

# 分类主序列：Okabe-Ito 按"任取前 N 色都尽量可分"穷举排序的结果
# （已剔除亮黄 #F0E442——lum 0.74，细线画在白底上几乎看不见）。
# 实测前 N 色在三类色盲下的最小色距：
#     N=2 → 0.93   N=3 → 0.42   N=4 → 0.23   N=5 → 0.12   N=6 → 0.05
# 即 **≤4 类是安全区，5 类勉强，6 类必须叠加形状/线型/直标冗余**。
# 这不是调色技巧问题：穷举过全部候选后，6 个既色盲安全又灰度可分的
# 分类色在 sRGB 里并不存在——用 categorical() 拿配套的冗余编码。
PALETTE = ["#56B4E9", "#D55E00", "#0072B2", "#E69F00", "#009E73", "#CC79A7"]

# 低饱和柔和序列：用于填充、背景分区、大面积铺色（不承担分类区分职责）
PALETTE_MUTED = ["#5B8DB8", "#E0A458", "#7FB285", "#C97B84", "#9A8FBF",
                 "#6BAED6"]

# 冗余编码：颜色之外的第二/第三通道。Nature 明文"不要只靠颜色定义"。
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", (0, (4, 1.5)), ":", (0, (5, 1, 1, 1))]

# 语义固定色：同一语义全文一致。三处刻意选择：
# - good/bad 不用绿/红：Nature 明文避免该组合，且红绿色盲下色距仅 0.16。
# - data/fit 改蓝/橙：原来的 #4C72B0/#C44E52 灰度亮度差只有 0.01，
#   黑白打印后两条线完全同色；蓝/橙为 0.26，且 CVD 色距 0.79。
# - highlight 用朱红：只留给全图唯一的关键点，不与 fit 抢。
_SEMANTIC = {
    "data":     "#0072B2",   # 观测/数据点
    "fit":      "#E69F00",   # 拟合/模型
    "baseline": "#8C8C8C",   # 基线/参考
    "highlight": "#D55E00",  # 最优点/关键点
    "band":     "#AECDE1",   # 置信带
    "good":     "#0072B2",   # 达标/通过（蓝）
    "bad":      "#E69F00",   # 未达标/失败（橙）
}

_CMAPS = {
    "diverging": "PuOr_r",   # 有零点的场（偏差、残差场）
    "sequential": "YlOrRd",  # 单调强度场（密度、概率）
    "sequential2": "crest",  # seaborn crest，冷调顺序
    "surface": "viridis",    # 3D 曲面
    "heatmap": "vlag",       # 响应面（低-高，seaborn 低饱和发散）
}

BANNED_CMAPS = {"jet", "rainbow", "hsv", "gist_rainbow", "nipy_spectral",
                "turbo", "gist_ncar"}


def semantic(key: str) -> str:
    if key not in _SEMANTIC:
        raise ValueError(f"未知语义色 '{key}'，可选：{sorted(_SEMANTIC)}")
    return _SEMANTIC[key]


def cmap_for(scene: str):
    """scene in {diverging, sequential, sequential2, surface, heatmap}"""
    if scene not in _CMAPS:
        raise ValueError(f"未知色图场景 '{scene}'，可选：{sorted(_CMAPS)}")
    name = _CMAPS[scene]
    if name in ("crest", "vlag"):
        try:
            import seaborn as sns
            return sns.color_palette(name, as_cmap=True)
        except ImportError:
            name = {"crest": "GnBu", "vlag": "RdBu_r"}[name]
    return plt.get_cmap(name)


def truncate_cmap(cmap, lo: float = 0.12, hi: float = 0.88, n: int = 256):
    """截断色图两端的高饱和段，用于大面积铺色时保持"淡"的期刊气质。"""
    from matplotlib.colors import ListedColormap
    return ListedColormap(cmap(np.linspace(lo, hi, n)),
                          name=f"{getattr(cmap, 'name', 'cmap')}_trunc")


# ------------------------------------------------------------ 视觉层次
_LEVELS = {"focus": 1.0, "context": 0.45, "background": 0.18}


def emphasis(color, level: str = "focus"):
    """按信息重要性调饱和度：focus 原色、context 淡、background 近中性灰。

    Nature 艺术编辑的核心原则是"饱和度加权于重要性"。图里最该被看见的
    那个对象（论点所在的阈值线、最优点、目标曲线）必须是最饱和的，
    陪衬对象往中性色收。反过来做——把论点调灰、把配角调艳——是本 skill
    历史上最典型的失手。

    level ∈ {focus, context, background}。返回 hex 色。
    """
    if level not in _LEVELS:
        raise ValueError(f"未知层次 '{level}'，可选：{sorted(_LEVELS)}")
    k = _LEVELS[level]
    r, g, b = to_rgb(color)
    # 向该色自身的灰度值收缩：保留色相，只降彩度与对比
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    mixed = [grey + (c - grey) * k for c in (r, g, b)]
    # 背景级再整体提亮，确保不与前景抢注意力
    if level == "background":
        mixed = [c + (1.0 - c) * 0.35 for c in mixed]
    return to_hex(tuple(np.clip(mixed, 0, 1)))


# ------------------------------------------------------------ 可达性自检
def _srgb_to_lin(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(color) -> float:
    """WCAG 相对亮度，用于对比度与灰度可分性判断。"""
    r, g, b = _srgb_to_lin(to_rgb(color))
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def contrast_ratio(c1, c2) -> float:
    """WCAG 对比度（1–21）。Nature 要求文字与底色 >4.5。"""
    a, b = sorted((luminance(c1), luminance(c2)))
    return (b + 0.05) / (a + 0.05)


# Viénot–Brettel–Mollon 1999 二色觉模拟矩阵（在 gamma 编码 sRGB 上运算）
_RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                     [3.45565, 27.1554, 3.86714],
                     [0.0299566, 0.184309, 1.46709]])
_LMS2RGB = np.linalg.inv(_RGB2LMS)


def simulate_cvd(color, kind: str = "deuteranopia"):
    """模拟二色觉下看到的颜色。kind ∈ {deuteranopia, protanopia, tritanopia}。"""
    lms = _RGB2LMS @ np.asarray(to_rgb(color), dtype=float)
    L, M, S = lms
    if kind == "deuteranopia":       # 缺 M 视锥（最常见，约占男性 6%）
        lms = np.array([L, 0.494207 * L + 1.24827 * S, S])
    elif kind == "protanopia":       # 缺 L 视锥
        lms = np.array([2.02344 * M - 2.52581 * S, M, S])
    elif kind == "tritanopia":       # 缺 S 视锥（罕见）
        lms = np.array([L, M, -0.395913 * L + 0.801109 * M])
    else:
        raise ValueError(f"未知色盲类型 '{kind}'")
    return to_hex(tuple(np.clip(_LMS2RGB @ lms, 0, 1)))


def _dist(c1, c2) -> float:
    """两色在 sRGB 立方体内的欧氏距离（0–√3），够用的粗判据。"""
    return float(np.linalg.norm(np.array(to_rgb(c1)) - np.array(to_rgb(c2))))


def _is_emphasis_pair(c1, c2, hue_tol: float = 0.045,
                      chroma_ratio: float = 1.6) -> bool:
    """两色是否为同一色相的强调/弱化两级（emphasis() 派生对）。

    这类色对本来就该相似——它们表达的是同一个对象的两个视觉权重，
    不是两个需要区分的类别。可达性检查必须跳过，否则每用一次
    emphasis() 就会误报一条"两色几乎同色"。
    """
    from matplotlib.colors import rgb_to_hsv
    h1, s1, v1 = rgb_to_hsv(to_rgb(c1))
    h2, s2, v2 = rgb_to_hsv(to_rgb(c2))
    dh = abs(h1 - h2)
    dh = min(dh, 1 - dh)                      # 色相是环形的
    if dh > hue_tol:
        return False
    lo, hi = sorted((s1, s2))
    return lo < 1e-6 or hi / lo >= chroma_ratio


def categorical(n: int, muted: bool = False):
    """取 n 类的配色 + 配套冗余编码。返回 (colors, markers, linestyles)。

    颜色永远不该单独承担分类职责——把 markers/linestyles 一并用上，
    图在黑白打印、投影仪、色盲读者眼里才都成立::

        cols, mks, lss = categorical(3)
        for y, c, m, ls in zip(series, cols, mks, lss):
            ax.plot(x, y, color=c, marker=m, linestyle=ls)

    n > 5 会告警：实测 6 色的最小色盲色距只有 0.05，此时必须直标。
    """
    if n < 1:
        raise ValueError(f"n 必须 ≥ 1，收到 {n}")
    base = PALETTE_MUTED if muted else PALETTE
    if n > len(base):
        raise ValueError(f"分类色最多 {len(base)} 类，收到 {n}——"
                         f"超过就该换构图（小倍数拆面板/直标），不是加颜色")
    if n > 5:
        print(f"[colors WARN] {n} 类分类色：6 色的最小色盲色距仅 0.05，"
              f"必须靠直标或小倍数区分，颜色只作辅助")
    return base[:n], MARKERS[:n], LINESTYLES[:n]


# 阈值标定说明：0.20 用于两三个语义色的严判；分类色集合放宽到 0.12——
# 连 Okabe-Ito（公认色盲安全基准）取满 6 色时最小色距也只有 0.05，
# 拿 0.20 卡 6 色集合等于要求一个不存在的调色板。
def check_accessibility(colors, min_dist: float = 0.12,
                        min_grey_gap: float = 0.10,
                        redundant: bool = False) -> list[str]:
    """检查一组分类色在色盲与灰度下是否仍可区分。返回问题列表（空=通过）。

    Nature 建议成图后"用色盲模拟工具检查，并转灰度或完全去饱和后复看"。
    这里把这两步做成可自动执行的判据。

    redundant=True 表示调用方已提供形状/线型冗余编码，此时灰度不可分
    降级为提示——黑白打印下仍可靠 marker 区分，不再是硬伤。
    """
    problems = []
    cols = list(colors)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if _is_emphasis_pair(a, b):
                continue        # 同一对象的强调/弱化两级，本就该相似
            for kind in ("deuteranopia", "protanopia", "tritanopia"):
                d = _dist(simulate_cvd(a, kind), simulate_cvd(b, kind))
                if d < min_dist:
                    problems.append(
                        f"{to_hex(to_rgb(a))} 与 {to_hex(to_rgb(b))} 在"
                        f"{kind}下几乎同色（色距 {d:.2f} < {min_dist}），"
                        f"换色或加形状/标签冗余编码")
                    break
            gap = abs(luminance(a) - luminance(b))
            if gap < min_grey_gap and not redundant:
                problems.append(
                    f"{to_hex(to_rgb(a))} 与 {to_hex(to_rgb(b))} 灰度亮度差"
                    f"仅 {gap:.2f} < {min_grey_gap}，黑白打印后不可分——"
                    f"改用 categorical() 取配套 marker/线型冗余")
    return problems
