"""配色：低饱和学术色 + 场景化色图选择。

原则：
- 分类色默认 Okabe-Ito（色盲安全），全文同一对象同一色。
- 发散数据（有物理零点）用 PuOr/RdBu；顺序数据用 YlOrRd/crest/viridis 族。
- 禁 jet/rainbow/hsv（qa.py 会拦截）。
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# Okabe-Ito 8 色（色盲安全）
OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}

# 主用低饱和序列（参考图同款气质：柔和、偏灰调）
PALETTE = ["#5B8DB8", "#E0A458", "#7FB285", "#C97B84", "#9A8FBF", "#6BAED6"]

# 语义固定色：同一语义全文一致
_SEMANTIC = {
    "data":     "#4C72B0",   # 观测/数据点
    "fit":      "#C44E52",   # 拟合/模型
    "baseline": "#8C8C8C",   # 基线/参考
    "highlight": "#D55E00",  # 最优点/关键点
    "band":     "#AECDE1",   # 置信带
    "good":     "#009E73",
    "bad":      "#C44E52",
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
    import numpy as np
    from matplotlib.colors import ListedColormap
    return ListedColormap(cmap(np.linspace(lo, hi, n)),
                          name=f"{getattr(cmap, 'name', 'cmap')}_trunc")
