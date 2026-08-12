"""全局样式：期刊栏宽、CJK 字体解析、印刷级 rcParams。

用法::

    from core import apply_style, new_figure, save_figure
    apply_style()                      # 默认 1.5 栏
    fig, ax = new_figure(width="single", ratio=0.75)
    ...
    save_figure(fig, "gallery/示例")   # 自动导出 png + svg
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

MM = 1 / 25.4  # mm -> inch

# Nature 系栏宽；"cn" 为中文数模 A4 正文常用图宽
COLUMN_WIDTHS = {"single": 89, "onehalf": 136, "double": 183, "cn": 150}

# 中文数模论文优先衬线（与正文宋体系一致），逐级回退
_CJK_CANDIDATES = [
    "Source Han Serif SC", "Source Han Serif CN", "Noto Serif CJK SC",
    "SimSun", "STSong", "SimHei", "Microsoft YaHei",
]
_LATIN_CANDIDATES = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]


def _resolve_fonts() -> list[str]:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in _LATIN_CANDIDATES if f in installed]
    fonts += [f for f in _CJK_CANDIDATES if f in installed]
    return fonts or ["DejaVu Sans"]


def apply_style(base_size: float = 9.0) -> None:
    """印刷档字号（pt）：正文 9，刻度 8，注释 7。dpi 只影响预览。"""
    fonts = _resolve_fonts()
    mpl.rcParams.update({
        # 直接给列表才能触发逐字符回退（拉丁用衬线、中文用 CJK 字体）
        "font.family": fonts,
        "font.serif": fonts,
        "axes.unicode_minus": False,
        "font.size": base_size,
        "axes.titlesize": base_size + 0.5,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linewidth": 0.4,
        "grid.alpha": 0.35, "grid.linestyle": "--",
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8", "legend.fancybox": False,
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.axisbelow": True,
        "mathtext.fontset": "stix",
    })


def new_figure(width: str | float = "onehalf", ratio: float = 0.62,
               **kwargs):
    """按栏宽建图。width 取 COLUMN_WIDTHS 键名或 mm 数值；ratio=高/宽。"""
    w_mm = COLUMN_WIDTHS.get(width, width) if isinstance(width, str) else width
    figsize = (w_mm * MM, w_mm * MM * ratio)
    return plt.subplots(figsize=figsize, **kwargs)


def save_figure(fig, path_no_ext: str, formats=("png", "svg")) -> list[str]:
    """导出 png(300dpi)+svg(文字可编辑)。返回输出文件列表。"""
    out = []
    for ext in formats:
        p = f"{path_no_ext}.{ext}"
        fig.savefig(p)
        out.append(p)
    return out
