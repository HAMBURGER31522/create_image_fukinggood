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


_DRAFT_MODE = False
_STYLE_APPLIED = False


def is_styled() -> bool:
    """当前 rcParams 是否处于本 skill 样式态（run_qa 用）。

    不只看调用标志：rcdefaults()/样式被覆盖后应视为未生效——
    用两个本样式独有的指纹参数核对。
    """
    return (_STYLE_APPLIED
            and mpl.rcParams["axes.spines.top"] is False
            and mpl.rcParams["mathtext.fontset"] == "stix"
            and mpl.rcParams["savefig.pad_inches"] == 0.02)


def is_draft() -> bool:
    """当前是否草稿档（run_qa 用）。"""
    return _DRAFT_MODE


def apply_style(base_size: float = 9.0, draft: bool = False) -> None:
    """印刷档字号（pt）：正文 9，刻度 8，注释 7。dpi 只影响预览。

    draft=True 为 72h 赛时草稿档：降 dpi、save_figure 只出 PNG，
    交付前必须用默认档重出一遍。
    """
    global _DRAFT_MODE, _STYLE_APPLIED
    _DRAFT_MODE = draft
    _STYLE_APPLIED = True
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
        "figure.dpi": 100 if draft else 150,
        "savefig.dpi": 120 if draft else 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.axisbelow": True,
        "mathtext.fontset": "stix",
    })


def new_figure(width: str | float = "onehalf", ratio: float = 0.62,
               **kwargs):
    """按栏宽建图。width 取 COLUMN_WIDTHS 键名或 mm 数值；ratio=高/宽。"""
    if isinstance(width, str):
        if width not in COLUMN_WIDTHS:
            raise ValueError(
                f"未知栏宽 '{width}'，可选：{sorted(COLUMN_WIDTHS)} 或 mm 数值")
        w_mm = COLUMN_WIDTHS[width]
    else:
        w_mm = width
    figsize = (w_mm * MM, w_mm * MM * ratio)
    return plt.subplots(figsize=figsize, **kwargs)


def delivered_width_in(fig, tight: bool = True) -> float:
    """交付 PNG 的实际宽度（英寸）：tight 裁剪 + 补白后的宽度。

    run_qa 校验这个值而非 fig.get_figwidth()——后者是裁剪前的画布，
    与落盘文件无关。
    """
    if not tight:
        return fig.get_figwidth()
    fig.canvas.draw()
    pad = mpl.rcParams["savefig.pad_inches"]
    bb_w = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad).width
    # save_figure 会把窄于声明宽度的 bbox 对称补回，故取 max
    return max(bb_w, fig.get_figwidth())


def save_figure(fig, path_no_ext: str, formats=("png", "svg"),
                tight: bool = True, exact_width: bool = True) -> list[str]:
    """导出 png(300dpi)+svg(文字可编辑)。返回输出文件列表。

    tight=True 时不直接用 savefig.bbox="tight"——那会把留白裁掉，
    交付宽度比声明栏宽小 3%–22%，排版放大后同一论文字号不一致。
    这里显式算 tight bbox 后把宽度对称补回声明栏宽（exact_width）。
    tight=False 用于 3D 图：mplot3d 的轴标签不计入 tight bbox，
    会被裁掉，此时改走 subplots_adjust 手动边距。
    草稿档（apply_style(draft=True)）只出 PNG。
    """
    from pathlib import Path
    Path(path_no_ext).parent.mkdir(parents=True, exist_ok=True)
    if _DRAFT_MODE:
        formats = ("png",)
        path_no_ext = f"{path_no_ext}_DRAFT"   # 草稿不与交付物混名
    bbox = None
    if tight:
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        pad = mpl.rcParams["savefig.pad_inches"]
        bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
        target = fig.get_figwidth()
        if exact_width and bb.width < target:
            dx = (target - bb.width) / 2
            bb = Bbox.from_extents(bb.x0 - dx, bb.y0, bb.x1 + dx, bb.y1)
        elif bb.width > target + 3 / 25.4:   # 与 run_qa 的 ±3mm 容差一致
            print(f"[style WARN] 内容溢出声明栏宽："
                  f"{bb.width * 25.4:.0f} > {target * 25.4:.0f} mm")
        if not exact_width:
            w_mm = bb.width * 25.4
            if not any(abs(w_mm - t) <= 3 for t in COLUMN_WIDTHS.values()):
                print(f"[style WARN] exact_width=False 且落盘宽 {w_mm:.0f}mm "
                      f"不是标准栏宽，字号契约不成立")
        bbox = bb
    out = []
    for ext in formats:
        p = f"{path_no_ext}.{ext}"
        fig.savefig(p, bbox_inches=bbox if tight else fig.bbox_inches)
        out.append(p)
    return out
