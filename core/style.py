"""全局样式：期刊栏宽、CJK 字体解析、印刷级 rcParams。

两档预设（SPEC §2）::

    apply_style("nature")   # 严格对齐 Nature 官方规范：sans-serif、正文 ≤7pt、
                            # 无背景网格、线宽 ≤1pt、黑色文字、矢量优先
    apply_style("cn")       # 中文数模交付档（默认）：衬线与论文正文同族、
                            # 字号大一级、极淡实线网格；层次/配色规则与 nature 同源

用法::

    from core import apply_style, new_figure, save_figure
    apply_style()                      # 默认 cn 档，1.5 栏
    fig, ax = new_figure(width="single", ratio=0.75)
    ...
    save_figure(fig, "gallery/示例")   # 自动导出 png + svg + pdf
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

MM = 1 / 25.4  # mm -> inch

# Nature 系栏宽；"cn" 为中文数模 A4 正文常用图宽
COLUMN_WIDTHS = {"single": 89, "onehalf": 136, "double": 183, "cn": 150}

# Nature 明文上限：图高不得超过 170 mm（整页图含图注的可用高度）
MAX_HEIGHT_MM = 170.0

# ---------------------------------------------------------------- 字体候选
# nature 档：Nature 要求 "sans-serif typeface, preferably Helvetica or Arial"
_LATIN_SANS = ["Arial", "Helvetica", "Helvetica Neue", "Nimbus Sans",
               "Liberation Sans", "DejaVu Sans"]
_CJK_SANS = ["Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei",
             "PingFang SC", "Hiragino Sans GB", "SimHei", "WenQuanYi Micro Hei"]

# cn 档：与中文论文宋体正文同族
_LATIN_SERIF = ["Times New Roman", "STIXGeneral", "Liberation Serif",
                "DejaVu Serif"]
_CJK_SERIF = ["Source Han Serif SC", "Noto Serif CJK SC", "SimSun", "STSong",
              "STZhongsong", "Microsoft YaHei", "SimHei"]

# font_manager 的 weight 可能是 int 也可能是字符串，统一成数值再比
_WEIGHT_WORDS = {
    "thin": 100, "extralight": 200, "ultralight": 200, "light": 300,
    "normal": 400, "regular": 400, "book": 400, "medium": 500,
    "semibold": 600, "demibold": 600, "bold": 700, "extrabold": 800,
    "ultrabold": 800, "black": 900, "heavy": 900,
}


def _as_weight(w) -> int:
    if isinstance(w, (int, float)):
        return int(w)
    return _WEIGHT_WORDS.get(str(w).strip().lower(), 400)


def font_weights(name: str) -> list[int]:
    """某字体家族在本机安装的所有 upright 字面的字重。"""
    return sorted({_as_weight(f.weight)
                   for f in font_manager.fontManager.ttflist
                   if f.name == name and f.style == "normal"})


def _has_regular(name: str, lo: int = 350, hi: int = 550) -> bool:
    """家族是否存在接近 Regular(400) 的字面。

    只按名字判断"装没装"是不够的：思源系列常常只安装了 Heavy(900)
    一个字面，matplotlib 会拿它当 Regular 用——中文全部渲染成粗黑块，
    紧挨着 400 字重的拉丁数字，看起来像 markdown 加粗没解析。
    这是本 skill 历史上最伤观感的一个缺陷，必须在解析期就滤掉。
    """
    return any(lo <= w <= hi for w in font_weights(name))


_FONT_REPORT: dict = {}


def font_report() -> dict:
    """上一次 apply_style 的字体解析结果（run_qa 与排障用）。"""
    return dict(_FONT_REPORT)


def _resolve_fonts(preset: str) -> list[str]:
    """按预设解析字体栈：拉丁在前、CJK 在后，逐字符回退。

    两级筛选：家族已安装 且 存在 Regular 字面。全被滤掉时退回
    matplotlib 自带 DejaVu（宁可拉丁难看，也不让中文变粗黑块）。
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    latin_c, cjk_c = ((_LATIN_SANS, _CJK_SANS) if preset == "nature"
                      else (_LATIN_SERIF, _CJK_SERIF))
    dropped = []

    def _pick(cands):
        ok = []
        for c in cands:
            if c not in installed:
                continue
            if _has_regular(c):
                ok.append(c)
            else:
                dropped.append((c, font_weights(c)))
        return ok

    latin, cjk = _pick(latin_c), _pick(cjk_c)
    _FONT_REPORT.update(preset=preset, latin=latin, cjk=cjk, dropped=dropped)
    if dropped:
        for name, ws in dropped:
            print(f"[style] 跳过 {name}：本机只装了字重 {ws}，无 Regular 字面，"
                  f"用它会让中文显著粗于拉丁")
    if not cjk:
        print("[style WARN] 未找到含 Regular 字面的中文字体，中文可能变豆腐块")
    return (latin + cjk) or ["DejaVu Sans"]


# ---------------------------------------------------------------- 预设
_PRESETS = {
    # Nature 官方规范：sans-serif / 正文 5–7pt / 无背景网格 / 线宽 0.25–1pt
    "nature": dict(
        base_size=7.0, grid=False, grid_style="-", grid_alpha=0.0,
        line_width=1.0, axes_lw=0.5, mathtext="dejavusans",
        panel_label_size=8.0, panel_label_fmt="{}",   # 小写 a b c，无括号
    ),
    # 中文数模交付档：与论文宋体正文同族，字号大一级，极淡实线网格
    "cn": dict(
        base_size=9.0, grid=True, grid_style="-", grid_alpha=0.22,
        line_width=1.2, axes_lw=0.6, mathtext="stix",
        panel_label_size=10.0, panel_label_fmt="({})",
    ),
}

_DRAFT_MODE = False
_STYLE_APPLIED = False
_PRESET = "cn"


def current_preset() -> str:
    """当前样式档（annotate/layout/qa 按档调整行为）。"""
    return _PRESET


def preset_cfg(key: str | None = None):
    cfg = _PRESETS[_PRESET]
    return cfg[key] if key else dict(cfg)


def is_styled() -> bool:
    """当前 rcParams 是否处于本 skill 样式态（run_qa 用）。

    不只看调用标志：rcdefaults()/样式被覆盖后应视为未生效——
    用本样式独有的指纹参数核对（按档取值）。
    """
    if not _STYLE_APPLIED:
        return False
    cfg = _PRESETS[_PRESET]
    return (mpl.rcParams["axes.spines.top"] is False
            and mpl.rcParams["mathtext.fontset"] == cfg["mathtext"]
            and mpl.rcParams["savefig.pad_inches"] == 0.02
            and mpl.rcParams["axes.grid"] is cfg["grid"])


def is_draft() -> bool:
    """当前是否草稿档（run_qa 用）。"""
    return _DRAFT_MODE


def apply_style(preset: str = "cn", base_size: float | None = None,
                draft: bool = False) -> None:
    """应用样式档。preset ∈ {"cn", "nature"}；base_size 覆盖档内默认字号。

    nature 档字号锁在 Nature 上限 7pt（面板标签 8pt 由 panel_label 单独给）；
    cn 档 9pt，适配 A4 中文论文缩放后的可读性。
    draft=True 为 72h 赛时草稿档：降 dpi、save_figure 只出 PNG，
    交付前必须用默认档重出一遍。
    """
    global _DRAFT_MODE, _STYLE_APPLIED, _PRESET
    if preset not in _PRESETS:
        raise ValueError(f"未知样式档 '{preset}'，可选：{sorted(_PRESETS)}")
    _DRAFT_MODE, _STYLE_APPLIED, _PRESET = draft, True, preset
    cfg = _PRESETS[preset]
    bs = cfg["base_size"] if base_size is None else base_size
    if preset == "nature" and bs > 7.0:
        print(f"[style WARN] nature 档正文字号上限 7pt，收到 {bs}pt —— "
              f"Nature 会退稿重排，确认是有意为之")
    fonts = _resolve_fonts(preset)
    mpl.rcParams.update({
        # 直接给列表才能触发逐字符回退（拉丁用拉丁字体、中文用 CJK 字体）
        "font.family": fonts,
        "font.serif": fonts, "font.sans-serif": fonts,
        "font.weight": "normal",
        "axes.unicode_minus": False,
        "font.size": bs,
        "axes.titlesize": bs + 0.5,
        "axes.labelsize": bs,
        "xtick.labelsize": bs - 1,
        "ytick.labelsize": bs - 1,
        "legend.fontsize": bs - 1.5,
        "axes.linewidth": cfg["axes_lw"],
        "xtick.major.width": cfg["axes_lw"], "ytick.major.width": cfg["axes_lw"],
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        # Nature: "No background gridlines"。cn 档保留极淡实线（虚线网格
        # 铺满是很强的 matplotlib 味，实线更接近排版惯例）
        "axes.grid": cfg["grid"], "grid.linewidth": 0.3,
        "grid.alpha": cfg["grid_alpha"], "grid.linestyle": cfg["grid_style"],
        "grid.color": "0.75",
        "lines.linewidth": cfg["line_width"], "lines.markersize": 4,
        # Nature: 用 keyline/key，不靠彩色文字；文字一律黑色
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8", "legend.fancybox": False,
        "figure.dpi": 100 if draft else 150,
        "savefig.dpi": 120 if draft else 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        # TrueType 2/42 内嵌，Nature 明确禁止 Type 3
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.axisbelow": True,
        "mathtext.fontset": cfg["mathtext"],
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
    h_mm = w_mm * ratio
    if h_mm > MAX_HEIGHT_MM:
        print(f"[style WARN] 图高 {h_mm:.0f}mm 超 Nature 上限 "
              f"{MAX_HEIGHT_MM:.0f}mm，收 ratio 到 "
              f"{MAX_HEIGHT_MM / w_mm:.2f} 以内")
    figsize = (w_mm * MM, h_mm * MM)
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


_QA_PASSED: set[int] = set()


def mark_qa_passed(fig) -> None:
    """run_qa 无问题时登记，save_figure 据此放行。"""
    _QA_PASSED.add(id(fig))


def save_figure(fig, path_no_ext: str, formats=("png", "svg", "pdf"),
                tight: bool = True, exact_width: bool = True,
                force: bool = False) -> list[str]:
    """导出 png(300dpi) + svg + pdf。返回输出文件列表。

    Nature 主图只收矢量（.pdf/.eps 优先），明确拒收 .png/.jpeg/.tiff——
    故默认三件套：pdf 交付、svg 可编辑、png 供 Read 目测复核。

    tight=True 时不直接用 savefig.bbox="tight"——那会把留白裁掉，
    交付宽度比声明栏宽小 3%–22%，排版放大后同一论文字号不一致。
    这里显式算 tight bbox 后把宽度对称补回声明栏宽（exact_width）。
    tight=False 用于 3D 图：mplot3d 的轴标签不计入 tight bbox，
    会被裁掉，此时改走 subplots_adjust 手动边距。
    草稿档（apply_style(draft=True)）只出 PNG。
    """
    from pathlib import Path
    # "坏图不落盘"必须是机制，不能靠自觉：run_qa(strict=False) 只打印
    # 问题然后照常保存，等于 QA 不存在。这里把它变成门禁。
    if id(fig) not in _QA_PASSED and not force:
        raise RuntimeError(
            "run_qa 未通过或未调用，坏图不落盘。修好问题后重跑；"
            "确需带问题交付传 save_figure(..., force=True)")
    waived = getattr(fig, "_ff_qa_waived", None)
    Path(path_no_ext).parent.mkdir(parents=True, exist_ok=True)
    if _DRAFT_MODE:
        formats = ("png",)
        path_no_ext = f"{path_no_ext}_DRAFT"   # 草稿不与交付物混名
    if waived or (id(fig) not in _QA_PASSED and force):
        # 绕过 QA 的图在交付目录里必须一眼可见，否则下次没人记得
        path_no_ext = f"{path_no_ext}_QAWAIVED"
        print(f"[style WARN] 该图绕过了 QA，文件名已标 _QAWAIVED："
              f"{(waived or ['force=True'])[:2]}")
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
