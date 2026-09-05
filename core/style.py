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

import os as _os
import weakref
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .journals import JournalProfile, get_journal, journal_names
from .manifest import FigureRecord, complete_record, write_manifest

MM = 1 / 25.4  # mm -> inch

# Nature 系栏宽；"cn" 为中文数模 A4 正文常用图宽
COLUMN_WIDTHS = {"single": 89, "onehalf": 136, "double": 183, "cn": 150}

# 本库 Nature 档推导上限：图高不得超过 170 mm（整页可用高度）
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
        # Nature 正文上限 7pt，标题不能例外（run_qa 硬拦 >7pt）
        title_size=7.0,
    ),
    # 中文数模交付档：与论文宋体正文同族，字号大一级，极淡实线网格
    "cn": dict(
        base_size=9.0, grid=True, grid_style="-", grid_alpha=0.22,
        line_width=1.2, axes_lw=0.6, mathtext="stix",
        panel_label_size=10.0, panel_label_fmt="({})",
        title_size=10.5,
    ),
}

_DRAFT_MODE = False
_STYLE_APPLIED = False
_PRESET = "cn"
_JOURNAL: str | None = None   # 期刊约束档（apply_style(journal=) 设置）


def _one_of(name: str, value, legal):
    """公开 API 的枚举型字符串参数，统一在这里校验。

    早先修了 `ref_line(orientation=)`，但那只是**一类**缺陷的一个
    实例。早先的排查在 `core/` 里又找出五处，主窗口把同一把尺子
    伸到 `recipes/`（api.md 逐个登记了签名，同样是交付面）又找出三处——
    其中 `parity(band=("relatve", …))` 拼错一个字母，写进论文的覆盖率
    从 98% 变成 20%。一处一处补还会继续漏，所以收到这个唯一入口。

    放在 style.py 而不是 annotate.py：recipes 也要用，而 style 不依赖
    任何兄弟模块，从这里往外导不会成环。
    """
    if value not in legal:
        raise ValueError(
            f"未知的 {name} {value!r}，可选：{sorted(legal, key=str)}"
            f"——静默走另一个分支比报错伤得多：报错你当场就改，"
            f"静默画错的图会直接进论文")
    return value


def ptx(v: float, kind: str = "font") -> float:
    """把**按 cn 档调好的**点值换算到当前档。

    recipe 里遍布 `fontsize=9` / `linewidth=1.6` 这类硬编码——它们是照
    cn 档（base 9pt、线宽 1.2）调的。切到 nature 档（base 7pt、上限 1pt）
    后这些值原样生效，于是 25 个模板在 nature 下全部撞"文字 > 7pt"和
    "线宽 > 1pt"的硬拒；而 SKILL.md 把 nature 与 cn 并列宣传为交付档、
    工作流又要求"从 recipes 复制模板"，两者合起来 100% 撞墙。

    kind="font" 按 base_size 比例缩放并夹在该档字号包线内；
    kind="lw"   按该档线宽上限封顶；
    kind="pt"   纯比例缩放、不加任何钳制——用于 markersize 这类"是点值、
                但既不受字号下限也不受线宽上限约束"的量。不缩放它们的话，
                nature 档（base 7 vs cn 9）下标记会相对文字明显偏大。
    """
    if kind not in ("font", "lw", "pt"):
        # 拼错的 kind 会静默落进 font 分支：`ptx(1.6, "linewidth")` 在
        # nature 档返回 5.0（被字号**下限**兜住）而不是 1.0——用户要的是
        # 1pt 线宽，拿到 5pt，整整 5 倍且毫无提示。
        raise ValueError(
            f"未知的 ptx(kind=) {kind!r}，可选：['font', 'lw', 'pt']"
            f"——拼错会静默按字号缩放，线宽会被字号下限兜成 5.0")
    if not v:
        return 0.0          # lw=0（无边框填充）与"不画文字"都是合法输入
    cfg = _PRESETS[_PRESET]
    prof = active_profile()
    if kind == "lw":
        # cn 的 `line_width` 是**默认**线宽，不是上限——把它当上限会把森林图
        # 的粗横棒、斜率图的高亮线一律压到 1.2，而 cn 档根本没有线宽 QA
        # 检查。这与库内既有的 `_mark_lw`（cn 恒等）直接矛盾，同一个库里
        # 不能有两套线宽政策。只有档注册表里有线宽上限（nature 0.25–1pt）
        # 才封顶；ieee/pnas 官方无线宽条款，不造数、不封顶。
        if prof.max_line_pt is None:
            return float(v)
        ref = _PRESETS["cn"]["line_width"]
        out = float(v) * cfg["line_width"] / ref
        return min(out, float(prof.max_line_pt))
    ref = _PRESETS["cn"]["base_size"]
    out = float(v) * float(plt.rcParams.get("font.size", ref)) / ref
    if kind == "pt":
        return out
    lo = prof.min_font_pt if prof.min_font_pt is not None else 6.5
    hi = prof.max_font_pt if prof.max_font_pt is not None else 1e9
    return max(lo, min(out, hi))


def current_preset() -> str:
    """当前样式档（annotate/layout/qa 按档调整行为）。"""
    return _PRESET


def current_journal() -> str | None:
    """当前期刊约束档名（apply_style(journal=) 设置）。

    None = 未显式选期刊档，交付约束沿用 preset 自带档（cn/nature）。
    """
    return _JOURNAL


def active_profile() -> JournalProfile:
    """当前生效的交付约束档：journal 显式指定优先，否则 preset 自带档。

    qa/new_figure/ptx 的档级数值（栏宽、字号区间、线宽上限、图高、
    网格/彩色文字许可）都从这里取，注册表是唯一真值源。
    """
    return get_journal(_JOURNAL) if _JOURNAL is not None else get_journal(_PRESET)


def column_widths() -> dict:
    """当前生效档的栏宽表。"""
    return dict(active_profile().column_widths_mm)


def column_width_mm(key: str) -> float:
    """按当前生效档把栏宽键名解析为 mm。

    报错必须带上「当前哪个档」：期刊档激活时合法键会变少（IEEE 没有
    1.5 栏），只说「可选 ['double','single']」的话，用户看着自己写了
    多轮的 'onehalf' 突然非法，不知道是档换了还是拼错了。
    """
    widths = column_widths()
    if key not in widths:
        where = (f"（当前期刊档 {_JOURNAL!r}）" if _JOURNAL is not None
                 else f"（当前 preset {_PRESET!r}）")
        src = ("——官方栏宽与出处见 core/journals.py"
               if _JOURNAL is not None else "")
        raise ValueError(
            f"未知栏宽 {key!r}{where}，可选：{sorted(widths)} "
            f"或直接给 mm 数值{src}")
    return widths[key]


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
    expected_grid = (cfg["grid"] if active_profile().grid_allowed else False)
    return (mpl.rcParams["axes.spines.top"] is False
            and mpl.rcParams["mathtext.fontset"] == cfg["mathtext"]
            and mpl.rcParams["savefig.pad_inches"] == 0.02
            and mpl.rcParams["axes.grid"] is expected_grid)


def is_draft() -> bool:
    """当前是否草稿档（run_qa 用）。"""
    return _DRAFT_MODE


def apply_style(preset: str | None = None, base_size: float | None = None,
                draft: bool = False, *, journal: str | None = None) -> None:
    """应用样式档。preset ∈ {"cn", "nature"}；base_size 覆盖档内默认字号。

    preset=None 时读环境变量 `FF_PRESET`，缺省 "cn"。**档位选择必须在库里**：
    早先是让 recipe 从 `_common` 拿 PRESET，而 SKILL.md 又指示"复制模板时
    删掉 `from _common import …` 那一行"——照文档走就 NameError，24/24 个
    模板全中。

    nature 档字号锁在 Nature 上限 7pt（面板标签 8pt 由 panel_label 单独给）；
    cn 档 9pt，适配 A4 中文论文缩放后的可读性。
    draft=True 为 72h 赛时草稿档：降 dpi、save_figure 只出 PNG，
    交付前必须用默认档重出一遍。

    journal="ieee"/"pnas" 叠加期刊交付约束档（core/journals.py，正交于
    preset：preset 管语言与纹理，journal 接管该档有出处的整套约束（栏宽、
    字号、图高以及适用的网格/彩字/线宽条款）。journal=None（缺省）逐
    rcParam、逐像素保持既有行为；重复
    调用 apply_style 会重置 journal——每次调用定义完整样式态。档内
    base_font_pt 只在未显式给 base_size 时生效；图题字号夹进该档字号
    区间，刻度/图例字号不落到下限之下。
    """
    global _DRAFT_MODE, _STYLE_APPLIED, _PRESET, _JOURNAL
    if preset is None:
        preset = _os.environ.get("FF_PRESET", "cn")
    if preset not in _PRESETS:
        raise ValueError(f"未知样式档 '{preset}'，可选：{sorted(_PRESETS)}")
    # 未知期刊档在 apply_style 这一层就报（带合法档名与出处指引），
    # 不等画完图 run_qa 才发现
    if journal is not None:
        try:
            _one_of("apply_style(journal=)", journal, tuple(journal_names()))
        except ValueError as exc:
            raise ValueError(f"{exc}；期刊枚举及出处见 core/journals.py") from exc
        prof = get_journal(journal)
    else:
        prof = None
    _DRAFT_MODE, _STYLE_APPLIED, _PRESET = draft, True, preset
    _JOURNAL = journal
    cfg = _PRESETS[preset]
    bs = cfg["base_size"] if base_size is None else base_size
    if prof is not None and prof.base_font_pt is not None \
            and base_size is None:
        bs = prof.base_font_pt
    if journal is None and preset == "nature" and bs > 7.0:
        print(f"[style WARN] nature 档正文字号上限 7pt，收到 {bs}pt —— "
              f"Nature 会退稿重排，确认是有意为之")
    if journal is None:
        title_size = cfg["title_size"]
        tick_size, legend_size = bs - 1, bs - 1.5
    else:
        # 期刊档激活：字号全部夹进该档官方区间。图题按基准字号等比
        # 缩放（cn 10.5/9 在 nature 纹理下是 7/7，切到别的基准要跟着
        # 走），再夹进 [min, max]；刻度/图例同层正文，不低于下限。
        title_size = cfg["title_size"] * bs / cfg["base_size"]
        if prof.min_font_pt is not None:
            title_size = max(title_size, prof.min_font_pt)
        if prof.max_font_pt is not None:
            title_size = min(title_size, prof.max_font_pt)
        floor = prof.min_font_pt if prof.min_font_pt is not None else 0.0
        tick_size, legend_size = max(bs - 1, floor), max(bs - 1.5, floor)
    fonts = _resolve_fonts(preset)
    mpl.rcParams.update({
        # 直接给列表才能触发逐字符回退（拉丁用拉丁字体、中文用 CJK 字体）
        "font.family": fonts,
        "font.serif": fonts, "font.sans-serif": fonts,
        "font.weight": "normal",
        "axes.unicode_minus": False,
        "font.size": bs,

        "axes.labelsize": bs,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        # figure.titlesize 留在 matplotlib 默认 "large"（= 1.2×base）时，
        # nature 档 base=7 会渲染成 8.4pt，而 run_qa 硬拒 >7pt——不加图题
        # 又报"无图题"，该档实际不可用。两个标题字号都必须显式定死。
        # （journal=None 时即 preset 原值；期刊档激活时按官方区间夹取）
        "figure.titlesize": title_size,
        "axes.titlesize": title_size,
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
        "svg.hashsalt": "figure-forge-v1",
        "axes.axisbelow": True,
        "mathtext.fontset": cfg["mathtext"],
    })
    if prof is not None:
        if not prof.grid_allowed:
            mpl.rcParams["axes.grid"] = False
        if prof.max_line_pt is not None:
            mpl.rcParams["lines.linewidth"] = min(
                float(cfg["line_width"]), float(prof.max_line_pt))
        if not prof.colored_text_allowed:
            for key in ("text.color", "axes.labelcolor", "xtick.color",
                        "ytick.color"):
                mpl.rcParams[key] = "black"
        if preset == "nature" and journal in ("ieee", "pnas"):
            print(f"[style note] journal={journal!r} 已接管档级约束："
                  "nature 档的无网格/无彩字/线宽上限三项不再检查"
                  f"（{journal.upper()} 官方无此条款）")


def new_figure(width: str | float = "onehalf", ratio: float = 0.62,
               **kwargs):
    """按栏宽建图。width 取栏宽键名或 mm 数值；ratio=高/宽。

    journal=None 时键名走 COLUMN_WIDTHS（既有行为）；期刊档激活时键名
    走该档官方栏宽（如 ieee 只有 single/double，传 onehalf 直接报错
    ——静默落回 136mm 会让 QA 宽度检查莫名爆红，报错当场就改）。
    """
    if isinstance(width, str):
        w_mm = column_width_mm(width)
    else:
        w_mm = width
    h_mm = w_mm * ratio
    cap = (MAX_HEIGHT_MM if _JOURNAL is None
           else get_journal(_JOURNAL).max_height_mm)
    if h_mm > cap:
        if _JOURNAL is None:
            print(f"[style WARN] 图高 {h_mm:.0f}mm 超 Nature 上限 "
                  f"{MAX_HEIGHT_MM:.0f}mm，收 ratio 到 "
                  f"{MAX_HEIGHT_MM / w_mm:.2f} 以内")
        else:
            print(f"[style WARN] 图高 {h_mm:.0f}mm 超期刊档 '{_JOURNAL}' "
                  f"上限 {cap:.0f}mm，收 ratio 到 {cap / w_mm:.2f} 以内")
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


_QA_PASSED = weakref.WeakSet()


def mark_qa_passed(fig) -> None:
    """run_qa 无问题时登记，save_figure 据此放行。"""
    _QA_PASSED.add(fig)


def revoke_qa_passed(fig) -> None:
    """重新检查 Figure 前撤销旧资格，避免图被改坏后仍可保存。"""
    _QA_PASSED.discard(fig)


def save_figure(fig, path_no_ext: str, formats=("png", "svg", "pdf"),
                tight: bool = True, exact_width: bool = True,
                force: bool = False, record: FigureRecord | None = None,
                manifest_path=None) -> list[str]:
    """导出 png(300dpi) + svg + pdf。返回输出文件列表。

    Nature 主图只收矢量（.pdf/.eps 优先），明确拒收 .png/.jpeg/.tiff——
    故默认三件套：pdf 交付、svg 可编辑、png 供 Read 目测复核。

    tight=True 时不直接用 savefig.bbox="tight"——那会把留白裁掉，
    交付宽度比声明栏宽小 3%–22%，排版放大后同一论文字号不一致。
    这里显式算 tight bbox 后把宽度对称补回声明栏宽（exact_width）。
    tight=False 用于 3D 图：mplot3d 的轴标签不计入 tight bbox，
    会被裁掉，此时改走 subplots_adjust 手动边距。
    草稿档（apply_style(draft=True)）只出 PNG。

    可选溯源台账（core/manifest.py）：同时给 record（FigureRecord，
    填 id/claim/source_data/generation_script 四项）与 manifest_path，
    全部格式落盘成功且 QA 通过后才 upsert 一行进台账；QA 未通过
    （force 绕行）的图不记账。不传这两个参数时行为与旧调用完全一致。
    """
    from pathlib import Path
    # "坏图不落盘"必须是机制，不能靠自觉：run_qa(strict=False) 只打印
    # 问题然后照常保存，等于 QA 不存在。这里把它变成门禁。
    if fig not in _QA_PASSED and not force:
        raise RuntimeError(
            "run_qa 未通过或未调用，坏图不落盘。修好问题后重跑；"
            "确需带问题交付传 save_figure(..., force=True)")
    waived = getattr(fig, "_ff_qa_waived", None)
    Path(path_no_ext).parent.mkdir(parents=True, exist_ok=True)
    if _DRAFT_MODE:
        formats = ("png",)
        path_no_ext = f"{path_no_ext}_DRAFT"   # 草稿不与交付物混名
    if waived or (fig not in _QA_PASSED and force):
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
            _targets = column_widths().values()
            if not any(abs(w_mm - t) <= 3 for t in _targets):
                print(f"[style WARN] exact_width=False 且落盘宽 {w_mm:.0f}mm "
                      f"不是标准栏宽，字号契约不成立")
        bbox = bb
    out = []
    for ext in formats:
        p = f"{path_no_ext}.{ext}"
        save_kw = {"bbox_inches": bbox if tight else fig.bbox_inches}
        if ext == "svg":
            save_kw["metadata"] = {"Date": None}
        elif ext == "pdf":
            save_kw["metadata"] = {"CreationDate": None}
        fig.savefig(p, **save_kw)
        out.append(p)
    if (record is None) != (manifest_path is None):
        raise ValueError(
            "record 与 manifest_path 必须同时给出，只给其一无法定位台账")
    if record is not None:
        if fig in _QA_PASSED:
            # 全部格式落盘成功才记账；QA 未通过的图（force 绕行）没有
            # 资格进台账；豁免码会被透传进 qa_status，避免和干净通过混淆。
            waived_codes = tuple(dict.fromkeys(
                getattr(fig, "_ff_qa_waived_codes", ())))
            qa_status = ("waived:" + ",".join(waived_codes)
                         if waived_codes else "passed")
            write_manifest(
                complete_record(record, out, manifest_path=manifest_path,
                                preset=current_preset(),
                                journal=current_journal(),
                                qa_status=qa_status),
                manifest_path)
        else:
            print(f"[style WARN] 该图未经 run_qa 通过（force/草稿），"
                  f"不入台账：{record.id}")
    return out
