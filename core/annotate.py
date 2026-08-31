"""注释层：统计注释框、引线标注、线端直标、参考线。

图内自含统计证据，读者不用查图例/正文——这是参考图的核心气质。
但注释是**证据层**不是装饰层：Nature 的视觉编辑五问（哪些是必要元素 /
有没有缺 / 删掉什么还能说清 / 有无重复 / 有无装饰）对注释同样适用。
角落里堆一个灰框小字不会让图变期刊级，只会变成模板签名。

nature 档下文字一律黑色、语义色只走边框（keyline）——Nature 明文
"Avoid coloured text"，改用 keylines/keys 承载语义。
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.transforms import blended_transform_factory

from .style import current_preset, preset_cfg, _one_of
from .colors import emphasis, semantic


def _ann_size(explicit: float | None = None) -> float:
    """注释字号：跟随当前档正文字号下调 1pt，并夹在 Nature 下限 5pt 以上。"""
    if explicit is not None:
        return explicit
    return max(5.0, plt.rcParams["font.size"] - 1.0)


def ink(color, min_ratio: float = 3.3):
    """把用作**文字**的语义色压暗到对白底 ≥3.3:1。

    目标比 QA 的 3.0 门槛**留一点余量**：按 3.0 压的话，一个刚好 3.06 的
    颜色（如判据色 #CC79A7）落在任何浅色带上就跌到 2.98——ink 的目标与
    检查阈值相等，等于按构造就没有余量。

    Okabe-Ito 的橙 #e69f00 对白底只有 2.25:1，用它写端点直标在屏幕上
    已经发虚、印刷后更糟。线条可以靠粗细补偿，文字不行——所以只压
    文字用色，线条仍用原色以保持与图例/标记的语义一致。
    """
    from .colors import luminance
    r, g, b = mcolors.to_rgb(color)
    for _ in range(24):
        if 1.05 / (luminance((r, g, b)) + 0.05) >= min_ratio:
            break
        r, g, b = r * 0.88, g * 0.88, b * 0.88
    return mcolors.to_hex((r, g, b))


def text_color(color: str) -> str:
    """nature 档强制黑字（语义色留给边框/标记）；cn 档允许彩色文字，
    但压暗到对白底 ≥3:1。

    压暗放在这里而不是各调用点：此前只在 dot_interval / slope_lines 两处
    调了，callout / end_label / end_labels 仍吐原色，实测 8 张 gallery 图
    还有 2.25:1 的直标。着色点每多一个就漏一个，只能下沉到唯一入口。
    """
    if current_preset() == "nature":
        return "black"
    return ink(color)


def _mark_lw(v: float) -> float:
    """标记类线宽按档封顶。

    nature 档数据线上限 1pt（run_qa 硬拦），而森林图的粗横棒、斜率图的
    高亮线在 cn 档下本该有分量——按档封顶，而不是两边都迁就。
    """
    if current_preset() != "nature":
        return v
    return min(v, float(preset_cfg("line_width")))


def _halo(txt, lw: float = 2.0):
    """给文字加白色描边光晕，替代不透明白底方框。

    白底方框会在图上打一个洞：它遮住自己压着的那段曲线、色块、
    甚至相邻的数据点，而描边只让字周围一圈变白，底下的图仍可见。
    实测中"白底标签"是遮挡缺陷里出现最频繁的机制。
    """
    import matplotlib.patheffects as pe
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])
    return txt


def stat_box(ax, lines, loc: str = "auto", fontsize: float | None = None,
             edgecolor: str = "0.7", facecolor: str = "white",
             alpha: float = 0.85, pad: float = 0.45,
             outside: str | None = None, expand_axes: bool = True):
    """统计注释框。lines 为 str 列表，如 ['n = 692', 'RMS = 5.068 cm']。

    三段式：设置（网格/步长/样本量）+ 结果（RMS/极值/占比）+ 判定。
    只写结果的框读者无法检验可信度。

    loc: "auto"（默认）在 8 个锚点里自动选压数据最少的；也可显式给
    upper/lower + left/right/center，此时仅在明显更差时才被自动改位。

    outside: "top"/"bottom" 把框放到坐标区**外**。满铺的场图（imshow /
    pcolormesh / contourf）轴内不存在真空位，`loc="auto"` 会自动降级到
    outside="top"——这正是期刊场图把统计框放在坐标区顶部的做法。

    expand_axes: 最佳位置仍压数据时，按框高扩一档轴限腾出真空带，
    而不是压上去。
    """
    from . import _probe

    text = "\n".join(lines)
    _one_of("stat_box(loc=)", loc, ("auto",) + tuple(_probe._ANCHORS))
    if outside is not None:
        _one_of("stat_box(outside=)", outside, ("top", "bottom"))

    size = _ann_size(fontsize)
    bbox = dict(boxstyle=f"round,pad={pad}", facecolor=facecolor,
                edgecolor=edgecolor, alpha=alpha, linewidth=0.5)

    if outside is None and loc == "auto" and _probe.has_field(ax):
        outside = "top"          # 满铺的场里没有真空位，别硬塞

    # Axes3D.text 的签名是 (x, y, z, s)，要走 text2D 才能用 transAxes
    draw = ax.text2D if hasattr(ax, "text2D") else ax.text

    def _place_outside(side, sz):
        """轴外落位：量出真实高度后实测避让排版文字；放不下返回 None。"""
        probe = draw(0.0, 1.02, text, transform=ax.transAxes, ha="left",
                     va="bottom", fontsize=sz, color="black",
                     linespacing=1.5, bbox=bbox, zorder=10)
        ax.figure.canvas.draw()
        _bp = probe.get_bbox_patch()
        h_px = (_bp.get_window_extent() if _bp is not None
                else probe.get_window_extent()).height
        yf = _probe.outside_y(ax, side, h_px)
        if yf is None:
            probe.remove()
            return None
        probe.set_position((0.0, yf))
        probe.set_va("bottom" if side == "top" else "top")
        # 打标记：stat_box 常在 set_title 之前调用，那时量不到标题高度。
        # run_qa 会在检查前调 reflow_outside() 用最终版面重排一次。
        probe._ff_outside = side
        return probe

    if outside in ("top", "bottom"):
        # 上方与面板标题天然争同一条带，先试下方——xlabel 只有一行，
        # 且位置可实测，冲突面小得多
        # 下方若还有面板，外推向下会侵占它——先试上方
        _order = ("top", "bottom") if _has_axes_below(ax) else             ("bottom", "top")
        for side in _order:
            for sz in (size, size - 0.5):
                got = _place_outside(side, max(sz, 5.0))
                if got is not None:
                    return got
        # 轴外两侧都放不下 → 回落轴内自动选位，别硬压标题
        outside, loc = None, "auto"

    if hasattr(ax, "text2D"):
        # 3D 轴没有二维意义上的"占用栅格"，自动选位无从谈起；
        # 投影后到处都可能有线，统一放到坐标区外最稳
        return draw(0.0, 1.02, text, transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=size, color="black",
                    linespacing=1.5, bbox=bbox, zorder=10)

    # 先按试探位置画一次，量出真实尺寸，再决定最终落点
    probe_loc = "upper left" if loc == "auto" else loc
    x, y, ha, va = _probe.anchor_pos(probe_loc)
    t = draw(x, y, text, transform=ax.transAxes, ha=ha, va=va,
             fontsize=size, color="black", linespacing=1.5, bbox=bbox,
             zorder=10)
    ax.figure.canvas.draw()
    tb = t.get_window_extent()
    ab = ax.get_window_extent()
    if ab.width <= 0 or ab.height <= 0:
        return t
    w_frac = min(tb.width / ab.width, 0.98)
    h_frac = min(tb.height / ab.height, 0.98)
    pick, cost = _probe.best_loc(ax, w_frac, h_frac,
                                 prefer=None if loc == "auto" else loc)

    # 八个角都压数据 → 说明这个面板没有真空位，移到坐标区外。
    # 必须走 _place_outside（实测避让 + 打标记让 reflow 能重排），
    # 不能像以前那样硬写 1.02——那正好是面板标题所在的那条带。
    if cost > 0.15:
        t.remove()
        _o = (("top", "bottom") if _has_axes_below(ax)
              else ("bottom", "top"))
        got = _place_outside(_o[0], size) or _place_outside(_o[1], size)
        if got is not None:
            return got
        if expand_axes and not _probe.has_field(ax):
            # 轴外两侧都放不下（多面板里常见）。此前是**静默压回轴内**、
            # 等着被 QA 的遮挡检查拦下，而 expand_axes 这个参数在签名、
            # docstring、api.md 三处都登记了"按框高扩一档轴限腾出真空带"，
            # 函数体却从未读过它。这里把它实现出来：往数据少的那一侧扩。
            _y0, _y1 = ax.get_ylim()
            _need = (h_frac + 0.04) * (_y1 - _y0)
            _occ = _probe.occupancy(ax)
            _top_busy = float(np.mean(_occ[-max(1, len(_occ) // 5):]))
            _bot_busy = float(np.mean(_occ[:max(1, len(_occ) // 5)]))
            if _top_busy <= _bot_busy:
                ax.set_ylim(_y0, _y1 + _need)
                pick = "upper left"
            else:
                ax.set_ylim(_y0 - _need, _y1)
                pick = "lower left"
            ax.figure.canvas.draw()
            ab = ax.get_window_extent()
        t = draw(*_probe.anchor_pos(pick)[:2], text, transform=ax.transAxes,
                 fontsize=size, color="black", linespacing=1.5, bbox=bbox,
                 zorder=10)
        ax.figure.canvas.draw()

    x, y, ha, va = _probe.anchor_pos(pick)
    t.set_position((x, y))
    t.set_ha(ha)
    t.set_va(va)
    # 落位后钳回轴框内：bbox 的 padding 是按点算的，锚在 0.98 的框实际会
    # 突出到 y>1.0，正好顶进面板标题那条带——这是"轴内框压住标题"的来源
    ax.figure.canvas.draw()
    bp = t.get_bbox_patch()
    pb = bp.get_window_extent() if bp is not None else t.get_window_extent()
    over_top = pb.y1 - ab.y1
    over_bot = ab.y0 - pb.y0
    if over_top > 0 or over_bot > 0:
        dy = (-over_top if over_top > 0 else over_bot) / max(ab.height, 1e-6)
        t.set_position((x, y + dy))
    return t


def callout(ax, xy, text, xytext=None, color: str = "#3D7A6B",
            fontsize: float | None = None, rad: float = 0.12,
            textcoords: str = "data", mark: bool = False,
            box: bool | None = None):
    """引线标注：短引线指向关键点（极值、拐点、最优解）。

    xy: 数据坐标目标点；xytext: 文字位置（默认数据坐标，可用 'axes fraction'）。
    mark=True 在目标点画同色小标记，读者可确认箭头指向
    （目标点本身没有星标/散点时务必开启）。

    rad 默认 0.12 而非大弧：Nature 点名"多种箭头粗细/样式而含义不明"
    是常见错误，横穿数据区的大弧引线既遮数据又制造无意义的视觉噪声。
    文字尽量放在目标点近旁，引线越短越好。
    """
    # 有限性要在 matplotlib 的**单位换算之后**判。直接
    # `np.asarray(xy, dtype=float)` 是把"有限性检查"写成了"原始值必须是
    # float"，会连分类轴（xy=("B", 2)）和日期轴一起拒掉——那是完全合法的
    # matplotlib 用法。原始 xy 照旧交给 annotate，别在这里替它做换算。
    try:
        _conv = [ax.convert_xunits(xy[0]), ax.convert_yunits(xy[1])]
        _fin = bool(np.all(np.isfinite(np.asarray(_conv, dtype=float))))
    except (TypeError, ValueError, IndexError):
        _fin = True     # 换算后仍非数值：交给 matplotlib 自己报，别瞎猜
    if not _fin:
        raise ValueError(
            f"callout 的目标点 xy={xy!r} 含非有限值——matplotlib 会把这条"
            f"标注渲染成 1x1 的退化框，肉眼完全不可见，而 run_qa 一路报"
            f"PASS：你以为标注上了，交付的图上没有。先在上游处理掉 nan")
    if mark:
        ax.plot([xy[0]], [xy[1]], "o", color=color, markersize=4.0,
                markeredgecolor="white", markeredgewidth=0.7, zorder=11)
    from . import _probe
    if xytext is None:
        # 自动挑方向：在目标点周围八向里选占用最低的一格，引线长度
        # 不超过轴对角线的 1/4——手填数据坐标最容易把框填到数据正中
        xytext, textcoords = _auto_xytext(ax, xy), "axes fraction"
    if box is None:
        # 满铺的场图上不用白底框：在色块中间开一个白洞，比压住几条线更糟。
        # 改用白描边文字——底下的场仍然看得见，字也读得清。
        box = not _probe.has_field(ax)
        if not box and _probe.bg_luminance(ax, xy) < 0.62:
            # 但深色/高饱和底上白描边同样读不清，这时还是要半透明白底
            box = True
            _forced_box = True
    ann = ax.annotate(
        text, xy=xy, xytext=xytext, textcoords=textcoords,
        fontsize=_ann_size(fontsize), color=text_color(color),
        ha="center", va="center",
        bbox=(dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor=color, alpha=0.92, linewidth=0.6)
              if box else None),
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.6,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=2, shrinkB=3),
        zorder=11,
    )
    if not box:
        _halo(ann, lw=2.4)
    elif locals().get("_forced_box"):
        # 深色场上的半透明白底框是**主动选择**（白描边在深底上读不清），
        # 不是"忘了避让"。打标记让 QA 的场覆盖检查放行——引线注释必须
        # 锚在数据点上，没有"挪到坐标区外"这个选项。
        ann._ff_intentional_box = True
    return ann


def _auto_xytext(ax, xy):
    """在目标点周围找低占用落点，返回 axes fraction 坐标。"""
    from . import _probe
    grid = _probe.occupancy(ax)
    ny, nx = grid.shape
    px, py = ax.transData.transform(xy)
    ab = ax.get_window_extent()
    fx = (px - ab.x0) / max(ab.width, 1e-6)
    fy = (py - ab.y0) / max(ab.height, 1e-6)
    best, bcost = (0.5, 0.5), 9.9
    for r in (0.16, 0.24, 0.32):
        for ang in range(0, 360, 30):
            a = np.deg2rad(ang)
            cx = float(np.clip(fx + r * np.cos(a), 0.10, 0.90))
            cy = float(np.clip(fy + r * np.sin(a), 0.10, 0.90))
            i = int(np.clip(cy * ny, 0, ny - 1))
            j = int(np.clip(cx * nx, 0, nx - 1))
            # 半径小的优先：引线越短越好
            cost = grid[max(i - 2, 0):i + 3, max(j - 3, 0):j + 4].mean() + r
            if cost < bcost:
                best, bcost = (cx, cy), cost
    return best


def end_label(ax, x, y, text, color, dx_pt: float = 4.0,
              fontsize: float | None = None, fontweight: str = "bold"):
    """线端直接标注（替代图例）：在曲线末端右侧写名字/数值。

    Nature 建议"尽可能在图内用 key/keyline 标注"——直标优于图例，
    读者不必在图例与曲线间来回扫视。
    dx_pt: 相对线端的水平偏移，单位 points。
    """
    return _halo(ax.annotate(
        text, xy=(x, y), xytext=(dx_pt, 0), textcoords="offset points",
        color=text_color(color), fontsize=_ann_size(fontsize),
        fontweight=fontweight, ha="left", va="center", zorder=10,
        annotation_clip=False,
    ))


def end_labels(ax, items, dx_pt: float = 4.0, fontsize: float | None = None,
               fontweight: str = "bold", min_gap_pt: float = 9.0):
    """成组的线端直标，自动竖直错开避免叠字。

    items: [(x, y, text, color), ...]。直标是图例的更优替代（读者不必在
    图例与曲线间来回扫视），但多条曲线末端 y 值接近时标签必然重叠——
    这正是大多数人放弃直标改用图例的原因。这里按 y 排序后强制拉开
    至少 min_gap_pt 点的间距，把直标变成可以无脑使用的默认做法。

    返回 Annotation 列表，顺序与 items 一致。
    """
    items = list(items)
    if not items:
        return []
    size = _ann_size(fontsize)
    # 在显示坐标里错开：数据坐标的"接近"与视觉上的"叠字"不是一回事
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    ys_disp = [ax.transData.transform((0, items[i][1]))[1] for i in order]
    adj = list(ys_disp)
    # ys_disp 是**显示像素**，min_gap_pt 是**点**——直接相比是单位混用。
    # 本项目 figure.dpi=150，于是 9pt 实际只拉开 9×72/150 = 4.3pt，而标签
    # 本身就有 7.6pt 高：两条线末端接近时必然叠字，还会被自家的直标互压
    # 检查拦下，而 api.md 承诺的是"一次排完并自动避让"。
    _dpi = ax.figure.dpi or 72.0
    _gap_px = min_gap_pt * _dpi / 72.0
    for k in range(1, len(adj)):
        if adj[k] - adj[k - 1] < _gap_px:
            adj[k] = adj[k - 1] + _gap_px
    # 整体回中，避免全部被顶到上方
    shift = (sum(ys_disp) - sum(adj)) / len(adj)
    adj = [a + shift for a in adj]

    out: list = [None] * len(items)
    for k, i in enumerate(order):
        x, yv, text, color = items[i]
        dy_pt = (adj[k] - ys_disp[k]) * 72.0 / ax.figure.dpi
        out[i] = _halo(ax.annotate(
            text, xy=(x, yv), xytext=(dx_pt, dy_pt),
            textcoords="offset points", color=text_color(color),
            fontsize=size, fontweight=fontweight, ha="left", va="center",
            zorder=10, annotation_clip=False))
    return out



def dot_interval(ax, labels, est, lo, hi, threshold=None, thr_label="",
                 better="high", sizes=None, sort=True, value_col=True,
                 focus=None, ctx="0.42", band="0.965", fontsize=None,
                 return_order=False):
    """点区间图（forest / dot-and-interval）：少量离散档位 + 区间 + 判据。

    数模里"四个档位""六个方案""三条策略"是最常见的数据形状。**数据少
    不是图空的理由**——把它画成折线是语义错（折线宣称档位之间可插值），
    画成均值柱是丢掉不确定度。正解是每档一行：点=估计、横棒=区间。

    五条硬默认（聚合自 resource/ref/_INDEX.md 第二批 14 张范例）：
    ①排序即论点 ②判据参考线+方向语义 ③context 压灰+焦点上彩
    ④判定进视觉编码 ⑤每点直标且标签走轴外数值列。

    threshold : 判据值。达标与否由**区间靠判据的那一侧**决定——点估计
                过线而下界没过，不能算达标。
    better    : "high" 下界 ≥ threshold 才达标；"low" 上界 ≤ threshold。
    sizes     : 可选，编码第二个量（样本量/根数）到点面积。
    value_col : True/"outside" 轴外右侧数值列（单图首选，版面最干净）；
                "inside" 就近直标在区间端点外侧、贴右边界时自动翻到左侧
                （多面板里没有轴外空间时用）；False 不标。

    return_order: 一并返回排序索引。**sort=True 会重排行**，只拿 ok 去
                  索引原始数组必然错位（图题指名的档位和图上标成达标的
                  那行对不上）。`原始索引 = order[显示行号]`。

    返回 ok（**排序后**顺序）；return_order=True 时返回 (ok, order)。
    """
    import numpy as np
    from .colors import PALETTE, semantic

    labels = list(labels)
    est = np.asarray(est, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if not (len(labels) == est.size == lo.size == hi.size):
        raise ValueError(
            f"labels/est/lo/hi 长度必须一致，收到 {len(labels)}/{est.size}/"
            f"{lo.size}/{hi.size}——zip 静默截断会让整行数据无声消失")
    if sizes is not None:
        _sz0 = np.asarray(sizes, dtype=float).ravel()
        if not np.all(np.isfinite(_sz0)):
            raise ValueError(
                "sizes 含非有限值：min/ptp 会被 NaN 传染，导致**每一行**的"
                "markersize 都变成 nan、所有点静默不渲染")
        if _sz0.size != est.size:
            raise ValueError(
                f"sizes 长度 {_sz0.size} 与 est 长度 {est.size} 不一致"
                f"——太短会 IndexError，太长会静默截断（标量同样非法）")
    # better 无校验时，任何非 "high" 的值（"higher" / "HIGH" / 拼错）
    # 都落进低值优分支，把每一行的达标判定**整个反转**——图上的达标
    # 着色、轴外数值列、返回的 ok 全跟着反，而没有任何一条 QA 会响。
    _one_of("dot_interval(better=)", better, ("high", "low"))
    # value_col 的实现只判 truthy：`value_col="False"` 这种字符串是
    # truthy，用户想关掉数值列、结果照画；`"no"` 同理。合法集见 docstring。
    _one_of("dot_interval(value_col=)", value_col,
            (True, False, "outside", "inside"))
    order = np.argsort(est) if sort else np.arange(len(est))
    labels = [labels[i] for i in order]
    est, lo, hi = est[order], lo[order], hi[order]
    sz = None if sizes is None else np.asarray(sizes, float)[order]
    size = _ann_size(fontsize)
    focus = PALETTE[2] if focus is None else focus

    if threshold is None:
        ok = np.ones(len(est), bool)
    elif better == "high":
        ok = lo >= threshold
    else:
        ok = hi <= threshold

    # 非有限行不画、不参与判定，数值列写"—"。此前它们会静默变成"未达标"
    # 并在数值列印出字面量 nan（姊妹函数 slope_lines 早就处理了这一类）。
    fin = np.isfinite(est) & np.isfinite(lo) & np.isfinite(hi)
    ok = ok & fin

    y = np.arange(len(est))
    # 端帽高度必须用**点**折算，不能写死数据单位：0.20 行高在窄区间上
    # 会变成区间宽度的 3 倍，整个字形读成一根竖棒而不是横向区间。
    _axh_pt = max(1e-6, ax.get_position().height
                  * ax.figure.get_figheight() * 72.0)
    _cap = min(0.20, 3.0 * (len(est) + 0.4) / _axh_pt)
    # 交替行底纹：多行时肉眼串行的唯一解药，也让稀疏面板不至于只有
    # 几根线漂着（ref/dotinterval__15）
    for i in range(len(est)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=band, zorder=0, linewidth=0)

    _rows = []          # (行号, 区间线, 标记, ms, 色) —— 轴限定型后回改
    for i in range(len(est)):
        if not fin[i]:
            continue
        c = focus if ok[i] else ctx
        _seg, = ax.plot([lo[i], hi[i]], [y[i], y[i]], color=c,
                        linewidth=_mark_lw(3.0 if ok[i] else 2.0),
                        solid_capstyle="butt", zorder=2)
        for x in (lo[i], hi[i]):        # 端帽：短区间没帽子会退化成点
            ax.plot([x, x], [y[i] - _cap, y[i] + _cap], color=c,
                    linewidth=_mark_lw(1.6 if ok[i] else 1.2), zorder=2)
        # **面积**随 sz 线性，不是直径。直径线性时面积比会被平方放大
        # （实测 N_A 之比 2.0×、面积之比 3.16×），图里若写"点面积 ∝ N"
        # 就是事实错误。
        if sz is None:
            ms = 6.5
        else:
            if np.ptp(sz) == 0:
                ms = 6.5            # 全等 = 没有第二个量可编码，回中性尺寸
            else:
                _t = (sz[i] - sz.min()) / np.ptp(sz)
                ms = float(np.sqrt(4.5 ** 2 + (8.0 ** 2 - 4.5 ** 2) * _t))
        _pt, = ax.plot(est[i], y[i], "o", markersize=ms,
                       color=c if ok[i] else "white", markeredgecolor=c,
                       markeredgewidth=1.3, zorder=4,
                       clip_on=False)   # 估计值贴轴边界时别被切成半圆
        _rows.append((i, _seg, _pt, ms, c))

    if threshold is not None:
        hl = semantic("highlight")
        ax.axvline(threshold, color=hl, linewidth=_mark_lw(1.1),
                   linestyle="--", zorder=1)
        x0, x1 = ax.get_xlim()
        left = threshold > (x0 + x1) / 2   # 标签放线的**空侧**
        txt = thr_label or f"判据 {threshold:g}"
        ax.text(threshold, len(est) - 0.30,
                (txt + " ") if left else (" " + txt), color=text_color(hl),
                fontsize=size, va="center", ha="right" if left else "left")
        lo_t, hi_t = ("未达标 ←", "→ 达标") if better == "high" \
            else ("→ 达标", "未达标 ←")
        ax.text(threshold, -0.60, lo_t + " ", ha="right", va="center",
                fontsize=size, color="0.45")
        ax.text(threshold, -0.60, " " + hi_t, ha="left", va="center",
                fontsize=size, color=text_color(hl))

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    # 有判据时底部要多留一档：方向语义（未达标 ← | → 达标）放在最下一行
    # 之下，-0.7 的下限会让它贴死在轴脊线上。
    ax.set_ylim(-0.85 if threshold is not None else -0.7, len(est) - 0.3)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # 全列统一小数位：`:.3g` 会让同一列出现 0.994 与 0.0698，位数参差、
    # 无法按小数点对齐。位数按**最大值**需要的有效位定——按最小值定会让
    # 一个 1e-4 的下界把整列拖成 6 位。
    _fv = np.abs(np.concatenate([est[fin], lo[fin], hi[fin]]))         if fin.any() else np.array([1.0])
    _pos = _fv[_fv > 0]
    _mag = _pos.max() if _pos.size else 1.0
    # 跨量级超过 10³ 时统一小数位必然坑一端（按最大值定会把 3.4e-3 印成
    # 0，按最小值定会把 1.2e6 拖成一长串），这种列只能退回有效数字。
    _wide = bool(_pos.size) and _pos.max() / _pos.min() > 1e3
    _dec = None if _wide else int(
        np.clip(3 - np.floor(np.log10(_mag)) - 1, 0, 6))

    def _num(v):
        return f"{v:.3g}" if _dec is None else f"{v:.{_dec}f}"

    # 区间比标记还窄时：标记转空心、区间压到标记**之上**，否则那一行只剩
    # 一个实心圆点，而最窄的往往正是结论所在的高精度档——整张图要论证的
    # "下界过线"在图上看不见。
    # 这一步**必须在轴限定型之后**：绘制过程中 viewLim 还停在默认 (0,1)，
    # 拿它换算出的像素宽度与真实渲染无关，实测两个方向都判反（0.45pt 的
    # 极窄区间给了实心点、38.8pt 的极宽区间反倒全转空心）。
    try:
        ax.autoscale_view()
        _dpi = ax.figure.dpi or 100.0
        for i, _seg, _pt, _ms, _c in _rows:
            _px = abs(ax.transData.transform((hi[i], y[i]))[0]
                      - ax.transData.transform((lo[i], y[i]))[0])
            if _px * 72.0 / _dpi < 1.1 * _ms:
                _pt.set_markerfacecolor("white")
                _seg.set_zorder(6)
    except Exception as e:
        print(f"[annotate note] 窄区间判定未执行：{e}")

    if value_col == "inside":
        # 多面板里轴外没有空间：标签跟着区间端点走，贴右边界时翻到左侧
        # ——"标签放空白处"（ref/orderedbar__23）
        x0, x1 = ax.get_xlim()
        for i in range(len(est)):
            if not fin[i]:
                continue
            flip = hi[i] > x0 + 0.72 * (x1 - x0)
            ax.annotate(_num(est[i]),
                        xy=(lo[i] if flip else hi[i], y[i]),
                        xytext=(-6 if flip else 6, 0),
                        textcoords="offset points",
                        va="center", ha="right" if flip else "left",
                        fontsize=size, color=text_color(focus if ok[i] else "0.35"),
                        fontweight="bold" if ok[i] else "normal")
        ax.set_xlim(x0, x1)
    elif value_col:
        xr = ax.get_xlim()

        for i in range(len(est)):
            if not fin[i]:
                ax.annotate("—（无有效数据）",
                            xy=(1.06, y[i]),
                            xycoords=("axes fraction", "data"),
                            va="center", ha="left", fontsize=size,
                            color="0.55", annotation_clip=False)
                continue
            ax.annotate(f"{_num(est[i])} "
                        f"[{_num(lo[i])}, {_num(hi[i])}]",
                        xy=(1.06, y[i]), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=size,
                        color=text_color(focus if ok[i] else "0.35"),
                        fontweight="bold" if ok[i] else "normal",
                        annotation_clip=False)
        ax.annotate("估计 [95% 区间]", xy=(1.06, len(est) - 0.32),
                    xycoords=("axes fraction", "data"), va="center",
                    ha="left", fontsize=size, color="0.3",
                    annotation_clip=False)
        ax.set_xlim(*xr)
        # 自己让位。数值列锚在轴外 1.06 且关了 clip，不预留右侧版面就会
        # 画出画布（实测溢出 9%）。把"记得 subplots_adjust"写进 docstring
        # 当调用方的义务，等于把已知缺陷转嫁给用户。
        # 收缩本轴而不是 subplots_adjust：后者会连带挪动多面板里的其他格。
        try:
            _fg = ax.figure
            for _ in range(3):
                _fg.canvas.draw()
                _rd = _fg.canvas.get_renderer()
                _need = max(t.get_window_extent(_rd).x1 for t in ax.texts)
                # 右界不只是画布右缘：多面板里右边还有邻居，越界会直接
                # 骑进隔壁面板，而 QA 的"压数据"只遍历带框注释，兜不住
                # 无框直标。取"画布右缘"与"右邻面板左缘"里更靠左的那个。
                _me = ax.get_window_extent(_rd)
                _limit = _fg.bbox.x1
                for _o in _fg.get_axes():
                    if _o is ax or _o.get_label() == "<colorbar>"                             or not _o.get_visible():
                        continue
                    # **判据用坐标区盒、度量用 tightbbox**，两者不能混用：
                    # 邻居的 ylabel 往左伸出后 tightbbox.x0 会跑到本轴右缘
                    # 左边，拿它做"是不是右邻"的判据会把邻居整个排除，
                    # 让位一次都不收缩（实测比不改还差一倍）。
                    _ow = _o.get_window_extent(_rd)
                    if not (_ow.x0 >= _me.x1 and _ow.y1 > _me.y0
                            and _ow.y0 < _me.y1):
                        continue
                    # 收缩到邻居的 tightbbox：ylabel 与刻度标签在坐标区
                    # 左侧之外，只避开坐标区仍会压上去。不可见轴的
                    # get_tightbbox() 返回 **None**（不抛异常），
                    # 不挡住会让 AttributeError 把整个让位循环吞掉。
                    _ot = None
                    try:
                        _ot = _o.get_tightbbox(_rd)
                    except Exception:
                        _ot = None
                    _limit = min(_limit, _ow.x0 if _ot is None else _ot.x0)
                _over = _need - _limit
                if _over <= 0:
                    break
                _ps = ax.get_position()
                _sh = (_over + 4) / max(1.0, _fg.bbox.width)
                _x1 = max(_ps.x0 + 0.15, _ps.x1 - _sh)
                ax.set_position([_ps.x0, _ps.y0, _x1 - _ps.x0, _ps.height])
            if _over > 1.0:
                # 撞到 0.15 下限后再收缩也无效。静默放弃会让用户拿到一张
                # 数值列骑进邻居的图而毫无提示，且 QA 的遮挡检查只遍历带框
                # 注释，兜不住无框直标。
                print(f"[annotate note] 数值列让位未完成，仍越界 "
                      f"{_over:.0f}px：改用 value_col='inside' 或加宽画布")
        except Exception as e:
            print(f"[annotate note] 数值列让位未执行：{e}")
    return (ok, order) if return_order else ok


def slope_lines(ax, labels, before, after, cond_names=("前", "后"), unit="",
                highlight=(), higher_is_better=True, mode="emphasis",
                verdict="", label_ends=True, fontsize=None):
    """斜率图：两状态各一列，每个个体一条连线。交叉即名次反转。

    替代"两期分组柱"：分组柱要读者在八根柱子间来回比高度，斜率图把
    "谁涨谁跌、谁反超谁"直接变成线的方向与交叉。

    mode:
      "emphasis"（默认，少量类别）——**每条线都按变化方向着色**，
          highlight 的加粗加深、其余降透明度。把非高亮线一律压成灰
          会丢掉方向信息，而方向正是这张图要说的事
          （ref/slopegraph__31：升红降蓝，半透明叠加显密度）。
      "cohort"（大 N）——群体线压成中性灰当背景，只有 highlight 上色
          （ref/slopegraph__30：灰群体 + 彩色高亮个体）。

    verdict: 判定/显著性文字，标在**面板顶部**而非塞进图例
             （ref/slopegraph__31 的 n.s. / ** / ***）。读者不必查表。

    label_ends: 两端直标"名称 值"。同一端 y 值接近的标签会自动错开
                （ref/slopegraph__32：六个类别挤在一起也不重叠）。

    返回 dict(up=, down=, flat=)。
    """
    import numpy as np
    from .colors import semantic

    _one_of("slope_lines(mode=)", mode, ("emphasis", "cohort"))
    labels = list(labels)
    b = np.asarray(before, float)
    a = np.asarray(after, float)
    if not (len(labels) == b.size == a.size):
        raise ValueError(
            f"labels/before/after 长度必须一致，收到 {len(labels)}/{b.size}/"
            f"{a.size}——zip 静默截断会让整行数据无声消失")
    hl = set()
    for i in highlight:
        j = int(i)
        j = j + len(labels) if j < 0 else j
        if not 0 <= j < len(labels):
            raise ValueError(
                f"highlight 索引 {i} 越界（共 {len(labels)} 项）。越界索引"
                f"永不匹配，emphasis 档会把所有线降透明、cohort 档一个"
                f"标签都不画，整张图静默退化成无标注灰线团")
        hl.add(j)
    size = _ann_size(fontsize)
    up_c, dn_c = semantic("good"), semantic("bad")
    if not higher_is_better:
        up_c, dn_c = dn_c, up_c

    def _fmt(v):
        """有效数字按量级定：:g 会把 56.4157 原样吐出来，读者读不动。"""
        av = abs(v)
        if av >= 100:
            return f"{v:.0f}"
        if av >= 10:
            return f"{v:.1f}"
        if av >= 1:
            return f"{v:.2f}"
        return f"{v:.3g}"

    ends = {0: [], 1: []}          # 端点标签，画完线再统一避让
    counts = {"up": 0, "down": 0, "flat": 0, "invalid": 0}
    for i, (lb, bi, ai) in enumerate(zip(labels, b, a)):
        # 非有限值不能算进"无变化"：NaN 与任何数比较都是 False，会掉进
        # flat 分支，而调用方把 counts 写进图题 → 图上声称"该项无变化"。
        # inf 还会让下面的标签避让算出 NaN，draw 时抛无线索的 StopIteration。
        if not (np.isfinite(bi) and np.isfinite(ai)):
            counts["invalid"] += 1
            continue
        d = ai - bi
        counts["up" if d > 0 else "down" if d < 0 else "flat"] += 1
        emph = (i in hl) if hl else True
        if mode == "cohort" and not emph:
            # 群体线仍然承载数据，不能淡到印不出来。0.72@0.62 叠白底的
            # 等效灰度 0.83，对比度仅 1.3:1；0.40@0.70 → 等效 0.58，约
            # 3:1，同时保留 alpha 叠加显示线束密度的作用。
            c, alpha, lw, ms = "0.40", 0.70, _mark_lw(0.9), 2.6
        else:
            c = up_c if d > 0 else dn_c if d < 0 else "0.55"
            alpha = 0.95 if emph else 0.5
            lw = _mark_lw(1.9 if emph else 1.0)
            ms = 5.2 if emph else 3.4
        ax.plot([0, 1], [bi, ai], "-o", color=c, alpha=alpha, linewidth=lw,
                markersize=ms, markeredgecolor="white", markeredgewidth=0.7,
                zorder=4 if emph else 2)
        if label_ends and not (mode == "cohort" and not emph):
            w = "bold" if emph and hl else "normal"
            ends[0].append([bi, f"{lb} {_fmt(bi)}{unit}", c, w,
                            size if emph else size - 0.5])
            ends[1].append([ai, f"{_fmt(ai)}{unit}", c, w,
                            size if emph else size - 0.5])

    ax.set_xlim(-0.45, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(cond_names, fontsize=size + 1)
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    if verdict:
        # 给判定文字**单独的顶部留白带**：直接压在数据上方必然撞上最高
        # 那条线的端点直标（实测撞「方案A 72.0 分」27%）。参考图里的
        # n.s./*** 也都待在专门的 headroom 里。
        _v0, _v1 = ax.get_ylim()
        ax.set_ylim(_v0, _v1 + 0.13 * (_v1 - _v0))

    # 同一端标签避让：按 y 排序后贪心撑开，最小间距由字号折算成数据单位
    y0, y1 = ax.get_ylim()
    # 间距按渲染字高折算：固定比例在不同 y 量程/面板高度下必然失准，
    # 实测 40 条线的 cohort 图里两个高亮标签仍然叠在一起。
    ax_h_in = max(1e-6, ax.get_position().height * ax.figure.get_figheight())
    gap = size * 1.5 * (y1 - y0) / (ax_h_in * 72.0)
    for side, items in ends.items():
        if not items:
            continue
        items.sort(key=lambda t: t[0])
        pos = [t[0] for t in items]
        for k in range(1, len(pos)):
            if pos[k] - pos[k - 1] < gap:
                pos[k] = pos[k - 1] + gap
        # 整体回拉，避免撑开后越出上边界
        over = pos[-1] - (y1 - 0.02 * (y1 - y0))
        if over > 0:
            pos = [p - over for p in pos]
        for (yv, txt, c, w, fs), py in zip(items, pos):
            # 锚到**撑开后**的 py：锚回 yv 等于避让白算，标签照样重叠，
            # 而下面那条引线会从数据点指向一个空位置。
            ax.annotate(txt, xy=(side, py),
                        xytext=(-6 if side == 0 else 6, 0),
                        textcoords="offset points",
                        ha="right" if side == 0 else "left", va="center",
                        fontsize=fs, color=text_color(c), fontweight=w,
                        annotation_clip=False)
            if abs(py - yv) > gap * 0.35:   # 挪动明显时补一条细引线
                ax.annotate("", xy=(side, yv), xytext=(side, py),
                            arrowprops=dict(arrowstyle="-", color=c,
                                            linewidth=0.5, alpha=0.6))

    if verdict:
        # 放面板**内**顶部（ref/slopegraph__31 的 n.s./*** 就在panel 内）：
        # 放轴外 y>1 必然和调用方的 set_title / suptitle 打架。
        ax.text(0.5, 0.99, verdict, transform=ax.transAxes, ha="center",
                va="top", fontsize=size + 0.5, fontweight="bold",
                color=text_color("0.25"), zorder=6)
    return counts

def ref_line(ax, value, orientation: str = "h", label: str | None = None,
             color: str | None = None, fontsize: float | None = None,
             label_loc: str | None = None, level: str = "focus",
             linewidth: float | None = None):
    """参考线 + 端点小标签（如 90% 阈值线）。

    level ∈ {focus, context, background} 控制视觉权重：
    **当这条线就是图的论点（"过没过阈值"）时必须留在 focus**——
    把论点线调成灰色、却让无关 marker 高饱和，是最典型的层次倒置。
    纯背景基准（y=x 对角线、零线）才用 context/background。

    orientation ∈ {"h" 横线（默认）, "v" 竖线}；其他值直接 ValueError。
    此前 docstring 与 api.md 都只写默认值 "h"、从没说过 "v" 存在，
    用户想画竖直阈值线翻遍文档也不知道该传什么。

    label_loc: 横线用 left/right（默认 right），竖线用 top/bottom
    （默认 top）。传另一方向的值会报错——竖线分支此前**根本不读**这个
    参数（文档登记、函数体不读的死参数），横线分支则把任何非 "right"
    的值静默当成 "left"。
    """
    # 走 semantic()，不要硬编码：换判据色时这里不跟着改，库里就会出现
    # 两个互不相同的"判据线色"（dot_interval 用 semantic("highlight")、
    # ref_line 用写死的值），同一张图里阈值线与其交点标记会是两种颜色。
    base = color if color is not None else (
        semantic("highlight") if level == "focus" else semantic("baseline"))
    c = base if level == "focus" else emphasis(base, level)
    lw = linewidth if linewidth is not None else (
        1.0 if level == "focus" else 0.7)
    size = _ann_size(fontsize)
    if orientation not in ("h", "v"):
        # 无校验时任何非 "h" 的值都静默落进竖线分支：`orientation=
        # "horizontal"` 这种极自然的写法会把阈值点错轴，而 run_qa 全部
        # 检查无一命中——图照常交付。这是唯一一条会让用户拿到**几何
        # 错误**的缺陷，必须在入口拦掉。
        raise ValueError(
            f"orientation 只能是 'h'（横线）或 'v'（竖线），收到 "
            f"{orientation!r}")
    _legal = ("left", "right") if orientation == "h" else ("top", "bottom")
    if label_loc is None:
        label_loc = "right" if orientation == "h" else "top"
    _one_of(f"ref_line(label_loc=) 在 orientation={orientation!r} 下",
            label_loc, _legal)
    tcol = text_color(c)   # 内含 ink() 压暗；判据色做文字时正好卡在 3:1 线上
    if orientation == "h":
        ax.axhline(value, color=c, linewidth=lw, linestyle=(0, (4, 3)),
                   zorder=2)
        if label:
            tf = blended_transform_factory(ax.transAxes, ax.transData)
            xa, ha = (0.985, "right") if label_loc == "right" else (0.015,
                                                                    "left")
            t = ax.text(xa, value, label, transform=tf, ha=ha, va="bottom",
                        fontsize=size, color=tcol, zorder=3)
            _halo(t)
    else:
        ax.axvline(value, color=c, linewidth=lw, linestyle=(0, (4, 3)),
                   zorder=2)
        if label:
            tf = blended_transform_factory(ax.transData, ax.transAxes)
            ya, va = ((0.985, "top") if label_loc == "top"
                      else (0.015, "bottom"))
            t = ax.text(value, ya, label, transform=tf, ha="left",
                        va=va, rotation=90, fontsize=size, color=tcol,
                        zorder=3)
            _halo(t)


def _has_axes_below(ax) -> bool:
    """该轴正下方是否还有别的坐标区（堆叠多面板）。

    此时把图例外推到轴下会直接侵占下一个面板——那里通常已经有自己的
    统计框或数据。多面板图里"轴外"并不是无主空间。
    """
    p0 = ax.get_position()
    for other in ax.figure.get_axes():
        if other is ax or not other.get_visible():
            continue
        p1 = other.get_position()
        if p1.y1 <= p0.y0 + 1e-6 and not (p1.x1 < p0.x0 or p1.x0 > p0.x1):
            return True
    return False

def _reflow_legend(ax):
    """按最终版面重排图例：smart_legend 常在数据画完前调用，位置只能按
    当时的状态选；这里用完整版面重算一次。"""
    from . import _probe
    leg = ax.get_legend()
    args_kw = getattr(ax, "_ff_legend_args", None)
    if leg is None or args_kw is None:
        return 0
    args, kw = args_kw
    fig = ax.figure
    fig.canvas.draw()
    lb, ab = leg.get_window_extent(), ax.get_window_extent()
    if ab.width <= 0 or ab.height <= 0:
        return 0
    pick, cost = _probe.best_loc(ax, min(lb.width / ab.width, 0.98),
                                 min(lb.height / ab.height, 0.98))
    leg.remove()
    if cost <= 0.10 or _has_axes_below(ax):
        ax.legend(*args, loc=pick, frameon=True, **kw)
    else:
        yf = _probe.outside_y(ax, "bottom", lb.height)
        ax.legend(*args, loc="upper left",
                  bbox_to_anchor=(0.0, yf if yf is not None else -0.22),
                  ncol=max(1, min(3, len(ax.get_legend_handles_labels()[1]))),
                  frameon=False, fontsize=_ann_size() - 0.5, **kw)
    return 1


def reflow_outside(fig):
    """把所有"坐标区外"的统计框按最终版面重新落位。

    stat_box 往往在 set_title / suptitle 之前被调用，那时标题还不存在，
    轴外落位只能靠预留高度猜——猜不准就把结论句压掉。run_qa 在检查前
    调用本函数，此时标题、图题、刻度都已确定，可以实测避让。
    """
    from . import _probe
    moved = 0
    for ax in fig.get_axes():
        try:
            moved += _reflow_legend(ax)
        except Exception:
            pass
        for t in list(ax.texts):
            side = getattr(t, "_ff_outside", None)
            if side is None:
                continue
            for attempt in range(3):
                fig.canvas.draw()
                bp = t.get_bbox_patch()
                h_px = (bp.get_window_extent() if bp is not None
                        else t.get_window_extent()).height
                placed = False
                _tr = ("top", "bottom") if _has_axes_below(ax) else                     ("bottom", "top")
                for trial in _tr:
                    yf = _probe.outside_y(ax, trial, h_px)
                    if yf is not None:
                        t.set_position((0.0, yf))
                        t.set_va("bottom" if trial == "top" else "top")
                        t._ff_outside = trial
                        moved += 1
                        placed = True
                        break
                if placed:
                    # 落位后再按"带边框"的真实范围微调：bbox 的 padding 会
                    # 超出锚点，仍可能蹭到 xlabel / 刻度 / 标题
                    for _ in range(3):
                        fig.canvas.draw()
                        bp2 = t.get_bbox_patch()
                        if bp2 is None:
                            break
                        pb = bp2.get_window_extent()
                        ab2 = ax.get_window_extent()
                        others = [ax.title, ax.xaxis.label] +                             list(ax.get_xticklabels())
                        _sup = getattr(fig, "_suptitle", None)
                        if _sup is not None and _sup.get_text().strip():
                            others.append(_sup)
                        leg2 = ax.get_legend()
                        if leg2 is not None and leg2.get_visible():
                            others.append(leg2)
                        # 让开方向由障碍物在上还是在下决定，不由外推方向
                        # 决定——撞到图题（在上方）却继续往上推，只会越撞越深
                        push = 0.0
                        for o in others:
                            if hasattr(o, "get_text") and                                     not o.get_text().strip():
                                continue
                            ob = o.get_window_extent()
                            if pb.y1 <= ob.y0 or pb.y0 >= ob.y1:
                                continue
                            if pb.x1 <= ob.x0 or pb.x0 >= ob.x1:
                                continue
                            up = ob.y1 - pb.y0      # 障碍在下 → 往上让
                            down = pb.y1 - ob.y0    # 障碍在上 → 往下让
                            step = up if up < down else -down
                            if abs(step) > abs(push):
                                push = step
                        if abs(push) <= 0.5:
                            break
                        dy = push / max(ab2.height, 1e-6) * 1.05
                        x0, y0 = t.get_position()
                        t.set_position((x0, y0 + dy))
                    break
                # 两侧都放不下 → 让版面腾地方：把坐标区压矮一档再试。
                # 注释层是版面的一部分，放不下就该缩轴，而不是压住标题。
                pos = ax.get_position()
                need = h_px / fig.bbox.height * 1.15
                if pos.height - need < 0.25:
                    break                       # 再缩就没图了
                ax.set_position([pos.x0, pos.y0,
                                 pos.width, pos.height - need])
            else:
                # 三次都放不下 → 回落轴内最优位。绝不能把框留在原来的
                # 轴外坏位置：那正是"框压住图题"的来源。
                fig.canvas.draw()
                bp3 = t.get_bbox_patch()
                tb3 = (bp3 if bp3 is not None else t).get_window_extent()
                ab3 = ax.get_window_extent()
                if ab3.width > 0 and ab3.height > 0:
                    pick, _ = _probe.best_loc(
                        ax, min(tb3.width / ab3.width, 0.98),
                        min(tb3.height / ab3.height, 0.98))
                    px, py, pha, pva = _probe.anchor_pos(pick)
                    t.set_position((px, py))
                    t.set_ha(pha)
                    t.set_va(pva)
                    t._ff_outside = None
                    moved += 1
    return moved


def smart_legend(ax, *args, **kw):
    """会实测选位的图例：优先放轴内真空角，放不下才移到轴外并避让轴标题。

    `ax.legend(loc="upper left")` 之类的显式定位完全绕过占用探测，是
    "图例压住数据/压住 x 轴标题"的主要来源；而无脑 `bbox_to_anchor=(0,-0.16)`
    把图例甩到轴下，又正好砸在 xlabel 上。这里两头都实测。
    """
    from . import _probe
    kw.pop("loc", None)
    kw.pop("bbox_to_anchor", None)
    kw.pop("frameon", None)
    leg = ax.legend(*args, loc="upper left", frameon=True, **kw)
    ax._ff_legend_args = (args, dict(kw))     # 供 reflow_outside 重排时重建
    fig = ax.figure
    fig.canvas.draw()
    lb = leg.get_window_extent()
    ab = ax.get_window_extent()
    if ab.width <= 0 or ab.height <= 0:
        return leg
    w_frac = min(lb.width / ab.width, 0.98)
    h_frac = min(lb.height / ab.height, 0.98)
    pick, cost = _probe.best_loc(ax, w_frac, h_frac)
    if cost <= 0.10 or _has_axes_below(ax):
        # 重建而不是 set_loc：不同 matplotlib 版本对 set_loc 的支持不一致，
        # 静默失败的后果是图例留在原地压着数据，而调用方以为已经挪好了
        leg.remove()
        leg = ax.legend(*args, loc=pick, frameon=True, **kw)
        fig.canvas.draw()
        return leg
    # 轴内没地方 → 移到轴下，位置实测让开 xlabel 与刻度
    yf = _probe.outside_y(ax, "bottom", lb.height)
    leg.remove()
    kw.pop("ncol", None)
    kw.pop("ncols", None)
    ncol = max(1, min(4, len(ax.get_legend_handles_labels()[1])))
    return ax.legend(*args, loc="upper left",
                     bbox_to_anchor=(0.0, yf if yf is not None else -0.22),
                     ncol=ncol, frameon=False, **kw)
