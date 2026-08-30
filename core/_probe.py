"""占用探测：一个面板里"哪里有数据、哪里是真空"。

annotate（自动选注释框位置）与 qa（判定注释框有没有压数据）共用同一套
探测逻辑——两份实现必然漂移，一边说"这里空"另一边说"这里压了"。

核心是两条互补判据：
- **采样点判据**：曲线/散点按落入框内的采样点数与整条系列的吞没率判；
- **像素判据**：把注释层隐藏后重渲一次，直接数框底下的非白像素。
  后者不依赖 artist 类型枚举，任何新画法都兜得住（满铺的场、
  imshow、三角面片、rasterized 的 pcolormesh 都算得对）。
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

# 场类 artist：这类东西按"面积占比"判，不按采样点判
FIELD_CLASSES = ("AxesImage", "QuadMesh", "PcolorImage", "NonUniformImage",
                 "TriMesh", "Poly3DCollection", "Line3DCollection",
                 # contourf / tricontourf 在 mpl≥3.8 的产物。漏掉它们会让
                 # 同一张图 pcolormesh 过、contourf 被判"面板是一维构图"
                 # 硬拒，并被指向 contour_field.py——而那正是唯一用
                 # contourf 的 recipe；stat_box(loc="auto") 的"满铺场图
                 # 自动降级到 outside" 也会因此对 contourf 失效。
                 "QuadContourSet", "TriContourSet")


def densify(xy, per_seg: int = 8):
    """沿折线插值采样：只查顶点会漏掉长直线段穿过框体。"""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2 or len(xy) > 400:
        return xy
    segs = [np.linspace(xy[k], xy[k + 1], per_seg, endpoint=False)
            for k in range(len(xy) - 1)]
    return np.vstack(segs + [xy[-1:]])


def is_filled_field(a) -> bool:
    """这个 artist 是不是**满铺**的场。

    `contour()` 与 `contourf()` 在 mpl≥3.8 同为 `QuadContourSet`，只能靠
    `filled` 属性区分：线等值线的轴大部分是白底，把它当满铺场会让"图例
    压在场图上（框底 100% 是色块）"这类诊断变成与事实相反的硬拒。
    """
    if a.__class__.__name__ not in FIELD_CLASSES:
        return False
    return bool(getattr(a, "filled", True))


def series_samples(ax):
    """返回 [(名称, 显示坐标点集)]：该轴上每条独立数据系列的采样点。

    逐系列返回而非合并，才能算"某条系列被整条吞掉"——这是最伤的
    遮挡形态（读者以为那档没数据），但合并统计下它只占很小比例。
    """
    from matplotlib.collections import LineCollection
    out = []
    tiny: dict = {}
    for i, ln in enumerate(ax.lines):
        if not ln.get_visible():
            continue
        xy = ln.get_xydata()
        if len(xy) < 1:
            continue
        if len(xy) <= 2 and str(ln.get_linestyle()) in ("None", "none", ""):
            # 逐点 ax.plot(x, y, "o") 会生成一堆 1 点 Line2D。各自算一条
            # "系列"的话，框蹭到任意一点都会被判成"整条系列被吞掉"；
            # 按同型（marker+颜色）合并成一个点云才是它真实的语义。
            key = (str(ln.get_marker()), str(ln.get_color()))
            tiny.setdefault(key, []).append(xy)
            continue
        out.append((f"line{i}", ax.transData.transform(densify(xy))))
    for k, (key, chunks) in enumerate(tiny.items()):
        pts = np.vstack(chunks)
        out.append((f"cloud{k}", ax.transData.transform(pts)))
    for j, c in enumerate(ax.collections):
        if is_filled_field(c) or not c.get_visible():
            continue
        if isinstance(c, LineCollection):
            # 整个 collection 算一条系列，不能按 segment 拆：网络图的
            # 612 条边若各算一条"系列"，框下随便压到一条短边就会被判成
            # "整条系列被吞掉"。语义上 collection 才是那条系列。
            segs = [densify(sg) for sg in c.get_segments() if len(sg) >= 1]
            if segs:
                out.append((f"lc{j}",
                            ax.transData.transform(np.vstack(segs))))
        else:
            offs = np.asarray(getattr(c, "get_offsets", lambda: [])())
            # mpl 3.10 的 `FillBetweenPolyCollection.get_offsets()` 返回
            # 退化的 `[[0, 0]]`（size=2），会**抢先命中** offsets 分支，
            # 于是整片置信带只产出一个 (0,0) 幽灵点、顶点采样永远走不到。
            # 判据要排掉"单点零偏移"这种退化值，并用 isinstance 认子类
            # （精确类名比对在 3.10 上对 FillBetweenPolyCollection 失配）。
            _degenerate = (offs.size <= 2
                           and not np.any(np.abs(offs.reshape(-1)) > 1e-12))
            if offs.size and not _degenerate:
                out.append((f"pts{j}", ax.transData.transform(offs)))
            elif isinstance(c, PolyCollection):
                # fill_between / fill 产生的填充带没有（有效的）offsets，
                # 靠顶点采样。漏掉它们的后果是占用栅格说"这里空"、像素
                # 实测说"41% 有内容"，自动选位就会把图例正正放在置信带上。
                verts = []
                for pth in c.get_paths():
                    v = pth.vertices
                    if len(v) >= 3:
                        verts.append(v)
                if verts:
                    vv = np.vstack(verts)
                    if len(vv) > 2000:
                        vv = vv[:: max(1, len(vv) // 2000)]
                    out.append((f"fill{j}", ax.transData.transform(vv)))
    for m, pch in enumerate(ax.patches):
        if not pch.get_visible():
            continue
        try:
            bb = pch.get_window_extent()
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        gx = np.linspace(bb.x0, bb.x1, 4)
        gy = np.linspace(bb.y0, bb.y1, 4)
        out.append((f"patch{m}",
                    np.array([[x, y] for x in gx for y in gy])))
    for k, cont in enumerate(ax.containers):
        pts = []
        for p in getattr(cont, "patches", []):
            bb = p.get_window_extent()
            pts.append([(bb.x0 + bb.x1) / 2, bb.y1])      # 柱顶=读数位置
        if pts:
            out.append((f"bar{k}", np.asarray(pts)))
    return out


def has_field(ax) -> bool:
    """该轴上是否存在满铺的场类 artist（此时轴内不存在真空位）。"""
    arts = list(ax.images) + list(ax.collections)
    return any(a.get_visible() and is_filled_field(a) for a in arts)


def occupancy(ax, nx: int = 40, ny: int = 40):
    """轴内占用栅格（ny, nx），值域 0–1，行序自下而上。

    场类 artist 直接把整个数据区判为满占用——满铺的场里没有真空。
    """
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    grid = np.zeros((ny, nx), dtype=float)
    if bb.width <= 0 or bb.height <= 0:
        return grid
    if has_field(ax):
        grid[:] = 1.0
        return grid
    xs = np.linspace(bb.x0, bb.x1, nx + 1)
    ys = np.linspace(bb.y0, bb.y1, ny + 1)

    def _mark(pts):
        if len(pts) == 0:
            return
        ix = np.clip(np.searchsorted(xs, pts[:, 0]) - 1, 0, nx - 1)
        iy = np.clip(np.searchsorted(ys, pts[:, 1]) - 1, 0, ny - 1)
        grid[iy, ix] = 1.0

    for _, pts in series_samples(ax):
        _mark(pts)
    # 已有的文字标注同样是"占用"：统计框压住直标，和压住曲线一样伤——
    # 后者读者还能猜，前者直接把一个数字切掉一半
    rd = ax.figure.canvas.get_renderer()
    for t in list(ax.texts) + ([ax.get_legend()] if ax.get_legend() else []):
        if t is None or not t.get_visible():
            continue
        if hasattr(t, "get_text") and not t.get_text().strip():
            continue
        try:
            tb = t.get_window_extent(rd)
        except Exception:
            continue
        gx = np.linspace(tb.x0, tb.x1, 6)
        gy = np.linspace(tb.y0, tb.y1, 4)
        _mark(np.array([[x, y] for x in gx for y in gy]))
    # 数据点之间的空隙也不该塞框：把占用向外膨胀一格
    pad = grid.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            pad = np.maximum(pad, np.roll(np.roll(grid, dy, 0), dx, 1))
    return pad


# 候选位置：(x, y, ha, va)，坐标为 axes fraction
_ANCHORS = {
    "upper left": (0.02, 0.98, "left", "top"),
    "upper right": (0.98, 0.98, "right", "top"),
    "lower left": (0.02, 0.02, "left", "bottom"),
    "lower right": (0.98, 0.02, "right", "bottom"),
    "upper center": (0.5, 0.98, "center", "top"),
    "lower center": (0.5, 0.02, "center", "bottom"),
    "center left": (0.02, 0.5, "left", "center"),
    "center right": (0.98, 0.5, "right", "center"),
}


def anchor_pos(loc: str):
    return _ANCHORS[loc]


def anchor_cost(ax, loc: str, w_frac: float, h_frac: float,
                grid=None) -> float:
    """把尺寸 (w_frac, h_frac) 的框放在 loc 处会压住多少数据（0–1）。"""
    grid = occupancy(ax) if grid is None else grid
    ny, nx = grid.shape
    x, y, ha, va = _ANCHORS[loc]
    x0 = {"left": x, "right": x - w_frac,
          "center": x - w_frac / 2}[ha]
    y0 = {"bottom": y, "top": y - h_frac,
          "center": y - h_frac / 2}[va]
    i0, i1 = int(np.floor(y0 * ny)), int(np.ceil((y0 + h_frac) * ny))
    j0, j1 = int(np.floor(x0 * nx)), int(np.ceil((x0 + w_frac) * nx))
    i0, j0 = max(i0, 0), max(j0, 0)
    i1, j1 = min(i1, ny), min(j1, nx)
    if i1 <= i0 or j1 <= j0:
        return 1.0
    return float(grid[i0:i1, j0:j1].mean())


def best_loc(ax, w_frac: float, h_frac: float, prefer: str | None = None):
    """在 8 个锚点里选压数据最少的。返回 (loc, cost)。

    prefer 给定且其代价不劣于最优值 +0.02 时优先保留——调用者的
    构图意图值得尊重，只在明显更差时才改。
    """
    grid = occupancy(ax)
    costs = {loc: anchor_cost(ax, loc, w_frac, h_frac, grid)
             for loc in _ANCHORS}
    best = min(costs, key=costs.get)
    if prefer in costs and costs[prefer] <= costs[best] + 0.02:
        return prefer, costs[prefer]
    return best, costs[best]


def ink_under(fig, rects, hidden) -> list[float]:
    """像素级判据：隐藏 hidden 里的 artist 后重渲，数每个矩形下的非白像素比。

    这是兜底检查——不枚举 artist 类型，任何画法都算得对。
    rects: [(名称, Bbox 显示坐标)]；hidden: 要临时隐藏的 artist 列表。
    返回与 rects 等长的比例列表。
    """
    states = [(a, a.get_visible()) for a in hidden]
    try:
        for a, _ in states:
            a.set_visible(False)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=float)
        rgb, alpha = buf[..., :3], buf[..., 3:4] / 255.0
        # 合成到白底：透明区域算作白，不能当成"有内容"
        comp = rgb * alpha + 255.0 * (1 - alpha)
        nonwhite = (comp < 247).any(axis=2)
        H = nonwhite.shape[0]
        out = []
        for _, bb in rects:
            # buffer 行序自上而下，Bbox 自下而上
            r0 = int(np.clip(H - bb.y1, 0, H))
            r1 = int(np.clip(H - bb.y0, 0, H))
            c0 = int(np.clip(bb.x0, 0, nonwhite.shape[1]))
            c1 = int(np.clip(bb.x1, 0, nonwhite.shape[1]))
            if r1 <= r0 or c1 <= c0:
                out.append(0.0)
            else:
                out.append(float(nonwhite[r0:r1, c0:c1].mean()))
        return out
    finally:
        for a, vis in states:
            a.set_visible(vis)
        fig.canvas.draw()


def panel_ink(fig, ax) -> float:
    """面板墨迹密度：坐标区内非白像素比。参考期刊图约 0.25–0.45。"""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=float)
    rgb, alpha = buf[..., :3], buf[..., 3:4] / 255.0
    comp = rgb * alpha + 255.0 * (1 - alpha)
    nonwhite = (comp < 247).any(axis=2)
    H, W = nonwhite.shape
    bb = ax.get_window_extent()
    r0, r1 = int(np.clip(H - bb.y1, 0, H)), int(np.clip(H - bb.y0, 0, H))
    c0, c1 = int(np.clip(bb.x0, 0, W)), int(np.clip(bb.x1, 0, W))
    if r1 <= r0 or c1 <= c0:
        return 0.0
    return float(nonwhite[r0:r1, c0:c1].mean())


def outside_y(ax, side: str, box_h_px: float):
    """坐标区外落位：实测让开 title / xlabel / suptitle。放不下返回 None。

    以前这里写死 1.02 / -0.16 两个常数，而 ax.title 恰好在 y≈1.0–1.05、
    xlabel 恰好在 y≈-0.14——统计框移到"轴外"后正好压死标题。
    实测一批 10 张图，这个常数落位压掉了 11 处标题，其中 2 处是整张图的
    结论句。轴外不是安全区，只是另一块要避让的区域。
    """
    fig = ax.figure
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    ab = ax.get_window_extent()
    if ab.height <= 0:
        return None
    if side == "top":
        # 预留一行面板标题的高度：stat_box 常在 set_title **之前**被调用，
        # 那时 title 还是空串，量不到——只按当前实测放，稍后标题一画就撞上。
        title_pt = fig.get_figwidth() * 0 + plt.rcParams["axes.titlesize"]
        y_px = ab.y1 + title_pt * fig.dpi / 72.0 * 1.7
        if ax.get_title().strip():
            y_px = max(y_px, ax.title.get_window_extent(rd).y1 + 2)
        # 注意：不能拿 fig.bbox.y1 当天花板。save_figure 用 tight bbox 落盘，
        # 轴外内容会让画布自己长出来——图边框不是约束，只有 suptitle 是。
        sup = getattr(fig, "_suptitle", None)
        if sup is not None and sup.get_text().strip():
            ceil = sup.get_window_extent(rd).y0 - 2
            if y_px + box_h_px > ceil:
                return None
        for other in fig.get_axes():      # 上方同样可能有别的面板
            if other is ax or not other.get_visible():
                continue
            ob = other.get_window_extent()
            if ob.y0 >= ab.y1 - 1e-6 and not (ob.x1 < ab.x0 or
                                              ob.x0 > ab.x1):
                if y_px + box_h_px > ob.y0 - 3:
                    return None
        return (y_px - ab.y0) / ab.height
    y_px = ab.y0
    for art in [ax.xaxis.label] + list(ax.get_xticklabels()):
        if art.get_text().strip():
            y_px = min(y_px, art.get_window_extent(rd).y0 - 3)
    leg = ax.get_legend()          # 图例也常被放到轴下，别叠上去
    if leg is not None and leg.get_visible():
        lb = leg.get_window_extent(rd)
        if lb.y0 < ab.y0:
            y_px = min(y_px, lb.y0 - 3)
    # 同理，图底也不设硬下限（tight bbox 会长出来），只避让已有的整幅图注
    for ft in fig.texts:
        if ft.get_text().strip():
            fb = ft.get_window_extent(rd)
            if fb.y1 < ab.y0 and y_px - box_h_px < fb.y1 + 3:
                return None
    # 关键：多面板图里"轴下"不是无主空间。下方若还有面板，
    # 往那儿放就是侵占别人的地盘——实测遇到过统计框整个盖住
    # 下一个面板的标题与刻度。
    for other in fig.get_axes():
        if other is ax or not other.get_visible():
            continue
        ob = other.get_window_extent()
        if ob.y1 <= ab.y0 + 1e-6 and not (ob.x1 < ab.x0 or ob.x0 > ab.x1):
            if y_px - box_h_px < ob.y1 + 3:
                return None
    return (y_px - ab.y0) / ab.height


def bg_luminance(ax, xy) -> float:
    """取数据点 xy 处的实际底色亮度（0–1），用于决定白描边是否够用。"""
    fig = ax.figure
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=float)
    rgb, alpha = buf[..., :3], buf[..., 3:4] / 255.0
    comp = (rgb * alpha + 255.0 * (1 - alpha)) / 255.0
    H, W = comp.shape[:2]
    px, py = ax.transData.transform(xy)
    r = int(np.clip(H - py, 0, H - 1))
    c = int(np.clip(px, 0, W - 1))
    patch = comp[max(r - 4, 0):r + 5, max(c - 4, 0):c + 5]
    if patch.size == 0:
        return 1.0
    return float((0.299 * patch[..., 0] + 0.587 * patch[..., 1] +
                  0.114 * patch[..., 2]).mean())


def panel_archetype(ax) -> str:
    """面板图种：geom / field / joint / 1d。用于构成比检查。"""
    if getattr(ax, "name", "") == "3d":
        return "geom"
    if has_field(ax):
        return "field"
    for c in ax.collections:
        if c.__class__.__name__ != "PolyCollection":
            continue
        # hexbin 返回的 PolyCollection 只带 **一个** path（六边形原型），
        # 靠 get_paths() 计数会把它误判成一维图；真正的规模在 offsets 上
        n = len(getattr(c, "get_offsets", lambda: [])())
        if n > 50 or len(getattr(c, "get_paths", lambda: [])()) > 50:
            return "joint"          # hexbin / 六边形联合分布
    # 稠密散点场（空间快照、蒙卡落点）承载的是分布结构，不是"一条线"
    for c in ax.collections:
        if c.__class__.__name__ == "PathCollection" and                 len(getattr(c, "get_offsets", lambda: [])()) > 200:
            return "field"
        # 空间网络：几百条线段构成的是拓扑结构，不是折线图
        if c.__class__.__name__ == "LineCollection" and                 len(getattr(c, "get_segments", lambda: [])()) > 80:
            return "field"
    return "1d"


def flat_color_ratio(fig, ax) -> float:
    """场图里单一颜色占的面积比。高墨迹但大片平色 = 指标高而信息低。"""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=float)
    rgb, alpha = buf[..., :3], buf[..., 3:4] / 255.0
    comp = rgb * alpha + 255.0 * (1 - alpha)
    H, W = comp.shape[:2]
    bb = ax.get_window_extent()
    r0, r1 = int(np.clip(H - bb.y1, 0, H)), int(np.clip(H - bb.y0, 0, H))
    c0, c1 = int(np.clip(bb.x0, 0, W)), int(np.clip(bb.x1, 0, W))
    if r1 <= r0 or c1 <= c0:
        return 0.0
    q = (comp[r0:r1, c0:c1] // 16).astype(np.int16).reshape(-1, 3)
    _, cnt = np.unique(q, axis=0, return_counts=True)
    return float(cnt.max() / cnt.sum()) if cnt.size else 0.0
