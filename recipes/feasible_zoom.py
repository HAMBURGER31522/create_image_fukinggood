"""约束优化决策空间：成本等值线 + 可行域填充 + 约束前沿 + 放大窗。

论点合同示例：
- 结论：最低成本点位于前沿右下端；该点邻域平坦，已被放大图排除歧义。
  （具体数字由数据算出、经 f-string 进图题——写死在这里必然与图漂移，
   这正是 SPEC 2.1 要防的手写常数，只不过 docstring 逃过了 QA）
- 证据链：等值线给成本梯度 → 可行域填充给约束 → 前沿点列 → 放大窗消除"平坦区"质疑。
对照：A 题「成本等值线与可行域」（已接近，此模板补注释层与统一工艺）。
"""
from _common import GALLERY
import numpy as np

from core import (text_color, ptx, smart_legend, apply_style, new_figure, save_figure, run_qa,
                  stat_box, callout, inset_zoom, panel_label, semantic)


def feasible_contour_zoom(X, Y, cost, frontier_xy, best, zoom_xlim, zoom_ylim,
                          xlabel="x", ylabel="y", cost_unit="元",
                          width="onehalf"):
    # 放大窗右移一点：原位置的左上角刚好压在主图"20 元"等值线标签上，
    # 窗内 y 刻度与那个标签重叠 55%（QA 的跨轴文字重叠检查抓到的）
    inset_rect = (0.60, 0.52, 0.38, 0.4)
    fig, ax = new_figure(width, ratio=0.78)
    cs = ax.contour(X, Y, cost, levels=10, colors="#5B8DB8", linewidths=0.8)
    # 主图等值线标签手动放置：避开放大窗 footprint 与轴边缘，防止裁切
    xa, xb = np.min(X), np.max(X)
    ya, yb = np.min(Y), np.max(Y)
    fx_frac = (X - xa) / (xb - xa)
    fy_frac = (Y - ya) / (yb - ya)
    ix0, iy0, iw, ih = inset_rect
    ok = ((fx_frac > 0.06) & (fx_frac < 0.94) &
          (fy_frac > 0.06) & (fy_frac < 0.94) &
          ~((fx_frac > ix0 - 0.03) & (fx_frac < ix0 + iw + 0.03) &
            (fy_frac > iy0 - 0.05) & (fy_frac < iy0 + ih + 0.05)))
    step_m = np.median(np.diff(cs.levels)) if len(cs.levels) > 1 else 1.0
    manual_m = []
    for lv in cs.levels:
        d = np.where(ok, np.abs(cost - lv), np.inf)
        idx = np.unravel_index(np.argmin(d), d.shape)
        if d[idx] < 0.25 * step_m:
            manual_m.append((X[idx], Y[idx]))
    if manual_m:
        ax.clabel(cs, inline=True, fontsize=ptx(6.5),
                  colors=text_color("#5B8DB8"),
                  fmt=f"%.0f {cost_unit}", manual=manual_m)

    fx, fy = frontier_xy
    # 可行域 = 前沿上方
    ax.fill_between(fx, fy, np.max(Y), color="#DDEEDD", alpha=0.6, lw=ptx(0, "lw"),
                    zorder=0)
    ax.plot(fx, fy, "-o", color=semantic("fit"), markersize=ptx(3.5, "pt"),
            linewidth=ptx(1.5, "lw"), label="约束前沿", zorder=4)
    ax.plot(*best, "*", color="k", markersize=ptx(13, "pt"), markeredgecolor="white",
            markeredgewidth=ptx(0.6, "lw"), zorder=5, label="最低成本点")
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    smart_legend(ax)

    axins = inset_zoom(ax, inset_rect, zoom_xlim, zoom_ylim)
    csz = axins.contour(X, Y, cost, levels=20, colors="#5B8DB8",
                        linewidths=0.6)
    # 手动选标签位置：只放在窗内 70% 中带，避免被窗框裁切；单位与主图统一
    x0, x1 = zoom_xlim
    y0, y1 = zoom_ylim
    inner = ((X > x0 + 0.15 * (x1 - x0)) & (X < x1 - 0.15 * (x1 - x0)) &
             (Y > y0 + 0.15 * (y1 - y0)) & (Y < y1 - 0.15 * (y1 - y0)))
    step = np.median(np.diff(csz.levels)) if len(csz.levels) > 1 else 1.0
    manual = []
    for lv in csz.levels:
        d = np.where(inner, np.abs(cost - lv), np.inf)
        idx = np.unravel_index(np.argmin(d), d.shape)
        if d[idx] < 0.25 * step:
            manual.append((X[idx], Y[idx]))
    if manual:
        axins.clabel(csz, inline=True, fontsize=ptx(6.5), colors=text_color("#5B8DB8"),
                     fmt=f"%.1f {cost_unit}", manual=manual)
    axins.fill_between(fx, fy, np.max(Y), color="#DDEEDD", alpha=0.6, lw=ptx(0, "lw"),
                       zorder=0)
    axins.plot(fx, fy, "-o", color=semantic("fit"), markersize=ptx(3, "pt"),
               linewidth=ptx(1.2, "lw"))
    axins.plot(*best, "*", color="k", markersize=ptx(11, "pt"),
               markeredgecolor="white", markeredgewidth=ptx(0.5, "lw"))
    return fig, ax, axins


if __name__ == "__main__":
    apply_style()
    x = np.linspace(0, 1.0, 200)
    y = np.linspace(0, 40, 200)
    X, Y = np.meshgrid(x, y)
    cost = 10.4 * X + 0.42 * Y
    fx = np.linspace(0.0, 0.92, 12)
    fy = 36.3 * (1 - (fx / 0.92) ** 0.55)
    cost_f = 10.4 * fx + 0.42 * fy
    i = int(np.argmin(cost_f))
    fig, ax, axins = feasible_contour_zoom(
        X, Y, cost, (fx, fy), (fx[i], fy[i]),
        zoom_xlim=(0.55, 0.95), zoom_ylim=(0, 11),
        xlabel="介质A 体积分数（%）", ylabel="介质B 体积分数（%）")
    stat_box(ax, [f"最低成本 = {cost_f[i]:.2f} 元",
                  f"位于 ({fx[i]:.2f}, {fy[i]:.1f})",
                  "切点邻域平坦（见放大）"], outside="top",
             fontsize=ptx(6.5))
    ax.set_title("等值线—前沿切点给出最低成本解，放大窗排除平坦歧义",
                 fontsize=ptx(9), pad=8)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "feasible_zoom"))
    print("feasible_zoom: OK")
