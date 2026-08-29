"""二维场图：圆形掩膜发散场（笛卡尔）与极坐标顺序场。

论点合同示例（掩膜场）：
- 结论：伸缩量场呈环状分层，最大伸/缩出现在半径 0.62R 环带，均在容差内。
- 证据链：PuOr 对零发散色 → 等高线分层 → 极值三角+引线 → RMS 统计框。
参考：skills/photo 图 8 / 图 15（口径场图）。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from core import (apply_style, new_figure, save_figure, run_qa,
                  stat_box, callout, cmap_for, semantic)


def masked_diverging_field(X, Y, Z, R, xlabel="x（m）", ylabel="y（m）",
                           zlabel="伸缩量（m）", width="onehalf"):
    """圆形口径内的发散场：r>R 掩膜为 nan，色标对零。"""
    Zm = np.where(np.hypot(X, Y) <= R, Z, np.nan)
    fig, ax = new_figure(width, ratio=0.88)
    vmax = np.nanmax(np.abs(Zm))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    # 显式 levels 覆盖 ±vmax，保证色条端值 = 场内极值（标注数字不越界）
    cf = ax.contourf(X, Y, Zm, levels=np.linspace(-vmax, vmax, 25),
                     cmap=cmap_for("diverging"), norm=norm)
    ax.contour(X, Y, Zm, levels=10, colors="k", linewidths=0.25, alpha=0.4)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(theta), R * np.sin(theta), color="0.2", linewidth=1.0)
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(zlabel, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 极值引线标注
    imax = np.unravel_index(np.nanargmax(Zm), Zm.shape)
    imin = np.unravel_index(np.nanargmin(Zm), Zm.shape)
    callout(ax, xy=(X[imax], Y[imax]),
            text=f"伸长最大 {Zm[imax]:+.3f}",
            xytext=(0.87, 0.92), textcoords="axes fraction",
            color="#3D7A6B", rad=0.25, mark=True)
    callout(ax, xy=(X[imin], Y[imin]),
            text=f"回缩最大 {Zm[imin]:+.3f}",
            xytext=(0.9, 0.10), textcoords="axes fraction",
            color="#8A5A83", rad=-0.25, mark=True)
    rms = np.sqrt(np.nanmean(Zm ** 2))
    stat_box(ax, [f"节点 n = {np.sum(~np.isnan(Zm))}",
                  f"RMS = {rms:.4f}",
                  f"最大幅值 {vmax:.3f}（容差内）"], outside="top",
             fontsize=6.5)
    info = dict(r_max_frac=np.hypot(X[imax], Y[imax]) / R,
                r_min_frac=np.hypot(X[imin], Y[imin]) / R,
                rms=rms, vmax=vmax)
    return fig, ax, info


def polar_field(theta, r, Z, zlabel="幅值", width="single"):
    """极坐标顺序场：方位-半径天生极向时优于笛卡尔硬切。"""
    from core import MM, COLUMN_WIDTHS
    w = COLUMN_WIDTHS.get(width, width) * MM
    fig = plt.figure(figsize=(w, w * 0.9))
    ax = fig.add_subplot(projection="polar")
    pm = ax.pcolormesh(theta, r, Z, cmap=cmap_for("sequential"),
                       shading="auto", rasterized=True)
    ax.set_theta_zero_location("N")
    # 径向刻度：**不能用 set_yticks + get_yticklabels**。实测在极坐标轴上
    # 这条路径的标签一个像素都画不出来（有刻度与无刻度的图逐像素相同），
    # 而 get_yticklabels() 返回的又不是真正参与绘制的那批对象，所以连
    # "标签是否可见"都探不出来——gallery 里这张图一直缺整条径向刻度，
    # 读者无法把任何一圈映射到半径值。改用显式文字标注，确定会渲染。
    ax.set_rlabel_position(22.5)
    rmax = float(np.max(r))
    ax.set_yticks([])
    ax.tick_params(labelsize=7, pad=1)
    _rt = [v for v in np.linspace(0, rmax, 6)[1:-1]]
    for v in _rt:
        ax.text(np.deg2rad(22.5), v, f"{v:.2g}", fontsize=7,
                ha="center", va="center", color="0.15", zorder=6,
                path_effects=_white_stroke())
    ax.grid(linewidth=0.3, alpha=0.4)
    # 收紧极轴、放宽右边距，防止 "270°" 被 colorbar 裁切
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06)
    cb = fig.colorbar(pm, ax=ax, shrink=0.7, pad=0.12)
    cb.set_label(zlabel, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return fig, ax


def _white_stroke():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2, foreground="white")]


if __name__ == "__main__":
    apply_style()
    x = np.linspace(-160, 160, 240)
    X, Y = np.meshgrid(x, x)
    Rr = np.hypot(X, Y)
    Z = 0.22 * np.sin(Rr / 34) * np.exp(-Rr / 220) + 0.02 * np.cos(X / 40)
    fig, ax, finfo = masked_diverging_field(
        X, Y, Z, R=150,
        xlabel="口径平面横坐标 ξ₁（m）",
        ylabel="口径平面纵坐标 ξ₂（m）",
        zlabel="促动器径向伸缩量（m）")
    # 硬规则：图题数字来自计算变量（极值所在环带半径）
    ax.set_title(f"伸缩量场环状分层，极值出现在 "
                 f"{finfo['r_max_frac']:.2f}R 环带且均在容差内",
                 fontsize=9, pad=8)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "contour_field_masked"))

    th = np.linspace(0, 2 * np.pi, 120)
    rr = np.linspace(0, 1, 60)
    TH, RR = np.meshgrid(th, rr)
    Zp = (1 - RR) * (1 + 0.35 * np.cos(3 * TH))
    fig, ax = polar_field(TH, RR, Zp, zlabel="相对密度")
    ax.set_title("方位分布呈三瓣对称", fontsize=9, pad=14)
    stat_box(ax, [f"网格 {Zp.shape[1]}×{Zp.shape[0]}（方位×径向）",
                  "三瓣对称：cos 3θ 分量主导"],
             outside="top", fontsize=6.5)
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "contour_field_polar"))
    print("contour_field: 2 figures OK")
