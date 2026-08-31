"""二维场图：圆形掩膜发散场（笛卡尔）与极坐标顺序场。

论点合同示例（掩膜场）：
- 结论：伸缩量场呈环状分层，最大伸/缩出现在某一环带，均在容差内。
  （具体半径由数据算出、经 f-string 进图题——写死在这里必然与图漂移，
   这正是 SPEC §2.1 要防的"手写常数"，只不过 docstring 逃过了 QA）
- 证据链：PuOr 对零发散色 → 等高线分层 → 极值三角+引线 → RMS 统计框。
参考：skills/photo 图 8 / 图 15（口径场图）。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from core import (current_preset, ptx, apply_style, new_figure, save_figure, run_qa,
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
    ax.plot(R * np.cos(theta), R * np.sin(theta), color="0.2", linewidth=ptx(1.0, "lw"))
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(zlabel, fontsize=ptx(8))
    cb.ax.tick_params(labelsize=ptx(7))
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
             fontsize=ptx(6.5))
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
    ax.set_theta_direction(-1)      # 罗盘方位顺时针增长，否则 90° 指向西
    # 径向刻度与网格必须显式抬到场图之上。根因是本库样式的
    # `axes.axisbelow=True`（core/style.py）把极轴 zorder 压到 0.5，被
    # zorder=1 的 pcolormesh 整块盖住——**裸 matplotlib 下同样的代码是
    # 正常渲染的**（实测 4539px vs 0px），所以这不是极坐标或 mpl 的性质，
    # 是本库样式与不透明场图的组合。它同时吃掉径向刻度**和网格圆环**，
    # 只补文字标注治不了后者，读者依然没有可对照的圈。
    ax.set_axisbelow(False)
    ax.set_rlabel_position(22.5)
    ax.tick_params(labelsize=ptx(7), pad=1)
    for lbl in ax.get_yticklabels():
        lbl.set_path_effects(_white_stroke())
    # nature 档明文禁背景网格；极坐标的径向圈是**刻度**不是装饰，
    # 但既然规范这么写，就按档走
    # 不能写 `ax.grid(False, linewidth=…, alpha=…)`——matplotlib 会警告
    # "First parameter to grid() is false, but line properties are supplied.
    # The grid will be enabled." 并**反向打开**网格。要关就单独关。
    if current_preset() == "nature":
        ax.grid(False)          # Nature 明文禁背景网格
    else:
        ax.grid(True, linewidth=ptx(0.3, "lw"), alpha=0.4)
    # 收紧极轴、放宽右边距，防止 "270°" 被 colorbar 裁切
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06)
    cb = fig.colorbar(pm, ax=ax, shrink=0.7, pad=0.12)
    cb.set_label(zlabel, fontsize=ptx(8))
    cb.ax.tick_params(labelsize=ptx(7))
    return fig, ax


def _white_stroke():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=ptx(2, "pt"), foreground="white")]


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
    # 硬规则：图题数字来自计算变量（极值所在环带半径）。
    # 此前只写了 r_max_frac 却用复数"极值"概括两端——极小值实测在
    # 1.00R 的最外圈，图上那个点就在边缘，图自己在打自己的脸。
    # 两个半径都写，都来自计算变量。
    ax.set_title(f"伸缩量场环状分层：极大值在 "
                 f"{finfo['r_max_frac']:.2f}R 环带、极小值在 "
                 f"{finfo['r_min_frac']:.2f}R 口径边缘，均在容差内",
                 fontsize=ptx(9), pad=8)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "contour_field_masked"))

    th = np.linspace(0, 2 * np.pi, 120)
    rr = np.linspace(0, 1, 60)
    TH, RR = np.meshgrid(th, rr)
    Zp = (1 - RR) * (1 + 0.35 * np.cos(3 * TH))
    fig, ax = polar_field(TH, RR, Zp, zlabel="相对密度")
    ax.set_title("方位分布呈三瓣对称", fontsize=ptx(9), pad=14)
    stat_box(ax, [f"网格 {Zp.shape[1]}×{Zp.shape[0]}（方位×径向）",
                  "三瓣对称：cos 3θ 分量主导"],
             outside="top", fontsize=ptx(6.5))
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "contour_field_polar"))
    print("contour_field: 2 figures OK")
