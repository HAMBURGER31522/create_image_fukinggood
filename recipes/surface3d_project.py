"""3D 响应面 + 底面等高投影 + 最优点标注。

论点合同示例：
- 结论：RMS 曲面沿 (D0, ε) 单谷，最优 (0.40, 7e-4)，谷底平坦方向为 ε。
- 证据链：曲面形状 → 底面投影等高（可读等值） → 最优星标 → 参数框。
参考：skills/photo 图 12（双参数响应曲面）。
替代：默认 plot_surface 彩虹面无投影（平庸）。
"""
from _common import GALLERY, PRESET
import numpy as np
import matplotlib.pyplot as plt

from core import (ptx, apply_style, MM, COLUMN_WIDTHS, save_figure, run_qa,
                  cmap_for)


def surface_with_projection(X, Y, Z, best=None, xlabel="x", ylabel="y",
                            zlabel="z", width="onehalf", elev=25, azim=-60,
                            stat_lines=None):
    w = COLUMN_WIDTHS.get(width, width) * MM
    fig = plt.figure(figsize=(w, w * 0.85))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=0.0, right=0.86, bottom=0.06, top=0.92)
    ax.plot_surface(X, Y, Z, cmap=cmap_for("surface"), linewidth=ptx(0, "lw"),
                    antialiased=True, alpha=0.9, rstride=2, cstride=2)
    zmin = np.min(Z) - 0.35 * (np.max(Z) - np.min(Z))
    ax.contour(X, Y, Z, levels=12, zdir="z", offset=zmin,
               cmap=cmap_for("surface"), linewidths=0.8)
    if best is not None:
        bx, by = best
        # 竖线顶端取 best 点处的曲面值（约束最优 ≠ 全局谷底时不画错）
        ib = np.unravel_index(np.argmin((X - bx) ** 2 + (Y - by) ** 2),
                              Z.shape)
        bz = Z[ib]
        ax.plot([bx], [by], [zmin], "*", color="#D55E00", markersize=ptx(12, "pt"),
                markeredgecolor="white", markeredgewidth=ptx(0.5, "lw"), zorder=10)
        ax.plot([bx, bx], [by, by], [zmin, bz], ":", color="#D55E00",
                linewidth=ptx(0.9, "lw"))
    ax.set_zlim(zmin, np.max(Z))
    ax.set_xlabel(xlabel, fontsize=ptx(8), labelpad=2)
    ax.set_ylabel(ylabel, fontsize=ptx(8), labelpad=2)
    ax.set_zlabel(zlabel, fontsize=ptx(8), labelpad=8, rotation=90)
    ax.tick_params(labelsize=ptx(6.5), pad=1)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    if stat_lines:
        ax.text2D(0.02, 0.98, "\n".join(stat_lines), transform=ax.transAxes,
                  fontsize=ptx(6.5), va="top", linespacing=1.5,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                            edgecolor="0.7", alpha=0.85, linewidth=ptx(0.6, "lw")))
    return fig, ax


if __name__ == "__main__":
    apply_style(PRESET)
    d0 = np.linspace(0.34, 0.46, 60)
    eps = np.linspace(4, 10, 60)
    X, Y = np.meshgrid(d0, eps)
    Z = 4.67 + 900 * (X - 0.40) ** 2 + 0.012 * (Y - 7) ** 2
    # 硬规则：统计框数字来自计算变量
    imin = np.unravel_index(np.argmin(Z), Z.shape)
    d0_best, eps_best, z_best = X[imin], Y[imin], Z[imin]
    fig, ax = surface_with_projection(
        X, Y, Z, best=(d0_best, eps_best),
        xlabel="顶点径向偏移 D₀（m）",
        ylabel="间距容差 ε（×10⁻⁴）",
        zlabel="节点拟合 RMS（cm）",
        stat_lines=[f"网格 {Z.shape[0]}×{Z.shape[1]}，每点内层 QP+SLP",
                    f"最优 D₀ = {d0_best:.2f}，ε = {eps_best:.0f}×10⁻⁴",
                    f"RMS = {z_best:.3f} cm，谷底沿 ε 平坦"])
    ax.set_title("双参数响应曲面单谷：精度对 D₀ 敏感、对 ε 平坦",
                 fontsize=ptx(9), pad=-2)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "surface3d_project"), tight=False)
    print("surface3d_project: OK")
