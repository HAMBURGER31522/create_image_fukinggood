"""渗流/相变图：列归一密度场 + 序参量曲线 + 临界线（拆面板，不用双 Y 轴）。

论点合同示例：
- 结论：最大簇占比在 φc = 0.72% 发生跃变，跃变宽度 0.06%，与导通概率 0.5 交点一致。
- 证据链：上图列归一密度场显示分布劈裂 → 下图序参量曲线 + 临界竖线贯穿两图。
对照：A 题「最大簇相变跃变」（已接近；此模板拆面板替代双轴并补涨落层）。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt

from core import (apply_style, MM, COLUMN_WIDTHS, save_figure, run_qa,
                  stat_box, cmap_for, panel_label, semantic)


def phase_density_orderparam(phi, samples, order_param, phi_c,
                             xlabel="填充率 φ",
                             y1label="最大簇占比分布", y2label="导通概率",
                             width="onehalf"):
    """samples: shape (len(phi), n_rep) 每个 φ 的最大簇占比样本。"""
    w = COLUMN_WIDTHS[width] * MM
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(w, w * 0.85), sharex=True,
        gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.12))

    # 上：列归一密度场
    ybins = np.linspace(0, 1, 60)
    H = np.zeros((len(ybins) - 1, len(phi)))
    for j in range(len(phi)):
        h, _ = np.histogram(samples[j], bins=ybins)
        H[:, j] = h / (h.max() or 1)
    pm = ax1.pcolormesh(phi, ybins[:-1], H, cmap=cmap_for("sequential2"),
                        shading="auto", rasterized=True)
    mean = samples.mean(axis=1)
    ax1.plot(phi, mean, color="#C44E52", linewidth=1.3, label="均值")
    cax = fig.add_axes([0.91, 0.45, 0.025, 0.4])
    cb = fig.colorbar(pm, cax=cax)
    # 避免竖排时"一"字旋转后形似竖线被误读为字体回退
    cb.set_label("列内相对密度", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax1.set_ylabel(y1label)
    ax1.legend(loc="upper left", fontsize=6.5)
    panel_label(ax1, "a")

    # 下：序参量
    ax2.plot(phi, order_param, "o-", color=semantic("data"), markersize=3,
             linewidth=1.2)
    ax2.axhline(0.5, color="0.6", linewidth=0.7, linestyle=":")
    ax2.set_ylabel(y2label)
    ax2.set_xlabel(xlabel)
    panel_label(ax2, "b")

    for ax in (ax1, ax2):
        ax.axvline(phi_c, color=semantic("highlight"), linewidth=1.0,
                   linestyle=(0, (5, 3)))
    ax2.annotate(f"φc = {phi_c:g}", xy=(phi_c, 0.5),
                 xytext=(8, 8), textcoords="offset points", fontsize=7.5,
                 color=semantic("highlight"), fontweight="bold")
    stat_box(ax2, [f"每点重复 n = {samples.shape[1]}",
                   f"φc = {phi_c:g}（P = 0.5 交点）"],
             loc="lower right", fontsize=6.5)
    fig.subplots_adjust(right=0.88)
    return fig, (ax1, ax2)


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(2)
    phi = np.linspace(0.5, 1.0, 26)
    nrep = 300
    phi_c = 0.72
    samples = np.zeros((len(phi), nrep))
    for j, p in enumerate(phi):
        f = 1 / (1 + np.exp(-(p - phi_c) * 30))
        low = rng.beta(1.6, 22, nrep) * 0.35
        high = np.clip(rng.normal(f, 0.05, nrep), 0, 1)
        pick = rng.uniform(0, 1, nrep) < f
        samples[j] = np.where(pick, high, low)
    order = 1 / (1 + np.exp(-(phi - phi_c) * 34))
    fig, _ = phase_density_orderparam(phi, samples, order, phi_c=phi_c)
    # φ 为体积分数（0–1 无量纲），不加 %，避免"0.72%"的单位歧义
    fig.suptitle(f"最大簇占比在 φc = {phi_c:g} 劈裂跃变，"
                 "与导通概率 0.5 交点一致",
                 fontsize=9.5, fontweight="bold", x=0.46)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "phase_transition"))
    print("phase_transition: OK")
