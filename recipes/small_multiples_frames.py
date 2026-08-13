"""迭代小倍数快照：共享色标、锁定轴限、每帧统计框。

论点合同示例：
- 结论：算法 12 帧内 RMS 从 8.9 降到 5.07 并稳定，越界主索归零。
- 证据链：四帧同色标同轴限可直接比色 → 每帧 RMS 框 → 末帧为最终答案。
参考：skills/photo 图 19/20（求解过程四帧快照）。
替代：多张独立图、色标漂移（平庸，无法比演化）。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from core import (apply_style, save_figure, run_qa, small_multiples,
                  cmap_for)


def frame_snapshots(frames, titles, stats, xy=None, R=None,
                    zlabel="偏差（m）", ncols=2, width="onehalf"):
    """frames: 每帧的散点值列表（同一组点位 xy 上的标量）。共享 vmin/vmax。"""
    vmax = max(np.abs(f).max() for f in frames)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    cmap = cmap_for("diverging")
    fig, axes = small_multiples(len(frames), ncols=ncols, width=width,
                                ratio=0.92)
    for ax, f, t, s in zip(axes, frames, titles, stats):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=f, s=4, cmap=cmap, norm=norm,
                        linewidths=0)
        if R is not None:
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(R * np.cos(th), R * np.sin(th), color="0.3",
                    linewidth=0.7, linestyle="--")
        ax.set_title(t, fontsize=7.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        # 无刻度就别留 L 形残脊线，虚线口径圆即面板边界
        ax.set_frame_on(False)
        ax.text(0.03, 0.03, s, transform=ax.transAxes, fontsize=6.5,
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.75", alpha=0.85, linewidth=0.5))
    fig.subplots_adjust(right=0.86, hspace=0.18, wspace=0.06)
    cax = fig.add_axes([0.885, 0.15, 0.025, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(zlabel, fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    return fig, axes


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(9)
    n = 700
    r = 150 * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    xy = np.c_[r * np.cos(th), r * np.sin(th)]
    base = 0.2 * np.sin(r / 30)
    frame_ids = [0, 2, 7, 12]            # 抽样的迭代帧号
    stage = ["QP 初始解", "SLP-2", "逐点纠偏", "最终答案"]
    frames, rms_list, stats = [], [], []
    for k, damp in enumerate([1.0, 0.55, 0.25, 0.1]):
        f = base * damp + rng.normal(0, 0.008, n)
        frames.append(f)
        rms_list.append(np.sqrt(np.mean(f ** 2)) * 100)
        stats.append(f"RMS = {rms_list[-1]:.2f} cm")
    titles = [f"第 {i} 帧 · {s}" for i, s in zip(frame_ids, stage)]
    fig, _ = frame_snapshots(frames, titles, stats, xy=xy, R=150,
                             zlabel="径向偏差（m）")
    # 硬规则：图题中的数字必须来自计算变量，禁止手写
    fig.suptitle(f"迭代 {frame_ids[-1]} 帧收敛："
                 f"RMS {rms_list[0]:.1f} → {rms_list[-1]:.2f} cm，"
                 "环带残差逐帧消退",
                 fontsize=9.5, fontweight="bold", x=0.45)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "small_multiples_frames"))
    print("small_multiples_frames: OK")
