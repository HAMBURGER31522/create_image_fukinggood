"""点区间图（forest / dot-and-interval）：少量离散档位 + 区间 + 判据。

论点合同示例：
- 结论：四档中仅 φ=1.00% 达标（下界 0.9889 ≥ 0.90），其余三档下界均未过线。
- 证据链：每档一行点+Wilson 区间 → 判据竖线 → 达标实心/未达标空心 →
          右侧数值列给精确值 → 统计框给设置与判定。
- archetype：dot_interval（离散档位 + 不确定度 + 判据）
- 替代（平庸）：四点折线（折线宣称档位可插值，语义错）、均值柱+误差棒。

构图本体在 `core.dot_interval(ax, ...)`，本文件只演示怎么用。
参考范例与五条硬默认见 resource/ref/_INDEX.md 第二批聚合。
"""
from _common import GALLERY
import numpy as np

from core import (ptx, apply_style, new_figure, save_figure, run_qa, stat_box,
                  dot_interval)


def wilson(k, n, z=1.96):
    """Wilson 得分区间：小样本/极端比例下比正态近似可靠得多。

    返回 (点估计, 下界, 上界)。数模里报"某档达标概率"必须给区间——
    只给点估计的比例是不可检验的断言。
    """
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, c - h, c + h


if __name__ == "__main__":
    apply_style()
    # 华数杯 A 题问题二的四档：M = 2000 次仿真，判据 P ≥ 0.90
    phi, M = [0.50, 0.60, 0.70, 1.00], 2000
    k = [162, 432, 994, 1987]                 # 各档导通次数
    NA = [354, 425, 496, 707]                 # 各档介质A 根数（独立于 P）
    p, lo, hi = wilson(k, M)
    thr = 0.90

    fig, ax = new_figure("onehalf", ratio=0.16 + 0.062 * len(phi))
    # 右侧不必手动留位：value_col=True 会自己收缩本轴给数值列让位
    fig.subplots_adjust(left=0.17, top=0.88, bottom=0.20)
    ok = dot_interval(ax, [f"φ = {v:.2f}%" for v in phi], p, lo, hi,
                      threshold=thr, thr_label=f"题面判据 P ≥ {thr:.0%}",
                      # 点面积编码**独立于 x 轴**的第二个量：拿 k 编码
                      # 会和 x（p = k/M）完全共线，属于重复信息
                      sizes=NA)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("微结构导通概率 P（Wilson 95% 区间）")

    n_ok = int(ok.sum())
    fig.suptitle(f"四档中仅 {n_ok} 档达标：φ = {phi[-1]:.2f}% 的下界 "
                 f"{lo[-1]:.4f} ≥ {thr:.2f}",
                 fontsize=ptx(9.5), fontweight="bold")
    stat_box(ax, [f"每档 M = {M} 次仿真；点面积随 N_A 线性递增"
                  f"（{min(NA)}–{max(NA)} 根）",
                  f"区间半宽 {np.min((hi - lo) / 2):.4f}–"
                  f"{np.max((hi - lo) / 2):.4f}",
                  f"达标 {n_ok}/{len(phi)} 档（按下界判）"],
             outside="bottom")
    fig._ff_stats = {"p": list(p), "lo": list(lo), "hi": list(hi),
                     "phi": phi, "M": M, "k": k, "thr": thr, "n_ok": n_ok,
                     "half": list((hi - lo) / 2)}
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "dot_interval"))
    print("dot_interval: OK")
