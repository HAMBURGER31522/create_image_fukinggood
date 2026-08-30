"""蒙特卡洛收敛双联图：左 = 估计值+CI 带（ribbon），右 = 误差 log-log + 理论斜率。

论点合同示例：
- 结论：估计随 N 收敛，半宽按 N^(-1/2) 收缩。
  （具体数字由数据算出、经 f-string 进图题——写死在这里必然与图漂移，
   这正是 SPEC 2.1 要防的手写常数，只不过 docstring 逃过了 QA）
- 证据链：左图 CI 带收窄 + 终值直标；右图 log-log 斜率 -1/2 参考线贴合。
替代：多条误差棒折线（平庸）。
"""
from _common import GALLERY
import numpy as np

from core import (apply_style, MM, COLUMN_WIDTHS, save_figure, run_qa,
                  stat_box, end_label, ref_line, panel_label, semantic)
import matplotlib.pyplot as plt


def convergence_pair(n, est, half, true=None, ylabel="估计值"):
    w = COLUMN_WIDTHS["double"] * MM
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, w * 0.38))
    fig.subplots_adjust(wspace=0.28, bottom=0.16, top=0.80)

    c = semantic("data")
    ax1.fill_between(n, est - half, est + half, color=semantic("band"),
                     alpha=0.5, lw=0, label="95% 置信带")
    ax1.plot(n, est, color=c, linewidth=1.4, label="MC 估计")
    if true is not None:
        ref_line(ax1, true, "h", label=f"参考值 {true:g}", label_loc="left")
    ax1.set_xscale("log")
    ax1.set_xlabel("样本量 N")
    ax1.set_ylabel(ylabel)
    end_label(ax1, n[-1], est[-1], f" {est[-1]:.3f}", c)
    ax1.legend(loc="lower right")
    panel_label(ax1, "a")

    ax2.loglog(n, half, "o-", color=c, markersize=3.5, linewidth=1.2,
               label="CI 半宽")
    ref = half[0] * (n / n[0]) ** -0.5
    ax2.loglog(n, ref, "--", color="0.5", linewidth=1.0,
               label="1/√N 理论斜率")
    ax2.set_xlabel("样本量 N")
    ax2.set_ylabel("置信区间半宽")
    ax2.legend(loc="upper right")
    panel_label(ax2, "b")
    stat_box(ax2, [f"末端半宽 = {half[-1]:.4f}",
                   f"拟合斜率 = {np.polyfit(np.log(n), np.log(half), 1)[0]:.3f}"],
             loc="lower left")
    return fig, (ax1, ax2)


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(7)
    n = np.unique(np.logspace(1.3, 3.6, 24).astype(int))
    p = 0.62
    est = p + rng.normal(0, 1, len(n)) * np.sqrt(p * (1 - p) / n)
    half = 1.96 * np.sqrt(p * (1 - p) / n)
    fig, _ = convergence_pair(n, est, half, true=p, ylabel="导通概率估计")
    # 硬规则：图题数字来自计算变量
    fig.suptitle(f"导通概率估计收敛：N = {n[-1]} 时 CI 半宽 {half[-1]:.3f}，"
                 "误差按 1/√N 收缩",
                 fontsize=10, fontweight="bold", y=0.97)
    run_qa(fig, expect_width=("double",))
    save_figure(fig, str(GALLERY / "convergence_ci"))
    print("convergence_ci: OK")
