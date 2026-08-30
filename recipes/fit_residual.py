"""拟合 + 残差双联图：左 = 数据点+拟合曲线+阈值交点，右 = 残差+抽样误差带。

论点合同示例：
- 结论：Logistic 拟合优良，残差绝大多数落在 MC 抽样误差带内。
  （具体数字由数据算出、经 f-string 进图题——写死在这里必然与图漂移，
   这正是 SPEC 2.1 要防的手写常数，只不过 docstring 逃过了 QA）
- 证据链：左图曲线贴合+交点标注；右图残差围绕零线且 |r| < 抽样带。
"""
from _common import GALLERY
import numpy as np

from core import (smart_legend, apply_style, MM, COLUMN_WIDTHS, save_figure, run_qa,
                  stat_box, callout, ref_line, panel_label, semantic)
import matplotlib.pyplot as plt


def fit_residual_pair(x, y, xfit, yfit, resid, band, threshold=None,
                      x_at_threshold=None, xlabel="x", ylabel="y",
                      fit_label="拟合", data_label="观测"):
    w = COLUMN_WIDTHS["double"] * MM
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, w * 0.34))
    fig.subplots_adjust(wspace=0.26, bottom=0.17, top=0.86)

    ax1.plot(xfit, yfit, color=semantic("fit"), linewidth=1.6,
             label=fit_label, zorder=3)
    ax1.plot(x, y, "o", color=semantic("data"), markersize=4.5,
             markeredgecolor="white", markeredgewidth=0.7,
             label=data_label, zorder=4)
    if threshold is not None:
        ref_line(ax1, threshold, "h", label=f"{threshold:g}")
        if x_at_threshold is not None:
            ax1.plot([x_at_threshold], [threshold], "*",
                     color=semantic("highlight"), markersize=12,
                     markeredgecolor="white", markeredgewidth=0.6, zorder=5)
            callout(ax1, xy=(x_at_threshold, threshold),
                    text=f"交点 = {x_at_threshold:.3g}",
                    xytext=(0.72, 0.42), textcoords="axes fraction",
                    color=semantic("highlight"), rad=-0.25)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    smart_legend(ax1)
    panel_label(ax1, "a")

    ax2.fill_between(x, -band, band, color="0.85", alpha=0.8, lw=0,
                     label="MC 95% 抽样误差带")
    ax2.axhline(0, color="0.25", linewidth=0.8)
    ax2.plot(x, resid, "o", color=semantic("data"), markersize=4.5,
             markeredgecolor="white", markeredgewidth=0.7, label="拟合残差")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("残差")
    smart_legend(ax2)
    panel_label(ax2, "b")
    inside = np.mean(np.abs(resid) <= band) * 100
    stat_box(ax2, [f"{inside:.0f}% 残差落入抽样带", "无系统性弯曲"],
             outside="top")
    return fig, (ax1, ax2), inside


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(3)
    x = np.arange(0.5, 1.25, 0.05)
    logistic = lambda t, k=14, m=0.72: 1 / (1 + np.exp(-k * (t - m)))
    y = logistic(x) + rng.normal(0, 0.012, len(x))
    xfit = np.linspace(0.45, 1.25, 300)
    resid = y - logistic(x)
    band = 0.028 * np.exp(-((x - 0.78) ** 2) / 0.09) + 0.014
    # 硬规则：图例里的 R² 同样必须来自计算变量
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    fig, _, inside = fit_residual_pair(
        x, y, xfit, logistic(xfit), resid, band,
        threshold=0.9, x_at_threshold=0.72 + np.log(9) / 14,
        xlabel="介质A 体积分数（%）", ylabel="导通概率",
        fit_label=f"Logistic 拟合（R² = {r2:.3f}）", data_label="蒙特卡洛估计")
    # 硬规则：图题结论与图内统计来自同一计算变量
    desc = "全部" if inside >= 99.5 else f"{inside:.0f}%"
    fig.suptitle(f"Logistic 拟合可信：残差无结构，{desc}落在 95% 抽样误差带内",
                 fontsize=10, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.82)
    run_qa(fig, expect_width=("double",))
    save_figure(fig, str(GALLERY / "fit_residual"))
    print("fit_residual: OK")
