"""分类器评估双联：ROC 曲线 + PR 曲线（C 题分类必备）。

论点合同示例：
- 结论：模型 AUC 0.93 / AP 0.88，显著优于随机基线，正例率 30% 下仍可用。
- 证据链：ROC 越过对角线基线 → PR 在低正例率下不虚高 → 统计框给 n 与阈值。
替代：只报混淆矩阵单点指标（平庸，丢失阈值全貌）。
"""
from _common import GALLERY
import numpy as np

from core import (apply_style, MM, COLUMN_WIDTHS, save_figure, run_qa,
                  stat_box, PALETTE, panel_label)
import matplotlib.pyplot as plt


def _roc_pr_points(y_true, score):
    """按分数降序扫阈值，返回 (fpr, tpr, precision, recall, auc, ap)。"""
    y_true = np.asarray(y_true, dtype=int).ravel()
    score = np.asarray(score, dtype=float).ravel()
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P, N = tp[-1], fp[-1]
    tpr = np.r_[0, tp / P]
    fpr = np.r_[0, fp / N]
    recall = tpr
    precision = np.r_[1, tp / (tp + fp)]
    auc = float(np.trapezoid(tpr, fpr))
    ap = float(np.sum(np.diff(recall) * precision[1:]))
    return fpr, tpr, precision, recall, auc, ap


def roc_pr(models, width="double"):
    """models: [(名称, y_true, score), ...]，最多建议 3 个。

    左 ROC（对角线=随机基线）、右 PR（水平线=正例率基线）。
    返回 (fig, axes, info)，info[名称] = dict(auc, ap)。
    """
    w = COLUMN_WIDTHS[width] * MM
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.42))
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, bottom=0.16,
                        top=0.86)
    info = {}
    pos_rate = None
    for k, (name, y_true, score) in enumerate(models):
        fpr, tpr, prec, rec, auc, ap = _roc_pr_points(y_true, score)
        info[name] = dict(auc=auc, ap=ap)
        pos_rate = float(np.mean(np.asarray(y_true, dtype=int)))
        c = PALETTE[k % len(PALETTE)]
        axes[0].plot(fpr, tpr, color=c, linewidth=1.3)
        # 曲线在 (1,1) 汇聚，标签放右下空白区错行叠放
        axes[0].annotate(f"{name} AUC = {auc:.2f}", xy=(0.97, 0.26 - 0.09 * k),
                         xycoords="axes fraction", ha="right", fontsize=7,
                         color=c, fontweight="bold")
        axes[1].plot(rec, prec, color=c, linewidth=1.3)
        axes[1].annotate(f"{name} AP = {ap:.2f}", xy=(0.97, 0.90 - 0.09 * k),
                         xycoords="axes fraction", ha="right", fontsize=7,
                         color=c, fontweight="bold")
    axes[0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=0.8)
    axes[0].annotate("随机基线", xy=(0.62, 0.56), fontsize=6.5, color="0.5",
                     rotation=38)
    axes[1].axhline(pos_rate, ls="--", color="0.6", linewidth=0.8)
    axes[1].annotate(f"正例率基线 {pos_rate:.0%}", xy=(0.03, pos_rate),
                     xytext=(0, 3), textcoords="offset points",
                     fontsize=6.5, color="0.5")
    axes[0].set(xlabel="假正率 FPR", ylabel="真正率 TPR",
                xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="召回率 Recall", ylabel="精确率 Precision",
                xlim=(0, 1), ylim=(0, 1.05))
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    n = len(models[0][1])
    stat_box(axes[1], [f"n = {n}，正例率 {pos_rate:.0%}",
                       "PR 对类不平衡更敏感"], loc="lower left",
             fontsize=6.5)
    fig._ff_stats = {f"{k}_{m}": v for k, d in info.items()
                     for m, v in d.items()} | dict(pos_rate=pos_rate, n=n)
    return fig, axes, info


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(21)
    n = 600
    y = (rng.uniform(size=n) < 0.3).astype(int)
    s_good = y * rng.normal(1.6, 0.9, n) + (1 - y) * rng.normal(0, 1, n)
    s_weak = y * rng.normal(0.7, 1.0, n) + (1 - y) * rng.normal(0, 1, n)
    fig, axes, info = roc_pr([("XGBoost", y, s_good),
                              ("Logistic 基线", y, s_weak)])
    # 硬规则：图题数字来自 info
    fig.suptitle(f"XGBoost AUC {info['XGBoost']['auc']:.2f} / "
                 f"AP {info['XGBoost']['ap']:.2f}，"
                 f"全阈值段优于 Logistic 基线",
                 fontsize=10, fontweight="bold", y=0.98)
    run_qa(fig, expect_width=("double",))
    save_figure(fig, str(GALLERY / "roc_pr"))
    print("roc_pr: OK")
