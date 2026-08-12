"""Parity plot（预测-实测 45° 对照）：模型验证的通用主图。

论点合同示例：
- 结论：预测与实测 R² = 0.983，93% 样本落在 ±10% 带内，无系统偏差。
- 证据链：点云贴 1:1 线 → ±band 覆盖率直标 → 最大偏差点引线。
替代：只报 R² 数字不画图，或散点无参考线（平庸）。
"""
from _common import GALLERY
import numpy as np

from core import (apply_style, new_figure, save_figure, run_qa,
                  stat_box, callout, semantic)


def parity(y_true, y_pred, band_pct=0.10, xlabel="实测值", ylabel="预测值",
           width="single"):
    """1:1 参考线 + ±band_pct 相对误差带 + 指标统计框 + 最大偏差点引线。

    返回 (fig, ax, info)，info 含 r2/rmse/mape/inside（可用于图题）。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, ax = new_figure(width, ratio=0.95)
    xs = np.array([max(lo, 1e-12), hi])
    ax.fill_between(xs, xs * (1 - band_pct), xs * (1 + band_pct),
                    color="0.88", alpha=0.7, lw=0,
                    label=f"±{band_pct:.0%} 带")
    ax.plot([lo, hi], [lo, hi], "--", color="0.35", linewidth=0.9,
            label="y = x")
    ax.plot(y_true, y_pred, "o", color=semantic("data"), markersize=4,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4,
            label="样本")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=6.5)

    resid = y_pred - y_true
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    info = dict(
        r2=1 - ss_res / ss_tot,
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        mape=float(np.mean(np.abs(resid) / np.abs(y_true))) * 100,
        inside=float(np.mean(np.abs(resid) <= band_pct * np.abs(y_true))),
    )
    stat_box(ax, [f"n = {len(y_true)}",
                  f"R² = {info['r2']:.3f}，RMSE = {info['rmse']:.3g}",
                  f"MAPE = {info['mape']:.1f}%，"
                  f"带内 {info['inside']:.0%}"],
             loc="lower right", fontsize=6.5)
    iw = int(np.argmax(np.abs(resid) / np.abs(y_true)))
    callout(ax, xy=(y_true[iw], y_pred[iw]),
            text=f"最大偏差 {resid[iw]/y_true[iw]:+.0%}",
            xytext=(0.72, 0.16), textcoords="axes fraction",
            color=semantic("bad"), rad=-0.2, mark=True)
    return fig, ax, info


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(13)
    y = rng.uniform(20, 180, 60)
    yp = y * (1 + rng.normal(0, 0.05, 60))
    yp[7] = y[7] * 1.22
    fig, ax, info = parity(y, yp, xlabel="实测径流量（m³/s）",
                           ylabel="预测径流量（m³/s）")
    ax.set_title(f"预测可信：R² = {info['r2']:.2f}，"
                 f"{info['inside']:.0%} 样本落于 ±10% 带内",
                 fontsize=9)
    save_figure(fig, str(GALLERY / "parity"))
    run_qa(fig, expect_width=("single",))
    print("parity: OK")
