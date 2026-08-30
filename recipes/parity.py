"""Parity plot（预测-实测 45° 对照）：模型验证的通用主图。

论点合同示例：
- 结论：预测与实测高度一致，多数样本落在 ±10% 带内，无系统偏差。
  （具体数字由数据算出、经 f-string 进图题——写死在这里必然与图漂移，
   这正是 SPEC 2.1 要防的手写常数，只不过 docstring 逃过了 QA）
- 证据链：点云贴 1:1 线 → ±band 覆盖率直标 → 最大偏差点引线。
替代：只报 R² 数字不画图，或散点无参考线（平庸）。
"""
from _common import GALLERY, PRESET
import numpy as np

from core import (ptx, apply_style, new_figure, save_figure, run_qa,
                  stat_box, callout, semantic)


def parity(y_true, y_pred, band=("relative", 0.10), xlabel="实测值",
           ylabel="预测值", width="single"):
    """1:1 参考线 + 误差带 + 指标统计框 + 最大偏差点引线。

    band: ("relative", 0.10) 相对带（数据须同号且远离 0）或
          ("absolute", δ) 绝对带（数据跨 0 / 含 0 时用这个）。
          兼容旧用法：直接传 float 视为相对带。
    返回 (fig, ax, info)，info 含 r2/rmse/mape/inside（可用于图题）。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if not isinstance(band, (tuple, list)):
        band = ("relative", float(band))
    mode, bval = band
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, ax = new_figure(width, ratio=0.95)
    xs = np.array([lo, hi])
    if mode == "relative":
        band_lo, band_hi = (np.minimum(xs * (1 - bval), xs * (1 + bval)),
                            np.maximum(xs * (1 - bval), xs * (1 + bval)))
        band_label = f"±{bval:.0%} 带"
        tol = bval * np.abs(y_true)
    else:
        band_lo, band_hi = xs - bval, xs + bval
        band_label = f"±{bval:g} 带"
        tol = np.full_like(y_true, bval)
    ax.fill_between(xs, band_lo, band_hi, color="0.88", alpha=0.7, lw=ptx(0, "lw"),
                    label=band_label)
    ax.plot([lo, hi], [lo, hi], "--", color="0.35", linewidth=ptx(0.9, "lw"),
            label="y = x")
    ax.plot(y_true, y_pred, "o", color=semantic("data"), markersize=4,
            markeredgecolor="white", markeredgewidth=ptx(0.6, "lw"), zorder=4,
            label="样本")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=ptx(6.5))

    resid = y_pred - y_true
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    nz = np.abs(y_true) > 1e-12          # MAPE 只对非零实测值定义
    info = dict(
        r2=float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        mape=float(np.mean(np.abs(resid[nz]) / np.abs(y_true[nz]))) * 100
             if nz.any() else np.nan,
        mae=float(np.mean(np.abs(resid))),
        inside=float(np.mean(np.abs(resid) <= tol)),
    )
    err_line = (f"MAPE = {info['mape']:.1f}%" if np.isfinite(info["mape"])
                else f"MAE = {info['mae']:.3g}")
    r2_line = (f"R² = {info['r2']:.3f}" if np.isfinite(info["r2"])
               else "R² 未定义（实测无方差）")
    stat_box(ax, [f"n = {len(y_true)}",
                  f"{r2_line}，RMSE = {info['rmse']:.3g}",
                  f"{err_line}，带内 {info['inside']:.0%}"],
             loc="lower right", fontsize=ptx(6.5))
    dev = np.abs(resid) / np.where(nz, np.abs(y_true), np.inf) \
        if mode == "relative" else np.abs(resid)
    iw = int(np.argmax(dev))
    dev_txt = (f"{resid[iw]/y_true[iw]:+.0%}" if mode == "relative"
               else f"{resid[iw]:+.3g}")
    callout(ax, xy=(y_true[iw], y_pred[iw]),
            text=f"最大偏差 {dev_txt}",
            xytext=(0.60, 0.32), textcoords="axes fraction",
            color=semantic("bad"), rad=-0.2, mark=True)
    return fig, ax, info


if __name__ == "__main__":
    apply_style(PRESET)
    rng = np.random.default_rng(13)
    y = rng.uniform(20, 180, 60)
    yp = y * (1 + rng.normal(0, 0.05, 60))
    yp[7] = y[7] * 1.22
    fig, ax, info = parity(y, yp, xlabel="实测径流量（m³/s）",
                           ylabel="预测径流量（m³/s）")
    ax.set_title(f"预测可信：R² = {info['r2']:.2f}，"
                 f"{info['inside']:.0%} 样本落于 ±10% 带内",
                 fontsize=ptx(9))
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "parity"))
    print("parity: OK")
