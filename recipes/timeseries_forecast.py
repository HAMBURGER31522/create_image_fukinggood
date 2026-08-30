"""时间序列预测：历史观测 + 外推预测 + 逐渐加宽的置信扇 + 训练/测试分割。

论点合同示例：
- 结论：模型回测 MAPE，未来 12 期预测 95% 区间宽度 ≤ ±9%。
  （具体数字由数据算出、经 f-string 进图题——写死在这里必然与图漂移，
   这正是 SPEC 2.1 要防的手写常数，只不过 docstring 逃过了 QA）
- 证据链：历史/拟合贴合 → 分割线右侧回测点仍在扇内 → 扇形宽度直标。
替代：只画一条预测线不给区间（平庸且不可信）。

适用：灰色预测 / ARIMA / LSTM / 指数平滑等一切预测类题目的主图。
"""
from _common import GALLERY
import numpy as np

from core import (apply_style, new_figure, save_figure, run_qa,
                  stat_box, end_label, semantic)


def forecast_fan(t_hist, y_hist, t_fore, y_fore, bands, split=None,
                 y_test=None, xlabel="时间", ylabel="值",
                 model_label="模型预测", width="onehalf"):
    """bands: {置信度: (lo, hi)}，如 {0.5: (l1, h1), 0.95: (l2, h2)}。

    split: 训练/测试分割位置（x 值）；y_test: 分割后真实观测（回测点）。
    返回 (fig, ax, info)：info 含 mape/coverage/half_w_last（图题用这些，
    禁止手写）。统计框由函数内部生成，与图题同源。
    """
    fig, ax = new_figure(width, ratio=0.55)
    c_data, c_fit = semantic("data"), semantic("fit")

    # 置信扇：宽带浅、窄带深，末端直标置信度
    for level, (lo, hi) in sorted(bands.items(), reverse=True):
        alpha = 0.18 + 0.22 * level
        ax.fill_between(t_fore, lo, hi, color=semantic("band"),
                        alpha=alpha, lw=0)
        end_label(ax, t_fore[-1], hi[-1], f"{level:.0%}", "#6D93B5",
                  fontsize=6.5, fontweight="normal")
    ax.plot(t_hist, y_hist, "o-", color=c_data, markersize=3,
            linewidth=1.1, label="历史观测")
    ax.plot(t_fore, y_fore, "--", color=c_fit, linewidth=1.5,
            label=model_label)
    if split is not None:
        ax.axvline(split, color="0.4", linewidth=0.8, linestyle=(0, (4, 3)))
        # 放轴内顶部，避免与图题相压；垫白底避免与虚线相压
        ax.text(split, 0.975, " 预测起点", transform=ax.get_xaxis_transform(),
                fontsize=6.5, color="0.4", ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                          boxstyle="square,pad=0.1"))
    if y_test is not None:
        ax.plot(t_fore[: len(y_test)], y_test, "o", color="0.25",
                markersize=3.5, markerfacecolor="white",
                markeredgewidth=1.0, label="回测观测", zorder=5)
    end_label(ax, t_fore[-1], y_fore[-1], f" {y_fore[-1]:.3g}", c_fit)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=6.5)
    ax.margins(x=0.02)

    # 指标在函数内计算，图题引用 info——保证同源
    y_fore = np.asarray(y_fore, dtype=float)
    top = max(bands)
    lo_t, hi_t = (np.asarray(v) for v in bands[top])
    info = dict(n_train=len(t_hist), top_level=top,
                half_w_last=float((hi_t[-1] - lo_t[-1]) / 2 / abs(y_fore[-1]))
                if y_fore[-1] else np.nan)
    lines = [f"训练 {info['n_train']} 期"]
    if y_test is not None:
        y_test = np.asarray(y_test, dtype=float)
        k = len(y_test)
        info["mape"] = float(np.mean(
            np.abs(y_test - y_fore[:k]) / np.abs(y_test))) * 100
        info["coverage"] = float(np.mean(
            (y_test >= lo_t[:k]) & (y_test <= hi_t[:k])))
        lines += [f"留出回测 {k} 期，MAPE = {info['mape']:.1f}%",
                  f"回测点 {info['coverage']:.0%} 落在 {top:.0%} 扇内"]
    lines.append(f"末期 {top:.0%} 半宽 ±{info['half_w_last']:.0%}")
    stat_box(ax, lines, loc="lower right", fontsize=6.5)
    info = dict(info, n_hist=len(y_hist), n_future=len(y_fore),
                n_test=(len(y_test) if y_test is not None else 0))
    fig._ff_stats = info
    return fig, ax, info


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(9)
    t = np.arange(2010, 2024)
    y = 120 * 1.06 ** (t - 2010) * (1 + rng.normal(0, 0.02, len(t)))
    t_f = np.arange(2023, 2031)
    y_f = y[-1] * 1.062 ** (t_f - 2023)
    sig = 0.018 * np.sqrt(t_f - 2023 + 0.25) * y_f
    bands = {0.5: (y_f - 0.67 * sig, y_f + 0.67 * sig),
             0.95: (y_f - 1.96 * sig, y_f + 1.96 * sig)}
    # 回测：留出最后 3 期做检验
    y_test = y_f[:3] * (1 + rng.normal(0, 0.02, 3))

    fig, ax, info = forecast_fan(
        t, y, t_f, y_f, bands, split=t[-1], y_test=y_test,
        xlabel="年份", ylabel="需求量（万件）",
        model_label="GM(1,1) 预测")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 年份不出小数
    # 硬规则：图题数字来自 forecast_fan 返回的 info，与统计框同源
    ax.set_title(f"回测 MAPE {info['mape']:.1f}%：预测可信，"
                 f"至 {t_f[-1]} 年 95% 区间半宽 ±{info['half_w_last']:.0%}",
                 fontsize=9.5)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "timeseries_forecast"))
    print("timeseries_forecast: OK")
