"""时间序列预测：历史观测 + 外推预测 + 逐渐加宽的置信扇 + 训练/测试分割。

论点合同示例：
- 结论：模型回测 MAPE 3.1%，未来 12 期预测 95% 区间宽度 ≤ ±9%。
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
    return fig, ax


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
    mape = np.mean(np.abs(y_test - y_f[:3]) / y_test) * 100
    half_w = 1.96 * sig[-1] / y_f[-1]

    fig, ax = forecast_fan(
        t, y, t_f, y_f, bands, split=t[-1], y_test=y_test,
        xlabel="年份", ylabel="需求量（万件）",
        model_label="GM(1,1) 预测")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 年份不出小数
    stat_box(ax, [f"训练 {len(t)} 期，留出回测 {len(y_test)} 期",
                  f"回测 MAPE = {mape:.1f}%",
                  f"末期 95% 半宽 ±{half_w:.0%}"],
             loc="lower right", fontsize=6.5)
    ax.set_title(f"回测 MAPE {mape:.1f}%：预测可信，"
                 f"至 {t_f[-1]} 年 95% 区间半宽 ±{half_w:.0%}",
                 fontsize=9.5)
    save_figure(fig, str(GALLERY / "timeseries_forecast"))
    run_qa(fig, expect_width=("onehalf",))
    print("timeseries_forecast: OK")
