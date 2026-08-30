"""灵敏度：龙卷风图（排序）+ 笛卡尔蜘蛛图（非线性形状）。

论点合同示例（tornado）：
- 结论：结果由价格比主导（±31%），几何参数影响 < 5%，结论对建模细节稳健。
- 证据链：按 |Δ输出| 排序 → 低/高端分色 → 基准竖线 → 端点直标。
禁止：雷达图做灵敏度/排名。
"""
from _common import GALLERY, PRESET
import numpy as np

from core import (text_color, ptx, ink, apply_style, new_figure, save_figure, run_qa,
                  stat_box, semantic, PALETTE)


def tornado(factors, low, high, baseline, xlabel="输出", width="onehalf"):
    """low/high: 各因素取低/高水平时的输出值。自动按影响幅度排序。"""
    span = np.abs(np.asarray(high) - np.asarray(low))
    order = np.argsort(span)
    factors = [factors[i] for i in order]
    low = np.asarray(low)[order]
    high = np.asarray(high)[order]

    fig, ax = new_figure(width, ratio=0.6)
    y = np.arange(len(factors))
    c_lo, c_hi = "#8A5A83", "#D08C3C"   # PuOr 两端低饱和
    for yi, lo, hi in zip(y, low, high):
        ax.barh(yi, lo - baseline, left=baseline, color=c_lo, height=0.55,
                alpha=0.85)
        ax.barh(yi, hi - baseline, left=baseline, color=c_hi, height=0.55,
                alpha=0.85)
        ax.annotate(f"{lo:g}", xy=(lo, yi), xytext=(-4 if lo < hi else 4, 0),
                    textcoords="offset points",
                    ha="right" if lo < hi else "left", va="center",
                    fontsize=ptx(6.5), color=text_color(c_lo))
        ax.annotate(f"{hi:g}", xy=(hi, yi), xytext=(4 if hi > lo else -4, 0),
                    textcoords="offset points",
                    ha="left" if hi > lo else "right", va="center",
                    fontsize=ptx(6.5), color=text_color(c_hi))
    ax.axvline(baseline, color="0.2", linewidth=ptx(0.9, "lw"))
    ax.text(baseline, 0.99, f" 基准 {baseline:g}", fontsize=ptx(7), color="0.2",
            va="top", ha="left", transform=ax.get_xaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(factors)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15, y=0.2)
    ax.set_xlabel(xlabel)
    ax.plot([], [], "s", color=c_lo, label="低水平")
    ax.plot([], [], "s", color=c_hi, label="高水平")
    ax.legend(loc="lower right")
    return fig, ax


def spider_cartesian(pct, outputs, names, ylabel="输出", width="single"):
    """笛卡尔蜘蛛图：x=输入相对扰动(%)，y=输出；每因素一条线，看非线性。

    不是雷达图。线过多时只画 tornado 前 4 名。
    """
    fig, ax = new_figure(width, ratio=0.75)
    styles = ["-", "--", "-.", ":"]
    for (name, out), c, ls in zip(zip(names, outputs), PALETTE, styles):
        ax.plot(pct, out, color=c, linestyle=ls, linewidth=ptx(1.3, "lw"), label=name)
    ax.axvline(0, color="0.6", linewidth=ptx(0.7, "lw"))
    ax.set_xlabel("输入相对扰动（%）")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=ptx(6.5))
    return fig, ax


if __name__ == "__main__":
    apply_style(PRESET)
    factors = ["价格比 pB/pA", "导通阈值判定", "口径容差", "圆柱长径比",
               "边界处理方式"]
    base = 9.18
    perturb = 0.30                       # 各因素的输入扰动幅度（实验设置）
    low = [6.3, 8.7, 8.9, 9.0, 9.1]
    high = [12.0, 9.8, 9.5, 9.4, 9.3]
    fig, ax = tornado(factors, low, high, base, xlabel="最低总成本（元）")
    # 硬规则：统计框数字来自计算变量（perturb 是实验设置，同样引用变量）
    p_max = max(abs(low[0] - base), abs(high[0] - base)) / base
    geo_max = max(max(abs(l - base), abs(h - base))
                  for l, h in zip(low[2:], high[2:])) / base
    stat_box(ax, [f"输入扰动 ±{perturb:.0%}（5 因素）",
                  f"价格比 → 成本 ±{p_max:.1%}",
                  f"几何参数影响 ≈ {geo_max:.1%}"],
             loc="upper left", fontsize=ptx(6.5))
    ax.set_title("结论由价格比主导，对几何建模细节稳健", fontsize=ptx(9.5))
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "tornado"))

    pct = np.linspace(-30, 30, 41)
    outs = [base * (1 + 0.0105 * pct),
            base * (1 + 0.004 * pct + 0.0002 * pct ** 2),
            base * (1 + 0.0015 * pct),
            base * (1 + 0.0008 * pct)]
    fig, ax = spider_cartesian(pct, outs, factors[:4], ylabel="最低总成本（元）")
    ax.plot([0], [base], "o", color="0.25", markersize=ptx(4.5, "pt"),
            markeredgecolor="white", markeredgewidth=ptx(0.7, "lw"), zorder=5)
    ax.annotate(f"基准 {base:g} 元", xy=(0, base), xytext=(6, -10),
                textcoords="offset points", fontsize=ptx(7), color="0.25")
    # 斜率比来自计算：正扰动端 vs 负扰动端的平均斜率
    k_pos = (outs[1][-1] - outs[1][len(pct)//2]) / (pct[-1] - 0)
    k_neg = (outs[1][len(pct)//2] - outs[1][0]) / (0 - pct[0])
    stat_box(ax, [f"扰动范围 ±{int(abs(pct[0]))}%",
                  f"阈值判定右端斜率为左端 {abs(k_pos/k_neg):.1f} 倍"],
             loc="lower right", fontsize=ptx(6.5))
    ax.set_title("价格比近线性；阈值判定右端显著超线性", fontsize=ptx(9.5))
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "spider_cartesian"))
    print("tornado: 2 figures OK")
