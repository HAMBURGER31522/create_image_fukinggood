"""门槛证明：重画华数杯 A 题「两种介质的每元渗流效率」平庸分组柱。

原图问题：渗流阈值(%)、单价(0.1 元/µm³)、总成本(元) 三个不可通约的量
共用同一 y 轴，图例式阅读，无结论。

重画论点合同：
- 结论：达到同一 90% 导通目标，介质 A 的总成本仅为介质 B 的 53%，
  尽管 A 单价是 B 的 26 倍——因为 A 的渗流阈值低 45 倍。
- archetype：小倍数点距图（每量独立轴）+ 结论注释框 + 引线标注。
- 证据链：阈值差(对数轴) → 单价差 → 成本差(最终论点)。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from core import (apply_style, MM, save_figure, run_qa,
                  stat_box, callout, panel_label, semantic)

# ── 数据（来自 A 题问题四求解结果）───────────────────────────
media = ["介质A（直圆柱）", "介质B（正球体）"]
threshold = [0.80, 36.3]      # 单独使用时的渗流阈值 (%)
price = [10.4, 0.40]          # 单价 (0.1 元/µm³)
cost = [9.62, 18.1]           # 单独达到 90% 导通的总成本 (元)

C_A, C_B = "#5B8DB8", "#C97B84"
COLORS = [C_A, C_B]

apply_style()
W = 183 * MM
fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.34))
fig.subplots_adjust(wspace=0.35, left=0.10, right=0.97, top=0.82, bottom=0.16)

panels = [
    ("渗流阈值（%）", threshold, "log", "阈值低 {:.0f} 倍", threshold[1] / threshold[0]),
    ("单价（0.1 元/µm³）", price, "log", "单价高 {:.0f} 倍", price[0] / price[1]),
    ("达到 90% 导通总成本（元）", cost, "linear", "成本仅 {:.0%}", cost[0] / cost[1]),
]

y_pos = [1, 0]
for k, (ax, (title, vals, scale, fmt, ratio)) in enumerate(zip(axes, panels)):
    ax.set_yticks(y_pos)
    ax.set_yticklabels(media if k == 0 else ["", ""])
    ax.grid(axis="y", visible=False)
    if scale == "log":
        ax.set_xscale("log")
    ax.margins(x=0.22)
    # 点距图：连接线 + 端点
    ax.plot(vals, y_pos, color="0.75", linewidth=1.4, zorder=2)
    for v, y, c in zip(vals, y_pos, COLORS):
        ax.plot([v], [y], "o", color=c, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{v:g}", xy=(v, y), xytext=(0, 9),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, fontweight="bold", color=c)
    ax.set_title(title, fontsize=9, pad=10)
    ax.set_ylim(-0.6, 1.6)
    ax.tick_params(axis="y", length=0)
    # 每面板一句证据
    ax.text(0.5, -0.32, "A " + fmt.format(ratio), transform=ax.transAxes,
            ha="center", fontsize=8, color="0.25", style="italic")
    panel_label(ax, "abc"[k], dx=-0.02 if k else -0.30)

# 最终论点：成本面板引线强调（硬规则：数字全部来自计算变量）
saving = cost[1] - cost[0]
callout(axes[2], xy=(cost[0], 1), text=f"同一目标下\nA 省 {saving:.1f} 元",
        xytext=(0.62, 0.42), textcoords="axes fraction",
        color=semantic("highlight"), rad=-0.25)
stat_box(axes[0], ["设置：MC 10⁵ 次/点", "阈值差与成本均为求解输出"],
         loc="lower right", fontsize=6)

fig.suptitle(f"同一 90% 导通目标：介质 A 总成本仅为 B 的 {cost[0]/cost[1]:.0%}"
             f"——低阈值优势({threshold[1]/threshold[0]:.0f}×)"
             f"压过高单价劣势({price[0]/price[1]:.0f}×)",
             fontsize=10, fontweight="bold", y=0.99)

out_dir = Path(__file__).resolve().parents[1] / "gallery"
out_dir.mkdir(exist_ok=True)
save_figure(fig, str(out_dir / "demo_beat_baseline"))
run_qa(fig, expect_width=("double",))
print("OK ->", out_dir / "demo_beat_baseline.png")
