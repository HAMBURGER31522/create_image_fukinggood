"""技术路线图 / 模型框架图（论文 Figure 1）：分层泳道 + 模块框 + 流向箭头。

论点合同示例：
- 结论：三个子问题共享同一数据管线，问题三的反馈回路修正问题一的参数。
- 证据链：泳道分层（数据→建模→求解→检验）→ 实线数据流 → 虚线反馈。
替代：Visio/PPT 风格自由拼贴（字体、间距、箭头全不受控，平庸重灾区）。

用法：boxes 按泳道给文本（可含换行公式），flows/feedbacks 用 (泳道i,格j) 索引。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from core import apply_style, MM, COLUMN_WIDTHS, save_figure, run_qa, PALETTE

_LANE_BG = ["#F4F6F8", "#FFFFFF"]


def pipeline(lanes, flows=(), feedbacks=(), width="double", ratio=0.52,
             lane_colors=None):
    """lanes: [(泳道名, [模块文本, ...]), ...]，每条泳道一行。

    flows: [((i0,j0),(i1,j1)), ...] 实线箭头；feedbacks 同构，虚线回路。
    返回 (fig, ax, centers)，centers[i][j] 为模块中心坐标（继续加注释用）。
    """
    w = COLUMN_WIDTHS[width] * MM
    fig, ax = plt.subplots(figsize=(w, w * ratio))
    ax.set_axis_off()
    n_lane = len(lanes)
    lane_h = 1.0 / n_lane
    lane_colors = lane_colors or PALETTE
    centers = []
    for i, (lane_name, boxes) in enumerate(lanes):
        y0 = 1 - (i + 1) * lane_h
        yc = y0 + lane_h / 2
        ax.axhspan(y0, y0 + lane_h, color=_LANE_BG[i % 2], zorder=0)
        ax.text(0.012, yc, lane_name, ha="left", va="center", fontsize=8,
                fontweight="bold", color="0.35", rotation=90)
        n_box = len(boxes)
        row = []
        for j, text in enumerate(boxes):
            xc = 0.08 + (j + 0.5) * 0.92 / n_box
            c = lane_colors[i % len(lane_colors)]
            ax.text(xc, yc, text, ha="center", va="center", fontsize=7.5,
                    linespacing=1.5, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                              edgecolor=c, linewidth=1.1))
            row.append((xc, yc))
        centers.append(row)

    def _arrow(p0, p1, dashed=False, color="0.35"):
        arrow = FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=9, color=color,
            linewidth=0.9, linestyle=(0, (4, 3)) if dashed else "-",
            shrinkA=26, shrinkB=26,
            connectionstyle="arc3,rad=0.16" if dashed else "arc3,rad=0.0",
            zorder=2)
        if dashed:
            # 反馈线加白描边 + 大弧度，斜穿模块框边缘时不相压
            import matplotlib.patheffects as pe
            arrow.set_path_effects(
                [pe.withStroke(linewidth=2.6, foreground="white")])
        ax.add_patch(arrow)

    for (i0, j0), (i1, j1) in flows:
        _arrow(centers[i0][j0], centers[i1][j1])
    for (i0, j0), (i1, j1) in feedbacks:
        _arrow(centers[i0][j0], centers[i1][j1], dashed=True,
               color="#C44E52")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if feedbacks:
        ax.plot([], [], color="0.35", linewidth=0.9, label="数据流")
        ax.plot([], [], color="#C44E52", linestyle=(0, (4, 3)),
                linewidth=0.9, label="反馈/迭代")
        # 放首泳道左侧空白，避免压住任何模块框
        ax.legend(loc="upper left", bbox_to_anchor=(0.035, 0.99),
                  fontsize=6.5, frameon=True)
    return fig, ax, centers


if __name__ == "__main__":
    apply_style()
    lanes = [
        ("数据层", ["附件数据清洗\n（缺失/异常处理）", "几何建模\n抛物面基准态"]),
        ("模型层", ["问题一：单参数扫描\nmin RMS(D₀)",
                    "问题二：QP + SLP\n促动器协同优化",
                    "问题三：光路 MC\n接收比估计"]),
        ("求解层", ["两阶段精搜\n粗 0.02 → 细 0.002",
                    "凸松弛 + 逐点纠偏\n12 帧迭代收敛",
                    "10⁵ 光线采样\n分层抽样降方差"]),
        ("检验层", ["残差无结构性\n（拟合-残差双联）",
                    "约束满足率 100%\n间距/行程双校验",
                    "灵敏度：结论对\n参数扰动稳健"]),
    ]
    flows = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 1), (1, 2)),
             ((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (2, 2)),
             ((2, 0), (3, 0)), ((2, 1), (3, 1)), ((2, 2), (3, 2))]
    feedbacks = [((3, 2), (1, 0))]
    fig, ax, _ = pipeline(lanes, flows, feedbacks)
    fig.suptitle("技术路线：四层管线，灵敏度检验反馈修正问题一参数",
                 fontsize=10, fontweight="bold", y=0.99)
    save_figure(fig, str(GALLERY / "pipeline_diagram"))
    run_qa(fig, expect_width=("double",))
    print("pipeline_diagram: OK")
