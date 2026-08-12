---
name: figure-forge
description: 数学建模论文期刊级作图 skill。把图从"正确的默认 matplotlib"抬到"期刊论证图"：论点合同优先、硬拒绝平庸构图（饼图/分组柱/双Y轴/jet）、统计注释框+引线标注+边缘分布等期刊构图、中文数模适配、出图后视觉自检闭环。Use when 用户要画数学建模论文的图、要求期刊级/Nature 风格图表、重画平庸图、或提到柱状图/折线图/热力图/散点图/饼图/箱线图的高级替代。
---

# figure-forge：数学建模期刊级作图

一句话：**先写论点，再选构图，证据嵌进图内，出图后必须回看。**

## 工作流（严格按序执行）

```
Task Progress:
- [ ] 1. 明确论点：这张图要证明什么（一句话）
- [ ] 2. 填论点合同（见下）
- [ ] 3. 过硬拒绝清单，命中则换构图
- [ ] 4. 从 recipes/ 复制最接近的模板改数据，禁止从空白开始
- [ ] 5. 运行出图 → run_qa 自动检查
- [ ] 6. 用 Read 工具打开 PNG 目测，对照 QA 目测清单
- [ ] 7. 不合格改到合格，交付 PNG + SVG
```

### 1. 论点合同（写代码前必填）

```
结论（一句话，将成为图题）：…
证据链（图内哪些元素支撑结论）：…
archetype（查 reference/taxonomy.md 选图决策树）：…
注释层清单：统计框内容（n/RMS/CI/占比）+ 引线标注的关键点
栏宽：single(89mm) / onehalf(136mm) / double(183mm) / cn(150mm)
```

合同不完整不出图。图题写结论（"A 成本仅为 B 的 53%"），不写描述（"A 与 B 的对比"）。

### 2. 硬拒绝清单（命中即换）

| 平庸构图 | 替代（recipes 文件） |
|---|---|
| 饼图（>3 类或需精确比较） | 有序水平条+直标 / 华夫图（composition.py） |
| 分组柱表达"差异" | 哑铃/斜率/点距/蝴蝶（comparison_rank.py） |
| 不可通约量共用 y 轴 | 小倍数拆轴（comparison_rank.py: facet_metrics） |
| 双 Y 轴（单位不同族） | 拆面板（phase_transition.py 的做法） |
| jet/rainbow/hsv 色图 | cmap_for(scene)（core/colors.py） |
| 均值柱+误差棒当分布 | 雨云图（raincloud.py） |
| 雷达图做排名/灵敏度 | 平行坐标 / tornado（composition.py / tornado.py） |
| 散点墨团（N>500） | hexbin+边缘分布（joint_marginal.py） |
| 3D 柱 / 3D 饼 | 禁止，无例外 |

### 3. 中文与数学混排硬规则

**同一字符串禁止同时含中文与 `$mathtext$`**（matplotlib 会把整串交 mathtext，
中文必然豆腐块）。混排用 Unicode：下标 ₀₁₂、上标 ⁻¹²³、希腊 φ ε η ξ σ μ Δ、
符号 ± × ≤ ≥ √ ∈。纯拉丁字符串才允许 `$...$`。run_qa 自动拦截。

### 4. 出图代码骨架

```python
import sys; sys.path.insert(0, r"<figure-forge 根目录>")
from core import apply_style, new_figure, save_figure, run_qa, \
    stat_box, callout, end_label, ref_line, panel_label, \
    marginal_grid, small_multiples, inset_zoom, \
    PALETTE, OKABE_ITO, cmap_for, semantic

apply_style()                            # 必须最先调用
fig, ax = new_figure("onehalf", ratio=0.62)
# ... 画图（从 recipes/ 抄构图）...
stat_box(ax, ["n = 692", "RMS = 5.068 cm"], loc="upper left")
callout(ax, xy=(x0, y0), text="最优点", xytext=(0.7, 0.6),
        textcoords="axes fraction", color=semantic("highlight"))
save_figure(fig, "输出路径不带扩展名")     # 自动出 png(300dpi)+svg
run_qa(fig, expect_width=("onehalf",))   # 不过直接抛错
```

### 5. QA 目测清单（Read PNG 后逐项过）

- [ ] 一句话能说出论点，图内证据支撑它（图题=结论）
- [ ] 至少一个统计注释框；关键点有引线直接标注
- [ ] 文字无重叠、无裁切、无豆腐块；图例不遮数据
- [ ] 配色低饱和、语义一致（同一对象全文同色）
- [ ] 多面板：共享色标、锁定轴限、有 (a)(b) 面板标签
- [ ] 对照基线（如有）：信息密度与论证力明显更强

## 图种索引（recipes/，每个可直接运行看 demo）

| 论证场景 | recipe | archetype |
|---|---|---|
| 类别对比/排名 | comparison_rank.py | 棒棒糖、哑铃、蝴蝶、小倍数拆轴 |
| 占比/组成 | composition.py | 有序条+直标、华夫、堆叠条、平行坐标 |
| 分布对比 | raincloud.py | 雨云、ridgeline |
| 联合分布 | joint_marginal.py | hexbin+边缘直方图+分位圆 |
| 拟合诊断 | fit_residual.py | 拟合+残差双联 |
| MC 收敛 | convergence_ci.py | CI 带 + log-log 双联 |
| 参数扫描 | scan_curve.py | log y、星标、引线框、axvspan |
| 二维场 | contour_field.py | 圆形掩膜发散场、极坐标场 |
| 约束优化 | feasible_zoom.py | 等值线+可行域+放大窗 |
| 响应面/前沿 | front_overlay.py | 热力+前沿叠加、Pareto 膝点 |
| 双参响应 | surface3d_project.py | 3D 曲面+底面投影 |
| 迭代过程 | small_multiples_frames.py | 共享色标快照 |
| 灵敏度 | tornado.py | 龙卷风、笛卡尔蜘蛛 |
| 相变/渗流 | phase_transition.py | 列归一密度场+序参量拆面板 |
| 空间网络 | network_percolation.py | 簇高亮小倍数、统一轴限 |

选图决策树与"平庸 vs 期刊级"完整对照：见 [reference/taxonomy.md](reference/taxonomy.md)。
设计规范与验收标准：见 [SPEC.md](SPEC.md)。

## 质量标准

- **合格线**：期刊图 / 模范数模论文（国赛美赛 Outstanding）水准。
- **底限（硬性）**：优于普通交付物图——信息密度、论证力、视觉工艺至少两项胜出，否则返工。
- 示例产出见 gallery/（全部由 recipes 生成，可复现）。
