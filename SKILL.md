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
- [ ] 4. 读数据：Excel/CSV 用 core.load_table + as_1d 转 ndarray
- [ ] 5. 从 recipes/ 复制最接近的模板改数据，禁止从空白开始
- [ ] 6. run_qa 自动检查 → 通过后 save_figure（坏图不落盘）
- [ ] 7. 用 Read 工具打开 PNG 目测，对照 QA 目测清单
- [ ] 8. 不合格改到合格，交付 PNG + SVG + 图注草稿
```

读数据示例（GBK CSV / xlsx 常见坑已处理）：

```python
from core import load_table, as_1d
df = load_table("求解结果.xlsx")           # 或 .csv（自动试 utf-8-sig/gbk）
y_true, y_pred = as_1d(df, "实测"), as_1d(df, "预测")
```

**复制 recipe 时注意**：recipes 顶部的 `from _common import GALLERY` 是 demo 专用的
输出路径工具，复制到自己的脚本时删掉这一行，改用第 4 节骨架里的
`sys.path.insert + save_figure("自己的输出路径")`。

**批量出图（赛时 15–20 张）**：逐张 Read 目测太慢时，先
`python tools/contact_sheet.py <图目录>` 把整批拼成联络表，一次 Read 扫完，
只对可疑图单独放大复检。赶工期可用 `apply_style(draft=True)` 草稿档
（降 dpi、只出 PNG），**交付前必须换回默认档重出**。

### 1. 论点合同（写代码前必填）

```
结论（一句话，将成为图题）：…
证据链（图内哪些元素支撑结论）：…
archetype（查 reference/taxonomy.md 选图决策树）：…
注释层清单：统计框内容（n/RMS/CI/占比）+ 引线标注的关键点
栏宽：single(89mm) / onehalf(136mm) / double(183mm) / cn(150mm)
```

合同不完整不出图。图题写结论（"A 成本仅为 B 的 53%"），不写描述（"A 与 B 的对比"）。

**图注草稿随图交付**（参考图的结论都走图注，正文引用离不开它）：

```
图 N　<结论句（与图题同源，可直接复用）>。<方法与读图细节：色标含义 /
带宽或 CI 定义 / n 与数据来源 / 特殊标记（星标、虚线圆）的含义>。
```

图内标题已写结论时，图注只补方法细节，不重复结论。

**统计框三段式**：参考期刊图的注释框同时含三类信息——**设置**（网格数/步长/
样本量/自由度）、**结果**（RMS/极值/占比）、**判定**（是否在容差内/是否可行）。
只写结果的框读者无法检验可信度，至少给"设置 + 结果"两层。

**图题数字硬规则：图题/统计框/标注中出现的一切数字必须用 f-string 引用计算变量，
禁止手写常数。** 手写数字会在数据或随机种子变化后与图内统计矛盾——这是事实错误级
缺陷，比任何视觉问题都严重。正确写法：

```python
rms0, rms1 = rms_list[0], rms_list[-1]        # 与帧内统计同源
fig.suptitle(f"迭代收敛：RMS {rms0:.1f} → {rms1:.2f} cm")   # ✓
fig.suptitle("迭代收敛：RMS 8.9 → 5.07 cm")                  # ✗ 手写，禁止
```

### 2. 硬拒绝清单（命中即换）

| 平庸构图 | 替代（recipes 文件） |
|---|---|
| 饼图（>3 类或需精确比较） | 有序水平条+直标 / 华夫图（composition.py） |
| 分组柱表达"差异" | 哑铃/斜率/点距/蝴蝶（comparison_rank.py）；同单位 ≤4 组确需分组柱时传 `run_qa(allow=("grouped_bars",))` 豁免 |
| 不可通约量共用 y 轴 | 小倍数拆轴（comparison_rank.py: facet_metrics） |
| 双 Y 轴（单位不同族） | 拆面板（phase_transition.py 的做法） |
| jet/rainbow/hsv 色图 | cmap_for(scene)（core/colors.py） |
| 均值柱+误差棒当分布 | 雨云图（raincloud.py） |
| 雷达图做排名/灵敏度 | 平行坐标 / tornado（composition.py / tornado.py） |
| 散点墨团（N>500） | hexbin+边缘分布（joint_marginal.py） |
| seaborn 默认全阵热力图 | 下三角/行归一+双标注（annotated_heatmap.py） |
| 预测只画一条线无区间 | 置信扇+回测点（timeseries_forecast.py) |
| Visio/PPT 风格流程图 | 泳道管线图（pipeline_diagram.py） |
| **少量离散档位连折线**（3–6 个点） | 点区间图（dot_interval.py）；折线宣称档位可插值，语义错 |
| **少量类别就拉大留白** | 加证据层而非留白：判据线 + 直标数值列 + 判定着色（dot_interval.py） |
| 3D 柱 / 3D 饼 | 禁止，无例外 |

饼图（Wedge）与双 Y 轴（twinx）由 run_qa 自动拦截；其余清单项靠此表执行。

### 3. 中文与数学混排硬规则

**同一字符串禁止同时含中文与 `$mathtext$`**（matplotlib 会把整串交 mathtext，
中文必然豆腐块）。混排用 Unicode：下标 ₀₁₂、上标 ⁻¹²³、希腊 φ ε η ξ σ μ Δ、
符号 ± × ≤ ≥ √ ∈。纯拉丁字符串才允许 `$...$`。run_qa 自动拦截。

### 3b. 两个样式档：`apply_style("cn" | "nature")`

对齐 [Nature 官方制图规范](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)。

| | `cn`（默认，中文数模交付） | `nature`（投期刊） |
|---|---|---|
| 字体 | 衬线，与论文宋体正文同族 | sans-serif（Arial/Helvetica） |
| 正文字号 | 9pt | **≤7pt**（Nature 上限） |
| 面板标签 | (a)(b)(c) 10pt 粗 | **小写 a b c，8pt 粗，无括号** |
| 背景网格 | 极淡实线 alpha 0.22 | **无**（Nature 明文禁止） |
| 数据线宽 | 1.2pt | **≤1pt**（区间 0.25–1pt） |
| 标注文字 | 允许语义色 | **一律黑色**，语义走 keyline |

两档共用：图高上限 **170mm**、`save_figure` 出 png+svg+**pdf**（Nature 主图
只收矢量，明确拒收 png/jpeg/tiff）、下同的层次与配色规则。

**字体解析只收有 Regular 字面的家族。** 思源系列常只装 Heavy(900) 一个字面，
matplotlib 会拿它当 Regular，中文全渲染成粗黑块紧挨 400 字重的拉丁数字——
看起来像渲染坏了。`apply_style` 自动跳过并打印提示，`run_qa` 复查。

### 3c. 视觉层次：饱和度必须跟着重要性走

Nature 艺术编辑的核心原则：**最重要的元素饱和度最高，背景元素用中性色**。
用 `emphasis(color, level)` 显式分三级：

```python
ref_line(ax, 0.90, label="题面判据 P ≥ 90%", level="focus")      # 论点线：最饱和
ax.plot(x, base, color=emphasis(PALETTE[0], "context"))          # 陪衬
ax.plot(x, ref,  color=emphasis(PALETTE[0], "background"))       # 背景
```

**把论点线调灰、让无关 marker 高饱和，是本 skill 最典型的失手。**
`ref_line` 的 `level` 默认 `focus` 就是为了防这个。

### 3d. 分类色上限 4 类，超了必须加冗余编码

穷举验证过：Okabe-Ito 取前 N 色在三类色盲下的最小色距为
**N=2→0.93，N=3→0.42，N=4→0.23，N=5→0.12，N=6→0.05**。
即 **≤4 类安全，5 类勉强，6 类必须靠形状/线型/直标区分**——
6 个既色盲安全又灰度可分的分类色在 sRGB 里并不存在，这不是调色技巧问题。

```python
cols, mks, lss = categorical(3)     # 颜色 + 配套 marker + 线型，一并用上
```

`semantic("good"/"bad")` 用蓝/橙不用绿/红（Nature 明文避免红绿；红绿在
deuteranopia 下色距仅 0.16、灰度差仅 0.08）。`run_qa` 会模拟三类色盲 + 灰度复查。

### 3e. 稀疏数据（3–8 个离散档位/方案）：靠证据层填满，不靠留白

数模里大量出现"四个档位""六个方案""三条策略"。**数据少不是图空的理由**——
范例里 2 行、5 个点的面板照样饱满（见 resource/ref/_INDEX.md 第二批聚合）。
五条硬默认，`recipes/dot_interval.py` 是参考实现：

1. **排序即论点**——按值排，不按输入序/字母序。排序本身在回答"谁最好"。
2. **判据参考线**——阈值/null/组均值画成线，并在两侧标方向语义
   （"未达标 ← | → 达标"），读者才能自己判。
3. **context 压灰 + 焦点上彩**——未达标压到中性色，达标上饱和色。
   注意是**对照**不是消失：中性色别淡过 0.45，否则在底纹上就没了。
4. **判定进视觉编码**——达标/不达标走实心/空心 + 颜色，不要只写在文字里。
   判定按**区间靠判据的那一侧**定（下界过线才算达标），不是按点估计。
5. **每点直标，标签走轴外数值列**——轴内没有真空位，硬塞必然压数据。
   数值列格式 `估计 [下界, 上界]`，达标行加粗。

**配色不要往低饱和调。** 期刊规范虽建议克制，但实际筛选反馈明确要求
"颜色再鲜艳点""不要黑白单调"——`cn` 档保持 `PALETTE`（Okabe-Ito 原色），
`PALETTE_MUTED` 只在需要压 context 层时局部使用，不作默认。

### 4. 出图代码骨架

```python
import sys; sys.path.insert(0, r"<figure-forge 根目录>")
from core import apply_style, new_figure, save_figure, run_qa, \
    stat_box, callout, end_label, end_labels, ref_line, panel_label, ink, \
    figure, marginal_grid, small_multiples, inset_zoom, share_colorbar, \
    PALETTE, OKABE_ITO, cmap_for, semantic, emphasis, categorical

apply_style()                            # 必须最先调用（默认 cn 档）
fig, ax = new_figure("onehalf", ratio=0.62)
# ... 画图（从 recipes/ 抄构图）...
# 框里每个数字都必须是变量，禁止手写常数
stat_box(ax, [f"n = {len(x)}", f"RMS = {rms:.3f} cm"], loc="upper left")
callout(ax, xy=(x0, y0), text="最优点", xytext=(0.7, 0.6),
        textcoords="axes fraction", color=semantic("highlight"))
run_qa(fig, expect_width=("onehalf",))   # 先 QA：不过直接抛错，坏图不落盘
save_figure(fig, "输出路径不带扩展名")     # 过了再出 png(300dpi)+svg
```

### 4b. 一张 Figure = 一次多面板装配，不是一个 chart

期刊感的最大来源是**构图**，不是注释。单轴单图只在论点确实只有一层时用；
一条论证线有多个环节时，用 `figure()` 按内容分配面板尺寸（**不是等分**）：

```python
fig, ax = figure([[("field", 1.2), ("hist", 1)],      # 主面板宽，辅面板窄
                  [("curve", 1), ("resid", 1)]],
                 width="double", height=132, row_heights=[1, 0.9])
ax["field"].contourf(...)                 # a/b/c/d 已自动编号
share_colorbar(fig, im, [ax["field"], ax["hist"]], label="导通概率 P")
```

Nature 的硬约束：整页图 **≤6 面板**、读序 **左→右、上→下**、
面板尺寸反映内容需要、**尽量压缩留白**、**不重复信息**
（同一条曲线不要在两个面板里各画一遍）。

判断该不该合并：**若几张图在论证同一个结论，它们本来就该是一张图的几个面板。**
把一条论证拆成五张各自为战的图，是"作业感"最大的来源。

### 5. QA 目测清单（Read PNG 后逐项过）

- [ ] 一句话能说出论点，图内证据支撑它（图题=结论）
- [ ] 图题中的每个数字与图内统计框/标注一致（同一变量生成）
- [ ] 至少一个统计注释框；关键点有引线直接标注
- [ ] 文字无重叠、无裁切、无豆腐块；图例不遮数据、不压直标
- [ ] **中文与拉丁字重一致**（中文明显更粗 = 字体解析到了 Heavy 字面）
- [ ] **文字对其背景对比度 ≥3:1**：语义色直接写字往往不够暗（Okabe-Ito 橙
      仅 2.25:1）。`stat_box`/`callout`/`end_label` 等已内置压暗；裸写
      `ax.annotate(color=…)` 要自己套 `ink(color)`。深底白字同理——浅色段上
      写白字会掉到 1.6:1，按底色亮度选黑/白（见 annotated_heatmap.py）
- [ ] **最饱和的元素就是论点所在**；陪衬已用 emphasis 压到 context/background
- [ ] 配色语义一致（同一对象全文同色）；分类 >4 类时有 marker/线型冗余
- [ ] **任何视觉编码都有解释**：置信带、第二组 marker、色标都能在图内查到含义
- [ ] 多面板：共享色标、锁定轴限、有面板标签、面板尺寸按内容而非等分
- [ ] **没有重复信息**：同一条曲线/同一个数没在两个面板里各出现一遍
- [ ] 对照基线（如有）：信息密度与论证力明显更强

Nature 艺术编辑的**视觉编辑五问**，落盘前逐条自问：
必要元素是哪些 / 有没有缺 / 删掉什么还能说清 / 有无重复 / 有无纯装饰。

## 图种索引（recipes/，每个可直接运行看 demo）

| 论证场景 | recipe | archetype |
|---|---|---|
| 模型框架（Figure 1） | pipeline_diagram.py | 泳道管线+数据流/反馈线型区分 |
| 类别对比/排名 | comparison_rank.py | 棒棒糖、哑铃、斜率图、蝴蝶、小倍数拆轴 |
| 占比/组成 | composition.py | 有序条+直标、华夫、堆叠条、平行坐标 |
| 分布对比 | raincloud.py | 雨云、ridgeline |
| 联合分布 | joint_marginal.py | hexbin+边缘直方图+分位圆 |
| 拟合诊断 | fit_residual.py | 拟合+残差双联 |
| 模型验证 | parity.py | 预测-实测 45° 对照+误差带 |
| 时序预测 | timeseries_forecast.py | 历史+置信扇+回测点 |
| 分类评估/相关性 | annotated_heatmap.py | 混淆矩阵、下三角相关阵 |
| 分类器阈值全貌 | roc_pr.py | ROC+PR 双联+基线+AUC/AP 直标 |
| 聚类结果 | cluster_scatter.py | 簇着色+簇心星标+2σ 椭圆+轮廓系数 |
| 路径规划/调度 | route_map.py | 多车路线+途经序号+里程统计框 |
| **离散档位 + 区间 + 判据** | **dot_interval.py** | **点区间/森林图：排序 + 判据线 + 达标着色 + 右侧数值列** |
| MC 收敛 | convergence_ci.py | CI 带 + log-log 双联 |
| 算法收敛 | algo_convergence.py | best-so-far+收敛代标注 |
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
全部函数签名速查：见 [reference/api.md](reference/api.md)。
设计规范与验收标准：见 [SPEC.md](SPEC.md)。

## 质量标准

- **合格线**：期刊图 / 模范数模论文（国赛美赛 Outstanding）水准。
- **底限（硬性）**：优于普通交付物图——信息密度、论证力、视觉工艺至少两项胜出，否则返工。
- 示例产出见 gallery/（全部由 recipes 生成，可复现）。
