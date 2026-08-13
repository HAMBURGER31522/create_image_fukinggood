# 图种分类体系（Taxonomy）

> 来源：agent1（grok4.6）基于 `skills/photo` 8 张参考图、华数杯 A 题交付物、期刊级作图资源目录的调研。
> 分类按**论证场景**而非数据形状；基础六图种（柱/折线/热力/散点/饼/箱线）只是平庸默认，不是选图大脑。

## 0. 选图原则（先于一切图种）

- **先写论点合同，再选构图。** 图种是论证手段，不是数据形状的查找表。
- **一张图一个论点。** 多图层只为同一结论服务（场+极值标注+统计框）。
- **硬拒绝：** 饼图/3D 饼/爆炸饼；均值柱+误差棒作为分布证据；双 Y 轴叠不可通约量；rainbow/jet；雷达图做排名；无单位/无 N/无不确定度的定量图。
- **期刊级不是换皮肤。** 跃迁来自构图：边缘分布、小倍数、底面投影、引线标注、可行域+放大。

## 选图决策短树

```
论点是什么？
├─ 分布/残差/不确定度 → 雨云 / jointplot / hexbin / 拟合+残差 / CI 带
├─ 单参数扫描/收敛 → 标注曲线 + axvspan；MC 则双联 log
├─ 两连续参数/空间场 → 等高填充 ± 掩膜 ± 前沿叠加；要可读极值则 3D+底投影
├─ 约束优化/权衡 → 等值线+可行域+放大；目标空间 Pareto
├─ 谁最敏感 → tornado；要非线性 → 笛卡尔蜘蛛；两参数交互 → 热力+临界线
├─ 占比/组成 → 有序水平条或华夫；禁止饼
├─ 类别对比 → 点距/哑铃/蝴蝶/小倍数；分组柱仅同单位
├─ 迭代/多条件同一场 → 小倍数共享色标
├─ 连通/相变 → 空间网络小倍数 + 列归一密度相变图
├─ 几何机制 → 3D 场景或 2D 投影；禁止 3D 柱/饼
├─ 分类器好坏（C 题） → ROC+PR 双联 + 混淆矩阵；禁止只报单点准确率
├─ 聚类结果（K-means/DBSCAN） → 簇着色+簇心+协方差椭圆+轮廓系数；禁止裸 tab10 散点
├─ 路径/调度解（B/C 题 TSP/VRP） → 路线图+途经序号+里程统计
├─ 多指标评价/TOPSIS/熵权 → sorted_lollipop + stacked_share（不要新几何）
└─ 3+ 指标轮廓 → 平行坐标或小倍数；禁止雷达排名
```

## 1. 分类体系与优先级

P0 = 必须有可运行模板（`recipes/`）；P1 = 应实现；P2 = 给构图指导即可。

### 1.1 分布与不确定度
| 图种 | 优先级 | recipe |
|---|---|---|
| 雨云图（半小提琴+箱+抖动点） | P0 | `raincloud.py` |
| Jointplot / hexbin + 边缘分布 | P0 | `joint_marginal.py` |
| 直方图/KDE（阈值线+统计框） | P0 | `raincloud.py` 内含 |
| 置信带折线（ribbon） | P0 | `convergence_ci.py` |
| 拟合+残差双联 | P0 | `fit_residual.py` |
| Ridgeline（>6 组分布漂移） | P1 | `raincloud.py` 内含 |
| 森林图/分位点图 | P2 | 指导见 §2.5 |

### 1.2 趋势、扫描与收敛
| 图种 | 优先级 | recipe |
|---|---|---|
| 1D 扫描曲线（log y、星标、引线框、axvspan） | P0 | `scan_curve.py` |
| 迭代小倍数（共享色标、锁轴） | P0 | `small_multiples_frames.py` |
| MC 收敛双联（估计+CI 带 / 误差 log-log） | P0 | `convergence_ci.py` |
| 时序预测扇形（历史+置信扇+回测点） | P0 | `timeseries_forecast.py` |
| 优化算法收敛（best-so-far+收敛代） | P1 | `algo_convergence.py` |
| 斜率图 / 插图放大 inset | P1 | `comparison_rank.py` / `feasible_zoom.py` |
| 面积图 / bump chart | P2 | 指导见 §2.5 |

### 1.3 二维参数空间 / 场与响应面
| 图种 | 优先级 | recipe |
|---|---|---|
| 填充等高+线等高+clabel（含圆形掩膜） | P0 | `contour_field.py` |
| 热力/等高 + 前沿/散点叠加 | P0 | `front_overlay.py` |
| 等值线 + 可行域 + 放大窗 | P0 | `feasible_zoom.py` |
| 3D 响应面 + 底面等高投影 | P0 | `surface3d_project.py` |
| 极坐标场 / quiver / 离散注释热力 | P1 | `contour_field.py` 内含极坐标 |
| 相关矩阵（下三角+强相关强调） | P1 | `annotated_heatmap.py` |

### 1.4 优化、权衡与前沿
| 图种 | 优先级 | recipe |
|---|---|---|
| 2 目标 Pareto 前沿（非支配连线+膝点） | P0 | `front_overlay.py` |
| 决策空间前沿（=可行域图） | P0 | `feasible_zoom.py` |
| 平行坐标（3+ 目标，雷达替代） | P1 | `composition.py` 内含 |
| 沿前沿 1D 切片 / 方案对照条 | P1 | `scan_curve.py` / `comparison_rank.py` |

### 1.5 灵敏度、检验与稳健性
| 图种 | 优先级 | recipe |
|---|---|---|
| 龙卷风图 tornado | P0 | `tornado.py` |
| 笛卡尔蜘蛛灵敏度（非雷达） | P1 | `tornado.py` 内含 |
| 2D 灵敏度热力+临界分区 | P1 | `front_overlay.py` 可复用 |
| 多种子稳定性点距 | P1 | `comparison_rank.py` 可复用 |
| Parity plot（预测 vs 实测 45°） | P0 | `parity.py` |
| 混淆矩阵（行归一+双标注） | P1 | `annotated_heatmap.py` |
| ROC + PR 双联（分类阈值全貌） | P0 | `roc_pr.py` |
| 聚类结果（簇心+椭圆+轮廓系数） | P0 | `cluster_scatter.py` |
| 路径/调度解（TSP/VRP 路线图） | P0 | `route_map.py` |
| 瀑布图 | P2 | 指导见 §2.5 |

### 1.6 组成与占比（饼图降级）
| 图种 | 优先级 | recipe |
|---|---|---|
| 有序水平条+直标 %（饼图默认替代） | P0 | `composition.py` |
| 华夫图（≤4 类） | P1 | `composition.py` |
| 堆叠水平条（跨组构成） | P1 | `composition.py` |
| treemap/Upset/Venn/马赛克 | P2 | 指导见 §2.5 |

### 1.7 类别对比与排名（柱状图主战场）
| 图种 | 优先级 | recipe |
|---|---|---|
| Cleveland 点距 / 棒棒糖 | P0 | `comparison_rank.py` |
| 哑铃 / 斜率图（两条件） | P0 | `comparison_rank.py`（dumbbell + slopegraph） |
| 蝴蝶图（发散条） | P0 | `comparison_rank.py` |
| 不可通约指标 → 小倍数拆轴（硬规则） | P0 | `comparison_rank.py` |
| 分组柱 | P2 | 仅同单位、≤4 组，需 `run_qa(allow=("grouped_bars",))` 显式豁免；否则拒绝 |
| 雷达图作排名 | 禁止 | 用平行坐标/小倍数 |

### 1.8 网络、结构与渗流相变
| 图种 | 优先级 | recipe |
|---|---|---|
| 空间网络小倍数（真实坐标、簇着色、统一轴限） | P0 | `network_percolation.py` |
| 相变图：列归一密度场+序参量+临界线 | P0 | `phase_transition.py` |
| 度分布小倍数 / 邻接矩阵 | P1 | `network_percolation.py` 可扩展 |
| 桑基/冲积（仅流量/转移论点） | P2 | 指导见 §2.5 |

### 1.9 三维场景与几何机制
| 图种 | 优先级 | recipe |
|---|---|---|
| 3D 曲面+底面等高（见 1.3） | P0 | `surface3d_project.py` |
| 3D 点云/网格着色（误差映射） | P0 | `small_multiples_frames.py` 可扩展 3D |
| 3D 几何场景（半透明面+光线+轨迹） | P1 | 指导+参考图 |
| 2D 三视图替代 3D | P1 | 指导 |
| 3D 柱 / 3D 饼 | 禁止 | — |

### 1.10 方法示意 / 技术路线图（论文 Figure 1）
| 图种 | 优先级 | recipe |
|---|---|---|
| 技术路线图（泳道分层+数据流/反馈线型区分） | P0 | `pipeline_diagram.py` |
| 机制矢量示意（几何/光路等） | P2 | TikZ 或手绘矢量，指导见 §2.5 |

技术路线图是每篇数模论文评委看到的第一张图，禁止 Visio/PPT 自由拼贴：
字体与正文图统一、模块框内嵌关键公式/参数、实线=数据流、虚线=反馈回路。

## 2. 平庸形态 vs 期刊级形态（速查）

| 基础图种 | 平庸默认 | 期刊级替代 |
|---|---|---|
| 柱状图 | 分组竖柱+图例+tab10 | 有序水平条直标 / 点距 / 哑铃 / 蝴蝶 / 小倍数拆轴 |
| 折线图 | 多实线+图例+线性轴 | 置信带 ribbon、log 轴、axvspan 区间、星标+同色引线框、线端直标 |
| 热力图 | imshow+jet+无等高 | contourf+contour+clabel、PuOr 对零、掩膜边界、叠加采样点/前沿、统计框 |
| 散点图 | 点云墨团 | hexbin/2D KDE、jointplot 边缘分布、分位圆、空心点 |
| 饼图 | 扇区/爆炸/3D | 有序水平条+直标（默认）、华夫（≤4 类）、堆叠条（跨组） |
| 箱线图 | 箱+须藏样本 | 雨云图（半小提琴+箱+抖动点）、ridgeline（>6 组） |
| 误差棒 | 柱顶对称棒 | CI ribbon / 雨云 / 点+CI，注明 SD/SEM/CI 与 N |

## 2.5 P2 图种构图指导（无模板，按此手写）

- **森林图/分位点图**：横向点+CI 横线，按效应量排序，`axvline` 标零效应；用 `sorted_lollipop` 骨架改造。
- **瀑布图**：`barh`/`bar` 逐项累积，正负分 semantic("good"/"bad") 色，末柱为合计并加框强调；连接虚线用 `ax.plot` 灰细线。
- **面积图/bump chart**：面积图仅当"总量+构成同时是论点"；bump chart（排名随期变化）用 `slopegraph` 扩展到多期，只强调变位者。
- **treemap/马赛克**：层级占比才用；matplotlib 无原生支持，用 `squarify` 或改用嵌套有序条。Upset/Venn 集合交集用 `upsetplot`。
- **桑基/冲积**：流量转移论点才用；matplotlib `Sankey` 类可用但工艺差，量大时改堆叠条+连接带（`fill_betweenx`）。
- **甘特/排产**：`barh` + 日期轴，任务按开始时间排序，关键路径加深色 + 里程碑竖线；每条任务右端直标工期。
- **路径/路线（TSP/VRP）**：真实坐标 `LineCollection`，途经顺序用小号数字标注，多车分 PALETTE 色，仓库/起点用星标；参考 `network_percolation.py` 的边绘制。
- **地理分布**：能不用地图就不用（散点+边界线即可）；必须用时 `geopandas` 读边界，色阶走 `cmap_for("sequential")`，禁止默认 viridis 满涂。
- **机制矢量示意**：TikZ / Inkscape 手绘，字体嵌入与正文一致；matplotlib 只负责其中的定量面板。

## 3. 实现要点备忘（P0 关键技术）

- 统计框：`core.annotate.stat_box`；引线：`core.annotate.callout`（arc3 弧线、同色 bbox）。
- 发散场用 `TwoSlopeNorm(vcenter=0)` + PuOr；顺序场 YlOrRd/YlGnBu/crest/cividis。
- jointplot 用 `core.layout.marginal_grid`（GridSpec，锁栏宽），不用 `sns.jointplot`。
- 圆形掩膜：极网格生成后 `r>R` 置 nan，或 `set_clip_path`。
- 小倍数：预计算全局 vmin/vmax，共享 colorbar，锁 xlim/ylim。
- 3D：`plot_surface(cmap='cividis', lw=0)` + `contour(zdir='z', offset=zmin)` + `ax.text2D` 统计框。
- 相变密度场：逐列直方图归一 `pcolormesh(rasterized=True)`；序参量优先拆面板，双轴需线型大差+图注声明。
- 蝴蝶图：左系列取负值画 `barh`，`axvline(0)`，条端直标绝对值。
- 龙卷风：按 |Δ输出| 排序 `barh`，基准 `axvline`，低/高端分色（PuOr 两端）。
- 空间网络：`LineCollection` 画边，贯穿簇高 zorder 亮色，其余灰；多组统一轴限。

## 4. 参考锚点

- Wilke《Fundamentals of Data Visualization》：分布、不确定度、比例、小倍数构图词典。
- Rougier《Scientific Visualization: Python + Matplotlib》：印刷级工艺。
- Ten Simple Rules for Better Figures（PLOS Comp Biol 2014）。
- Allen et al., Wellcome Open Res 4:63 (2019)：雨云图。
- Eschenbach, Interfaces 1992：tornado 管排序、spider 管形状。
- data-to-viz.com / Python Graph Gallery：只查「怎么画」，不查「画什么」。
- 视觉标尺：`skills/photo` 8 张参考图；对照平庸基线：华数杯 A 题分组柱。
