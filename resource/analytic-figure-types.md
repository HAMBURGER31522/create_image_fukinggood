# Analytic Figure 种类分览（按范文识别）

> 第二个子标题交付物。  
> 分类对齐 `reference/taxonomy.md` 的**论证场景**，不是 Excel 默认六图。  
> 每类下列出：**定义 / 在数模里何时用 / 已识别范文链接 / 识别依据 / figure-forge recipe**。  
> 本地完整 PDF 库（2026-08-17 起）：`F:\mathematical modeling\papers`（已从 skill `resource\papers` 迁出）。

证据标记：

- 🧪 = 本轮下载 PDF 或打开 OA 页，从图题/正文识别  
- 📚 = 方法文明确推荐该图种  
- 入口 = 稳定搜索入口，需点进单篇再确认

---

## 总览：范文里真实出现的 analytic figure 族

| # | 图种类 | 竞赛范文频率 | 期刊方法文态度 | figure-forge |
|---|---|---|---|---|
| 1 | 技术路线 / 流程图 | 极高（几乎每篇 Fig.1） | 中性 | `pipeline_diagram.py` |
| 2 | 散点 + 拟合曲线 | 高（A/C） | 强推（替代裸均值） | `fit_residual.py` / `parity.py` |
| 3 | 时序 / 过程曲线（含 loss、收敛） | 极高 | 要加 CI/标注 | `timeseries_forecast.py` / `algo_convergence.py` / `scan_curve.py` |
| 4 | 热力图 / 矩阵色块 | 高（B/C 空间与相关） | 常用 | `annotated_heatmap.py` / `front_overlay.py` |
| 5 | 空间地图 / 栅格 / 概率场 | 高（B/C/E） | 常用 | `route_map.py` / `contour_field.py` |
| 6 | 直方图 / 经验分布 | 中 | 强推 | `raincloud.py`（hist/KDE） |
| 7 | 箱线 / 小提琴 / **雨云** | 竞赛少、期刊多 | **最强推荐** | `raincloud.py` |
| 8 | 混淆矩阵 | C 题 ML 常见 | 标准 | `annotated_heatmap.py` |
| 9 | ACF / PACF / 残差诊断 | C 题时序 | 标准 | `fit_residual.py` |
| 10 | 等高 / 地形 / 场 | A/B 机理与空间 | 常用 | `contour_field.py` |
| 11 | 路线 / 轨迹 / 路径 | B 交通/巡游 | 常用 | `route_map.py` |
| 12 | 网络 / 结构示意（含 CNN） | C/D | 中性 | `network_percolation.py` / `pipeline_diagram.py` |
| 13 | 敏感度曲线 / 参数扫描 | 几乎每篇末 | 常用 | `tornado.py` / `scan_curve.py` |
| 14 | 策略/组别对比条或折线 | 高 | 警告均值柱 | `comparison_rank.py` |
| 15 | 3D 示意 / 响应面 | 中（易滥用） | 慎用 | `surface3d_project.py` |
| 16 | 小倍数（small multiples） | 高（多情景） | 强推 | `small_multiples_frames.py` |

**竞赛 vs 期刊落差（读范文时要有数）：**

- O 奖里**大量**仍是：均值柱、无 CI 折线、jet 热力、PPT 风流程图。  
- 期刊方法文要你升级到：raincloud / univariate scatter / 边缘分布 / 注释层。  
- 所以链接集里**同时**放 O 奖（题型构图）和 PLOS/Wellcome（统计构图）。

---

## 1. 技术路线图 / 流程图 / 框架图

**是什么：** 泳道或模块箭头，交代数据→模型→检验全流程。  
**论点：** “我们做了什么、模块如何耦合”，不是数据证据。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2100454.pdf 🧪 | Fig.2 “Framework of the GAME Model” |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2101951.pdf 🧪 | Fig.2 “Overview of this work” |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/B/2102199.pdf 🧪 | Fig.3 “Flow Chart of Our Work” |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/A/2200289.pdf 🧪 | Fig.1 “Structure of Our Work”；Fig.8 SA flow |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/C/2200401.pdf 🧪 | 策略/网格流程示意 |
| https://github.com/linggm3/2023_CUMCM_National-First-Prize 🧪 | 国赛一等全文 + figs |

**figure-forge：** `recipes/pipeline_diagram.py`  
**升级要点：** 实线数据流 / 虚线反馈；框内嵌关键符号；禁 Visio 彩虹。

---

## 2. 散点图 + 拟合曲线（含对比实验点）

**是什么：** 原始点云 + 模型曲线（有时再加残差或对照实验曲线）。  
**论点：** “模型抓住了关系 / 偏差可控”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2101951.pdf 🧪 | Fig.4a/b 扩展率–分解率曲线 vs 实验；Fig.5 原点+拟合 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2100454.pdf 🧪 | Fig.3 log(hyphal extension) 关系曲线 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2014/25142.pdf 🧪（本地 `mcm2014_25142`） | 速度–安全间距关系 |
| https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128 📚🧪 | 系统论证 **univariate / paired scatter** 优于 bar |

**figure-forge：** `fit_residual.py`、`parity.py`、`joint_marginal.py`  
**升级要点：** 残差面板、N、RMSE 进统计框；大 N 改 hexbin。

---

## 3. 时序曲线 / 动态过程 / 训练 loss / 优化收敛

**是什么：** 横轴时间、迭代、里程；纵轴状态量或目标值。  
**论点：** “系统如何演化 / 算法是否收敛 / 预测是否贴合”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2100454.pdf 🧪 | Fig.5 真菌密度随时间；Fig.6–9 效率/密度动态 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2101951.pdf 🧪 | Fig.6–11 生物量与分解速度多情景 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/C/2101166.pdf 🧪 | Fig.7 auto-encoder **loss curve**；准确率曲线 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/A/2200289.pdf 🧪 | Fig.9–12 功率分配；Fig.11 **optimization process curve** |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/C/2200401.pdf 🧪 | Fig.5 ARIMA 预测序列；多策略资金曲线 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2014/25142.pdf 🧪 | 多规则下平均速度/危险指数随参数变化 |

**figure-forge：** `timeseries_forecast.py`、`algo_convergence.py`、`convergence_ci.py`、`scan_curve.py`  
**升级要点：** 预测必须带置信扇；多序列用线端直标而非巨大图例。

---

## 4. 热力图 / 色块矩阵（空间栅格 · 相关 · 参数网格）

**是什么：** 二维矩阵映射为颜色。  
**论点：** “哪里高/低、哪对变量强相关、哪组参数更优”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/B/2102199.pdf 🧪 | Fig.7 “**Heat map after Rasterization**”；火点热力 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/B/2104673.pdf 🧪 | Fig.9 “Fire severity **matrix**”（色表示严重度） |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/C/2200401.pdf 🧪 | Fig.6 “**Heat map**, ACF and PACF figure” |
| PLOS Comp Biol heatmap 检索入口 | https://journals.plos.org/ploscompbiol/search?q=heatmap 入口 |
| Nature Communications heatmap 检索 | https://www.nature.com/ncomms/search?q=heatmap 入口 |

**figure-forge：** `annotated_heatmap.py`、`front_overlay.py`、`contour_field.py`  
**升级要点：** 禁 jet；相关阵用下三角+数值；连续场优先 contourf+clabel。

---

## 5. 空间分布图 / 概率地图 / 传播小倍数

**是什么：** 地理底图或规则网格上的点、斑块、概率着色。  
**论点：** “事件在哪里、下一步往哪扩”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/C/2101587.pdf 🧪 | Fig.1 hive 分布；Fig.4–5 巢穴现位/次年可能位置 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/C/2101166.pdf 🧪 | Fig.3a–f 迁移与活动范围小倍数 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/B/2102199.pdf 🧪 | Fig.1/4 澳大利亚火情与 Victoria 筛选地图 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/B/2104673.pdf 🧪 | Fig.3 东南澳地形；Fig.8 CA 火蔓延场 |

**figure-forge：** `route_map.py`、`contour_field.py`、`network_percolation.py`  
**升级要点：** 统一投影/轴限；小倍数共享色标。

---

## 6. 直方图 / 经验分布

**是什么：** 单变量频次或密度。  
**论点：** “分布形状、是否偏态、组间是否可分”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/C/2101166.pdf 🧪 | Fig.6 “**Frequency histograms** of the comment length” |
| https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128 📚🧪 | 明确把 histogram 列为应替代 bar 的分布图 |

**figure-forge：** `raincloud.py`（hist/KDE 分支）  
**升级要点：** 加均值/分位竖线与 N；多组用 ridgeline 或 raincloud。

---

## 7. 箱线图 · 小提琴图 · 雨云图（期刊核心）

**是什么：**  
- box：五数概括  
- violin：核密度镜像  
- **raincloud：半小提琴 + 箱 + 原始点雨丝**（分布最完整）

**论点：** “组间分布差异”，不是“均值有个误差棒”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://wellcomeopenresearch.org/articles/4-63 📚 | **Raincloud 原始定义与示例**（Allen et al.） |
| https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128 📚🧪 | 正文反复对比 bar vs **scatter / box / histogram**；配对数据用连线 scatter |
| PLOS / Nature violin 检索 | https://journals.plos.org/ploscompbiol/search?q=violin+plot · https://www.nature.com/srep/search?q=violin+plot 入口 |
| Scientific Reports raincloud 检索 | https://www.nature.com/srep/search?q=raincloud 入口 |

**竞赛观察：** 本轮抽样的 MCM O 奖 PDF **几乎不出现**标准 violin/raincloud；这是 figure-forge 相对 O 奖的**构图超额利润区**。  
**figure-forge：** `recipes/raincloud.py`（P0）

**读雨云范文时看什么：**

1. 半琴是否与箱对齐同一 y 标尺  
2. 原始点 alpha / jitter 是否仍能看见重叠  
3. 是否标 n、效应量，而不是只做 *p* 星号

---

## 8. 混淆矩阵（分类结果）

**是什么：** 真值×预测的计数或行归一热力。  
**论点：** “分类器错在哪一类，不是只报 accuracy”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/C/2101587.pdf 🧪 | **Fig.8 Confusion Matrix** |

**figure-forge：** `annotated_heatmap.py`（行归一 + 双标注）  
**升级要点：** 同步给 ROC+PR（`roc_pr.py`），禁止只丢一个准确率点。

---

## 9. ACF / PACF / 残差诊断面板

**是什么：** 自相关、偏自相关、残差-拟合。  
**论点：** “时序模型定阶合理 / 残差近似白噪声”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/C/2200401.pdf 🧪 | Fig.6 同图含 Heat map 与 **ACF and PACF** |

**figure-forge：** `fit_residual.py`、`timeseries_forecast.py`  
**升级要点：** 残差直方图或 Q-Q 可作副面板。

---

## 10. 等高线 / 地形 / 连续场

**是什么：** contour / contourf / 地形晕渲。  
**论点：** “场的梯度、峰谷、可行界”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/B/2104673.pdf 🧪 | Fig.3 “Topographic map and brief **contour**” |
| https://github.com/linggm3/2023_CUMCM_National-First-Prize 🧪 | 定日镜场/光学能量密度类场图（仓库 figs + 全文） |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/A/2200289.pdf 🧪 | Fig.2 赛道地形 topographic map |

**figure-forge：** `contour_field.py`、`surface3d_project.py`、`feasible_zoom.py`  
**升级要点：** 填充+线+clabel；极值 callout；3D 必须带底投影。

---

## 11. 路线图 / 轨迹 / 巡游路径

**是什么：** 真实或抽象坐标上的折线路径、途经点。  
**论点：** “调度/巡游/赛段策略长什么样”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/A/2200289.pdf 🧪 | Fig.13 “**Route map** of self-designed course” |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/B/2102199.pdf 🧪 | 侦察/悬停路线与 EOC 布局叙述 + 图 |

**figure-forge：** `route_map.py`  
**升级要点：** 途经序号、仓库星标、多车分色、里程统计框。

---

## 12. 网络结构 / 算法结构示意

**是什么：** 节点-边，或 CNN/模块框图（偏示意）。  
**论点：** “实体如何连接 / 模型层级是什么”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/C/2101166.pdf 🧪 | Fig.4 CNN structure；Fig.5 Auto-Encoder structure |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/C/2101587.pdf 🧪 | Fig.6 CNN |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2013/12218.pdf 🧪（本地 mcm2013） | multi-graph / supergraph 关系 |

**figure-forge：** `network_percolation.py`、`pipeline_diagram.py`  
**定量连通/相变**才上渗流小倍数；纯 CNN 框图保持克制配色。

---

## 13. 灵敏度 / 参数扫描 / （可升级为）Tornado

**是什么：** 参数扰动 → 输出变化；竞赛里多为多条 scan 曲线，期刊决策分析常用 tornado。  
**论点：** “谁最敏感、稳健区间在哪”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2100454.pdf 🧪 | §Sensitivity Analysis；多参数曲线 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/B/2104673.pdf 🧪 | §5 Sensitivity；临界火概率等 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/A/2200289.pdf 🧪 | 风速风向等敏感度 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/C/2200401.pdf 🧪 | 网格参数/策略区间对比 |

**figure-forge：** `tornado.py`、`scan_curve.py`  
**升级要点：** 单参数扫描用星标+axvspan；多因素一次比较用 tornado，别堆 12 条交叉折线。

---

## 14. 类别 / 策略对比（条、点距、小倍数）

**是什么：** 多方案指标对照。  
**论点：** “谁更好、好多少”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2022/C/2200401.pdf 🧪 | Fig.7 “Comparison of 8 different strategies” |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2100454.pdf 🧪 | Fig.6 Relative Decomposition Efficiency；Fig.10 Ability Value |
| https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128 📚 | 反面教材：mean±SE **bar** 掩盖分布 |

**figure-forge：** `comparison_rank.py`（点距/哑铃/蝴蝶）、`composition.py`  
**升级要点：** 有序水平条或 Cleveland 点距；不可通约指标拆小倍数。

---

## 15. 小倍数（Small Multiples）

**是什么：** 同一几何、多面板，共享色标/坐标。  
**论点：** “情景/时间/物种差异一眼可比”。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/C/2101166.pdf 🧪 | Fig.3a–g 传播阶段 + logistic 同版 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2101951.pdf 🧪 | Fig.9–10 竞争/环境波动多面板 |
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2014/25142.pdf 🧪 | 多车道密度 Fig.7–10 成组 |

**figure-forge：** `small_multiples_frames.py`  
**升级要点：** 预计算全局 vmin/vmax；锁轴；总标题写结论。

---

## 16. 3D 示意 / 曲面（慎用族）

**是什么：** 3D 立方体示意、响应面。  
**论点：** 仅当几何关系 2D 说不清。

### 范文链接

| 链接 | 识别依据 |
|---|---|
| https://raw.githubusercontent.com/Jackksonns/MCM-ICM-Outstanding-Papers/main/2021/A/2100454.pdf 🧪 | Fig.4 竞争示意（文中 light green **cube**） |
| https://github.com/linggm3/2023_CUMCM_National-First-Prize 🧪 | 定日镜几何/场（易出现 3D 或伪 3D） |
| https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833 📚 | Rule：默认与 3D 装饰常误导 |

**figure-forge：** `surface3d_project.py`  
**硬规则：** 禁止 3D 柱/饼；曲面必须底面等高投影。

---

## 17. 竞赛里少见、但期刊/figure-forge 应主动补的图种

这些在本轮 O 奖抽样中**少见或未见**，却是期刊论证利器——链接给**范式文/检索入口**，供你“抬构图”：

| 图种 | 推荐链接 | recipe |
|---|---|---|
| Raincloud | https://wellcomeopenresearch.org/articles/4-63 | `raincloud.py` |
| Joint + 边缘分布 / hexbin | 方法：Weissgerber；检索 `jointplot` / `marginal` on PLOS | `joint_marginal.py` |
| ROC + PR 双联 | C 题应标配；O 奖常只给 accuracy——用 recipe 补 | `roc_pr.py` |
| Pareto 前沿 | 多目标 B/E；检索竞赛 PDF 内 “Pareto” | `front_overlay.py` |
| Tornado 敏感度 | 决策分析标准图 | `tornado.py` |
| 平行坐标 | 3+ 指标；禁雷达排名 | `composition.py` |
| 华夫 / 有序条（替饼图） | Rougier 亦警示 pie | `composition.py` |

---

## 18. 按「你要证明的话」反查图种（速查）

| 你想说的话 | 用的图种 | 先打开哪条范文 |
|---|---|---|
| 分布不同，不是均值不同 | raincloud / scatter / box | Weissgerber · Raincloud 原文 |
| 模型拟合住了机理曲线 | 散点+拟合+残差 | 2021A 2101951 |
| 火灾/虫害在空间上聚在哪 | 热力栅格 / 概率地图 | 2021B 2102199 · 2021C 2101587 |
| 分类器是否偏科 | 混淆矩阵 + ROC/PR | 2021C 2101587 |
| 时序模型阶数是否靠谱 | ACF/PACF | 2022C 2200401 |
| 八个策略谁赢 | 点距/有序条（不要饼） | 2022C 2200401 Fig.7 |
| 参数谁最敏感 | tornado 或扫描曲线 | 2021B 2104673 · recipe tornado |
| 车怎么跑 / 无人机怎么巡 | route map | 2022A 2200289 |
| 评委 30 秒看懂结构 | pipeline | 任意 O 奖 Fig.1 + `pipeline_diagram.py` |

---

## 19. 识别方法备忘（以后你自己扩库）

```text
1. 从 Jackksonns 按 YEAR/PROBLEM 拉 PDF
2. pdftotext -layout paper.pdf paper.txt
3. 抓 ^(Figure|Fig\.|图)\s*\d+
4. 关键字：heatmap|violin|box|scatter|contour|ROC|confusion|ACF|route|histogram|Pareto
5. 扫描件（text_len≈0）→ 跳过或人工目测 figs/
6. 把结果追加到本文件对应 §，并更新 samples-verified.json
```

---

## 20. 和 taxonomy 的映射（避免两套语言）

| 本文图种 | taxonomy.md 章节 |
|---|---|
| 雨云/箱/小提琴/直方图 | §1.1 分布与不确定度 |
| 时序/收敛/扫描 | §1.2 趋势、扫描与收敛 |
| 热力/等高/场 | §1.3 二维参数空间 |
| Pareto/可行域 | §1.4 优化与前沿 |
| Tornado/ROC/混淆/聚类 | §1.5 灵敏度与检验 |
| 有序条/华夫 | §1.6 组成 |
| 点距/哑铃/蝴蝶 | §1.7 类别对比 |
| 空间网络 | §1.8 网络与相变 |
| 3D 曲面投影 | §1.9 三维 |
| 技术路线 | §1.10 方法示意 |
