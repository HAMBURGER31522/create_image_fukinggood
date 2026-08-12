# figure-forge

数学建模论文的**期刊级作图 skill**。把图从"正确的默认 matplotlib"抬到"期刊论证图"那一档：

- **论点合同优先**：先写"这张图要证明什么"，图题就是结论句，再选构图。
- **硬拒绝平庸构图**：饼图、分组柱表达差异、不可通约量共用 y 轴、双 Y 轴、jet/rainbow、均值柱+误差棒、雷达图排名——命中即给出期刊级替代。
- **注释层是一等公民**：统计注释框（n/RMS/CI/占比）、引线标注关键点、线端直标替代图例。
- **中文数模适配**：CJK 字体自动回退、混排豆腐块拦截（Unicode 数学字符方案）、负号修复、期刊栏宽（89/136/183 mm）与印刷字号档。
- **出图后自检闭环**：`run_qa` 自动查字号/色图黑名单/豆腐块/图例遮挡/画布宽度，再由 agent 目测 PNG 对照清单。

## 效果

`gallery/` 内 26 张示例图全部由 `recipes/` 模板生成、可复现。对照样本：华数杯 A 题
交付物的平庸分组柱（三个不可通约量共用 y 轴）被重画为
`gallery/demo_beat_baseline.png`（小倍数点距图 + 拆轴 + 结论式图题）。

## 安装

```bash
git clone https://github.com/HAMBURGER31522/create_image_fukinggood.git figure-forge
cd figure-forge
pip install -r requirements.txt
```

依赖：Python ≥ 3.10，matplotlib ≥ 3.8，numpy、scipy、seaborn。
中文字体：优先 Source Han Serif SC（思源宋体），回退 SimSun/SimHei/微软雅黑，装了任意一个即可。

## 作为 Cursor/Claude skill 使用

把本仓库放到 skill 目录（如 `~/.cursor/skills/figure-forge/` 或项目
`.cursor/skills/figure-forge/`），agent 会读取 `SKILL.md` 按工作流出图。
对话里直接说：

> 用 figure-forge 画问题二的导通概率图，数据在 xxx.csv

## 手动使用（三步）

```python
import sys; sys.path.insert(0, r"<figure-forge 根目录>")
from core import apply_style, new_figure, save_figure, run_qa, stat_box, callout

apply_style()                                  # 1. 全局样式（必须最先调）
fig, ax = new_figure("onehalf", ratio=0.62)    # 2. 按期刊栏宽建图
ax.plot(...)                                   #    构图抄 recipes/ 里最接近的模板
stat_box(ax, ["n = 692", "RMS = 5.07 cm"])     #    统计注释框
save_figure(fig, "out/图名")                    # 3. 导出 png(300dpi)+svg
run_qa(fig, expect_width=("onehalf",))         #    自动 QA，不过直接抛错
```

每个 recipe 都能独立运行看 demo：

```bash
python recipes/raincloud.py        # 单个
python recipes/run_all.py          # 全部（出图到 gallery/）
```

## 目录

```
SKILL.md                 skill 入口：工作流、论点合同、硬拒绝清单、QA 清单
SPEC.md                  设计规范与验收标准
reference/taxonomy.md    图种分类体系：选图决策树、平庸 vs 期刊级对照、P0/P1/P2
core/                    工具层：style(样式) colors(配色) annotate(注释)
                         layout(布局) qa(自检)
recipes/                 15 个图种模板（每个自带合成数据 demo，可直接运行）
gallery/                 全部示例产出（可复现）
demo/beat_baseline.py    门槛证明：重画平庸分组柱
```

## 图种覆盖（按论证场景）

| 场景 | archetype |
|---|---|
| 类别对比/排名 | 有序棒棒糖、哑铃、蝴蝶图、不可通约小倍数拆轴 |
| 占比/组成（饼图替代） | 有序水平条+直标、华夫图、堆叠条、平行坐标 |
| 分布与不确定度 | 雨云图、ridgeline、CI 置信带、拟合+残差双联 |
| 联合分布 | hexbin + 边缘直方图 + 分位圆 + 计数框 |
| 趋势/扫描/收敛 | 标注扫描曲线（log/星标/引线框）、MC 收敛 log-log 双联 |
| 二维场/响应面 | 圆形掩膜发散场、极坐标场、热力+前沿叠加、等值线+可行域+放大窗 |
| 优化与前沿 | Pareto 前沿+膝点、3D 曲面+底面投影 |
| 灵敏度/检验 | 龙卷风图、笛卡尔蜘蛛 |
| 网络/相变 | 空间网络簇高亮小倍数、列归一密度相变图 |
| 迭代过程 | 共享色标小倍数快照 |

选图不查"数据形状→图型"表，查 `reference/taxonomy.md` 的**论证场景决策树**。

## 质量标准

- 合格线：期刊图 / 模范数模论文（国赛美赛 Outstanding）图的水准。
- 底限（硬性）：优于普通交付物默认图——信息密度、论证力、视觉工艺至少两项胜出。

## License

MIT
