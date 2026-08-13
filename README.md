# figure-forge

数学建模论文的**期刊级作图 skill**。把图从"正确的默认 matplotlib"抬到"期刊论证图"那一档：

- **论点合同优先**：先写"这张图要证明什么"，图题就是结论句，再选构图。
- **硬拒绝平庸构图**：饼图、分组柱表达差异、不可通约量共用 y 轴、双 Y 轴、jet/rainbow、均值柱+误差棒、雷达图排名——命中即给出期刊级替代。
- **注释层是一等公民**：统计注释框（n/RMS/CI/占比）、引线标注关键点、线端直标替代图例。
- **中文数模适配**：CJK 字体自动回退、混排豆腐块拦截（Unicode 数学字符方案）、负号修复、期刊栏宽（89/136/183 mm）与印刷字号档。
- **出图后自检闭环**：`run_qa` 自动查字号/色图黑名单/饼图与双Y轴等硬拒绝构图/注释层存在性/豆腐块/图例遮挡/画布宽度，再由 agent 目测 PNG 对照清单。
- **72h 赛时适配**：`tools/contact_sheet.py` 把整批图拼成联络表一次目测；`apply_style(draft=True)` 草稿档加速迭代。

## 效果

`gallery/` 内 30+ 张示例图全部由 `recipes/` 模板生成、可复现。对照样本：华数杯 A 题
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
stat_box(ax, [f"n = {len(x)}",                 #    统计注释框：数字必须是变量
              f"RMS = {rms:.2f} cm"])
run_qa(fig, expect_width=("onehalf",))         # 3. 先 QA：不过直接抛错
save_figure(fig, "out/图名")                    # 4. 过了再导出 png(300dpi)+svg
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
reference/api.md         全部函数签名/返回值速查
core/                    工具层：style(样式) colors(配色) annotate(注释)
                         layout(布局) qa(自检)
recipes/                 20 个图种模板（每个自带合成数据 demo，可直接运行）
tools/contact_sheet.py   联络表：批量图拼网格，一次目测
gallery/                 全部示例产出（可复现）
demo/beat_baseline.py    门槛证明：重画平庸分组柱
```

## 图种覆盖（按论证场景）

技术路线图（Figure 1）、类别对比（棒棒糖/哑铃/斜率/蝴蝶/拆轴小倍数）、占比组成
（有序条/华夫/堆叠/平行坐标）、分布（雨云/ridgeline）、联合分布（hexbin+边缘）、
模型验证（parity/拟合残差/混淆矩阵/相关阵）、时序预测置信扇、MC 与算法收敛、
参数扫描、二维场与响应面、约束优化与 Pareto、3D 曲面投影、灵敏度（tornado/蜘蛛）、
网络渗流与相变、迭代小倍数。

完整索引与选图决策树见 [SKILL.md](SKILL.md) 与
[reference/taxonomy.md](reference/taxonomy.md)（唯一权威表，本节只做导航）。

## 质量标准

- 合格线：期刊图 / 模范数模论文（国赛美赛 Outstanding）图的水准。
- 底限（硬性）：优于普通交付物默认图——信息密度、论证力、视觉工艺至少两项胜出。

## License

MIT
