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
# 包安装（外部脚本推荐）：稳定导入名 figure_forge
pip install .                                # 仓库根目录直接装
python -m pip wheel --no-deps . -w dist      # 或先出 wheel 分发
pip install dist/figure_forge-*.whl
pip install -e .                             # 可编辑安装（开发模式）
```

```python
import figure_forge as ff
ff.__version__                                 # 版本单源于 figure_forge/_version.py
from figure_forge import apply_style, new_figure, save_figure, run_qa
from core import apply_style                   # 兼容期旧导入，继续可用
```

源码目录直接跑（agent skill 场景，无需安装）：

```bash
git clone https://github.com/HAMBURGER31522/create_image_fukinggood.git figure-forge
cd figure-forge
pip install -r requirements.txt
```

依赖：Python ≥ 3.10（CI 在 3.10/3.13 上做安装态测试），matplotlib ≥ 3.8，
numpy ≥ 1.26，pandas ≥ 2.0；seaborn 可选（`pip install "figure-forge[full]"`，
缺失时色图自动回退内置档）；scipy、openpyxl 仅供 recipes/ 使用，随
requirements.txt 装。
中文字体：优先 Source Han Serif SC（思源宋体），回退 SimSun/SimHei/微软雅黑，装了任意一个即可。

## 作为 Cursor/Claude skill 使用

把本仓库放到 skill 目录（如 `~/.cursor/skills/figure-forge/` 或项目
`.cursor/skills/figure-forge/`），agent 会读取 `SKILL.md` 按工作流出图。
对话里直接说：

> 用 figure-forge 画问题二的导通概率图，数据在 xxx.csv

## 手动使用（三步）

```python
from figure_forge import apply_style, new_figure, save_figure, run_qa, \
    stat_box, callout          # 未安装、直接用源码目录时见下方 SKILL.md 骨架

apply_style()                                  # 1. 全局样式（必须最先调）
fig, ax = new_figure("onehalf", ratio=0.62)    # 2. 按期刊栏宽建图
ax.plot(...)                                   #    构图抄 recipes/ 里最接近的模板
stat_box(ax, [f"n = {len(x)}",                 #    统计注释框：数字必须是变量
              f"RMS = {rms:.2f} cm"])
run_qa(fig, expect_width=("onehalf",))         # 3. 先 QA：不过直接抛错
save_figure(fig, "out/图名")                    # 4. 过了再导出 png(300dpi)+svg
```

可选溯源台账：拿到交付目录的人不开驱动就能回答"这张图主张什么、数据
从哪来、哪个脚本生成"。`record`/`manifest_path` 是可选参数，不传时
`save_figure(fig, stem)` 行为不变；QA 未通过的图不入台账。

```python
from figure_forge import FigureRecord

record = FigureRecord(
    id="q4_1_optimum",
    claim="满足证书下界的最低成本方案为纯介质A",
    source_data=("结果/二维响应面.csv", "结果/最终方案.csv"),
    generation_script=__file__,
)
save_figure(fig, "out/图名", record=record,
            manifest_path="out/figure_manifest.csv")
# out/figure_manifest.csv：id,path,formats,claim,source_data,
# generation_script,preset,journal,qa_status,sha256（逐格式哈希，幂等 upsert）
```

### 期刊档（IEEE / PNAS）

`journal=` 是正交于 `preset` 的交付约束档：preset 管语言与纹理，
journal 只覆盖**有官方出处**的栏宽、字号区间与图高上限（值与出处
登记在 `core/journals.py`，不可变 profile）。

```python
apply_style("nature", journal="ieee")   # sans 纹理 + IEEE 栏宽/字号约束
fig, ax = new_figure("single")          # 88.9mm（IEEE 官方单栏 3.5in）
run_qa(fig)                             # 按 IEEE 档核对：字号 8–10pt、高≤220mm
save_figure(fig, "out/fig")             # 未过 QA 依旧 0 文件落盘
```

| 档 | 栏宽 (mm) | 字号 (pt) | 图高上限 | 约束值官方出处（复核 2026-09-01） |
|---|---|---|---|---|
| `ieee` | single 88.9 / double 182 | 8–10（基准 9） | 220 mm | [IEEE Author Center · Resolution and Size](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/)、[Improve Your Graphics](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/improve-your-graphics/)、[IEEE PES 作者套件](https://ieee-pes.org/publications/authors-kit/preparation-of-a-formatted-technical-work/) |
| `pnas` | single 90 / onehalf 110 / double 180 | 6–12（基准沿用 preset） | 220 mm | [PNAS Author Center · Submitting Your Manuscript](https://www.pnas.org/author-center/submitting-your-manuscript) |

边界如实声明：IEEE/PNAS 官方**没有**背景网格、彩色文字、线宽的禁令
或区间，这两档因此不设这三项档级检查（不造数）；这三项硬拒只属于
`nature` 档（Nature 明文）。`journal="ieee"` 只说明"按 IEEE 官方约束
核对过"，**调用成功不等于合规**——交付以 `run_qa` 通过为准，官方
改版后须重核 `reviewed_on` 日期并更新注册表。

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
recipes/                 23 个图种模板（每个自带合成数据 demo，可直接运行），
                         含 ROC+PR、聚类、TSP/VRP 路线图等 B/C 题高频图种
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
