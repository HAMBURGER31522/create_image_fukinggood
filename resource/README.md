# figure-forge 范文与 analytic figure 资源库

> 生成日期：2026-08-17  
> 检索工具：smart-search + 本地 HTTP/GitHub API 核验 + `pdftotext` 图注识别  
> 口径：只收录**本机可打开/可核验**的入口；未核验的“幽灵 PDF 直链”一律不写。

本目录两份主文件：

1. [`paper-link-index.md`](paper-link-index.md) — **期刊 + 数学建模竞赛范文链接集总**
2. [`analytic-figure-types.md`](analytic-figure-types.md) — **按图种分类**，每类附范文链接与识别依据

辅助：

- [`samples-verified.json`](samples-verified.json) — 机器可读索引（URL、图种、证据）

## 本地 PDF 论文库（已迁出）

完整范文 PDF 库**不在本 skill resource 内**，已迁移到数学建模根目录：

```
F:\mathematical modeling\papers
```

- 库说明：`F:\mathematical modeling\papers\README.md`
- 迁移报告：`F:\mathematical modeling\papers\MIGRATE-REPORT.md`
- 旧路径路标：[`papers/MOVED.md`](papers/MOVED.md)（若目录仍在）

---

## 使用建议（对齐 figure-forge）

1. 先看 `analytic-figure-types.md` 的**图种**，不要先抄“某篇论文的全部图”。
2. 每个图种下的范文是**构图参考**，不是数据模板；数据形状变了构图也要变。
3. 选图仍以 `reference/taxonomy.md` 决策树为准：论点合同 → 图种 → recipe。
4. O 奖/国奖范文里常见“平庸但能用”的柱/折线；期刊方法文（Weissgerber / Allen raincloud / Rougier）用来抬构图层级。

## 证据边界（读之前先知道）

| 来源 | 能确认什么 | 不能假装确认什么 |
|---|---|---|
| COMAP 历年 results 页 | 队伍号、Outstanding 名单、题目分卷 | 官方并不在 results 页直接挂全文 PDF |
| GitHub O 奖合集 | 可下载全文 PDF（社区归档） | 不等于 COMAP 官方发布渠道；版权归原作者/COMAP |
| 国赛官方 mcm.edu.cn | 竞赛权威入口 | 本环境对官网 TLS 不稳定；优秀论文常分散在期刊/高校/网盘 |
| 打开的 O 奖 PDF + pdftotext | 图题/图注文字 → 图种 | 扫描件 PDF 抽不出字，图种只能目测或跳过 |
| PLOS / Wellcome OA | 方法级 analytic figure 范式 | 不是数模赛题范文，但是期刊级分布图权威 |

## 快速入口

- 美赛 O 奖 PDF 合集（2013–2025，403 篇）：https://github.com/Jackksonns/MCM-ICM-Outstanding-Papers  
- COMAP 历年赛题/结果总表：https://www.contest.comap.com/undergraduate/contests/mcm/previous-contests.html  
- 国赛/美赛/研赛资料大库：https://github.com/zhanwen/MathModel  
- 分布图范式（反对均值柱）：https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128  
- 雨云图原文：https://wellcomeopenresearch.org/articles/4-63  
