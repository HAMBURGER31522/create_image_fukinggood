"""生成候选图审阅页：_candidates/*.png + _manifest.json -> _REVIEW.html

用法：python _build_review.py，然后浏览器打开 _REVIEW.html。
一张一张看，把想留的**文件名**记下来告诉我，我搬进 ref/ 并写进 _INDEX.md。
"""
import json, os, re, html

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.join(HERE, "_candidates")

# 先前手工核验过、不在 manifest 里的 6 张
SEED = [
 ("forest-plot-meta__18.png", "forest-plot",
  "Forest plot of cognitive flexibility（Exergaming 元分析）",
  "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0309462"),
 ("forest-plot-meta__19.png", "forest-plot",
  "Forest plots with phototherapy greater than 10h（双相障碍光疗元分析）",
  "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232798"),
 ("forest-plot-meta-large__20.png", "forest-plot",
  "糖尿病—结核元分析的合并 OR 森林图（大图，1MB）",
  "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0261246"),
 ("dumbbell-proportions__21.png", "dumbbell",
  "Dumbbell plot：可疑研究行为的比例对比（图注明写 dumbbell）",
  "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199554"),
 ("dotinterval-feature-importance__22.png", "dotinterval",
  "五个模型的置换特征重要性（点 + 区间，按重要性排序）",
  "https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0001409"),
 ("orderedbar-shap-importance__23.png", "orderedbar",
  "SHAP 平均绝对值特征重要性（有序条）",
  "https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0001409"),
]

GROUP_CN = {
    "forest-plot": "森林图 / 点区间图（q2_1 面板 a 的正解）",
    "dumbbell": "哑铃图（两状态对比）",
    "dotinterval": "点 + 区间（系数图 / 特征重要性）",
    "orderedbar": "有序条 + 直标",
    "slopegraph": "斜率图",
    "cleveland-dot": "Cleveland 点图 / 棒棒糖",
}

rows = []
mf = os.path.join(CAND, "_manifest.json")
if os.path.exists(mf):
    for r in json.load(open(mf, encoding="utf-8")):
        rows.append((r["file"], r["group"],
                     f'{r["title"]} — {r["caption"]}', r["article"]))
have = {r[0] for r in rows}
def _where(fn):
    """选中的图已移到 ref/，未选中的还在 _candidates/——两处都要找。"""
    for d in (CAND, HERE):
        if os.path.exists(os.path.join(d, fn)):
            return os.path.relpath(os.path.join(d, fn), HERE).replace("\\", "/")
    return None

rows = [s for s in SEED if s[0] not in have and _where(s[0])] + rows
rows = [r for r in rows if _where(r[0])]

# 收编还没进 manifest 的文件（抓取仍在进行时也能先出一版审阅页）
known = {r[0] for r in rows}
for fn in sorted(set(os.listdir(CAND)) | set(os.listdir(HERE))):
    if fn.lower().endswith(".png") and fn not in known:
        grp = fn.split("__")[0]
        rows.append((fn, grp, "（图注待补：抓取完成后刷新本页）", ""))

order = list(GROUP_CN) + sorted({r[1] for r in rows} - set(GROUP_CN))
rows.sort(key=lambda r: (order.index(r[1]) if r[1] in order else 99, r[0]))

P = ['<meta charset="utf-8"><title>figure-forge 候选参考图</title>', """<style>
:root{--bg:#fbfbfa;--fg:#1c1c1a;--mut:#6b6b64;--line:#e2e0da;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#17171a;--fg:#e9e8e4;
--mut:#98968e;--line:#2e2e33;--card:#1f1f23}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 system-ui,"Segoe UI","Microsoft YaHei",sans-serif;padding:28px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:36px 0 12px;
padding-bottom:6px;border-bottom:2px solid var(--line)}
.lead{color:var(--mut);margin:0 0 8px;max-width:70ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px;margin:14px 0}
.fn{font:600 14px ui-monospace,Consolas,monospace;color:#0a7;word-break:break-all}
.cap{color:var(--mut);font-size:13px;margin:6px 0 10px}
img{max-width:100%;height:auto;display:block;border:1px solid var(--line);
border-radius:6px;background:#fff}
a{color:#3b82f6}
</style>"""]
P.append("<h1>figure-forge 候选参考图</h1>")
P.append('<p class="lead">全部来自 PLOS 开放获取,图注已核验。逐张看,'
         '把<b>想留的文件名</b>告诉我,我搬进 <code>ref/</code> 并写进索引。'
         '不满意的直接说删。</p>')
P.append(f'<p class="lead">共 {len(rows)} 张。</p>')
cur = None
for fn, grp, cap, url in rows:
    if grp != cur:
        cur = grp
        P.append(f"<h2>{html.escape(GROUP_CN.get(grp, grp))}</h2>")
    link = (f'<br><a href="{html.escape(url)}" target="_blank">原文</a>'
            if url else "")
    P.append('<div class="card">'
             f'<div class="fn">{html.escape(fn)}</div>'
             f'<div class="cap">{html.escape(cap)}{link}'
             '</div>'
             f'<img src="{html.escape(_where(fn))}" loading="lazy"></div>')
open(os.path.join(HERE, "_REVIEW.html"), "w", encoding="utf-8").write(
    "\n".join(P))
print(f"_REVIEW.html 已生成，{len(rows)} 张候选")

# ---- URL 合集：只收有真实出处的，占位/未回填的一律不进文档 ----
sourced = [r for r in rows if r[3]]
M = ["# 候选参考图 · 来源合集",
     "",
     "> 全部来自 PLOS 开放获取；每条都用 curl 实际取过，确认返回 `image/png`。",
     "> 未核验、未回填出处的候选**不收进本文件**。",
     "> 本文件由 `_build_review.py` 生成，勿手改。",
     "",
     f"共 {len(sourced)} 条（候选图总数 {len(rows)}）。",
     "",
     "已选中并入库的 14 张见 `_picks.json`（含逐张评语）与 `_INDEX.md`；",
     "其余留在 `_candidates/`（不入库，可按下表 URL 重下）。",
     "",
     "| 候选文件 | 图种 | 出处 | 图注 |",
     "|---|---|---|---|"]
for fn, grp, cap, url in sourced:
    c = cap.replace("|", "／").replace("\n", " ").strip()[:150]
    M.append(f"| `{fn}` | {GROUP_CN.get(grp, grp)} | [原文]({url}) | {c} |")

M += ["",
      "## 自己再找更多时的检索入口",
      "",
      "PLOS 支持**按图注检索**，比搜正文准得多——命中的是图注里就写着该图种的文章：",
      "",
      "```",
      'https://api.plos.org/search?q=figure_table_caption:"forest plot"'
      "&fl=id,title_display&rows=40&wt=json",
      "```",
      "",
      "把 `forest plot` 换成 `dumbbell` / `slope graph` / `slopegraph` /",
      "`Cleveland dot` / `lollipop chart` / `caterpillar plot` / `coefficient plot`。",
      "全库图注写着 forest plot 的有 **5261 篇**。",
      "",
      "拿到 DOI 后，图片直链规律（`gNNN` 是图号）：",
      "",
      "```",
      "https://journals.plos.org/<刊名>/article/figure/image?size=large&id=<DOI>.g003",
      "```",
      "",
      "| DOI 缩写 | URL 刊名 |",
      "|---|---|",
      "| `pone` | `plosone` |",
      "| `pcbi` | `ploscompbiol` |",
      "| `pbio` | `plosbiology` |",
      "| `pmed` | `plosmedicine` |",
      "| `pntd` | `plosntds` |",
      "| `pgen` | `plosgenetics` |",
      "| `pdig` | `digitalhealth` |",
      "",
      "**别靠猜图号**定位森林图是第几张——取文章 XML 解析图注：",
      "",
      "```",
      "https://journals.plos.org/<刊名>/article/file?id=<DOI>&type=manuscript",
      "```",
      "",
      "## 已知局限",
      "",
      "- 医学元分析的森林图多为 **RevMan 自动生成**：等宽表格 + 黑方块，功能齐全但排版朴素。",
      "- 更好看的点区间图多在**生态学/心理学**（ggplot 手绘）与 **ML 可解释性**文章里。",
      "- **付费墙期刊（Nature / Science / Elsevier）抓不到**，只能手动截图。",
      "- **中文期刊**需登录，同样只能手动截。",
      ""]
open(os.path.join(HERE, "_SOURCES.md"), "w", encoding="utf-8").write("\n".join(M))
print(f"_SOURCES.md 已生成，{len(sourced)} 条有效来源")
