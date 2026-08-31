# API 速查表

复用前先看这页，不必逐个读源码。约定：`width` 取 `single(89mm) / onehalf(136mm) / double(183mm) / cn(150mm)` 或 mm 数值。

## core（`from core import ...`）

| 函数 | 签名要点 | 返回 |
|---|---|---|
| `apply_style` | `(preset="cn"\|"nature", base_size=None, draft=False)`；必须最先调用；draft 降 dpi 只出 PNG。`preset=None` 时读环境变量 **`FF_PRESET`**（缺省 `"cn"`）——批量出图切档用它，不必改代码 | None |
| `new_figure` | `(width="onehalf", ratio=0.62, **subplots_kw)` | `fig, ax` |
| `save_figure` | `(fig, path_no_ext, formats=("png","svg","pdf"), tight=True, exact_width=True, force=False)`；tight bbox 补白回声明栏宽（交付宽度=声明宽度）；3D 图传 `tight=False`；自动建目录；草稿档文件名加 `_DRAFT` | 输出文件列表 |
| `run_qa` | `(fig, expect_width=None, strict=True, sourced=None, allow=())`；`expect_width` 可传单个档名/mm 数值或元组；先 QA 再 save，坏图不落盘。查：字号/色图/硬拒绝构图（分组竖柱、饼、双 Y 轴）/注释层/豆腐块/遮挡（图例·注释框·直标·标题两两互压）/交付宽度与图高/面板数/墨迹密度/轴限利用率/比例轴越界/数值溯源与图题-注释数值矛盾/跨面板重复系列/小倍数色标一致/可达性/稀疏离散点连折线/文字对比度（实测像素背景）/同轴不可通约量。`allow` 码：`grouped_bars` `unsourced` `overlap` `accessibility` `unexplained_band` `number_conflict` `duplicate_series` `clim_mismatch` `axis_slack` `sparse_line` `unit_axis_range` `incommensurable` `text_contrast` `sparse_panel` `nonfinite_text`；**未知码直接抛错**（拼错静默无效比报错更伤） | 问题列表 |
| `load_table` | `(path, sheet=0)`；xlsx/CSV 读表，CSV 自动试 utf-8-sig/gbk/utf-8 | DataFrame |
| `as_1d` | `(x, col=None)`；DataFrame 取列 / 任意序列 → 一维 float ndarray，非数值转 NaN | ndarray |
| `stat_box` | `(ax, lines, loc="auto", fontsize=None, outside=None, expand_axes=True)`；lines 为字符串列表；`loc="auto"` 在 8 个锚点里选压数据最少的；满铺场图自动降级到 `outside="top"`；轴内无真空位时传 `outside="bottom"` | Text |
| `callout` | `(ax, xy, text, xytext, color, rad=0.12, textcoords="data", mark=False)`；目标点无标记时开 `mark=True`。`xy` 含 nan/inf 直接抛错（退化标注肉眼不可见，而 QA 会一路 PASS） | Annotation |
| `end_label` | `(ax, x, y, text, color, dx_pt=4.0)`；线端直标替代图例 | Annotation |
| `end_labels` | `(ax, items, dx_pt=4.0, fontsize=None)`；items=[(x,y,text,color)]，多条线端直标一次排完并自动避让 | [Annotation] |
| `ref_line` | `(ax, value, orientation="h"\|"v", label=None, label_loc=None, level="focus")`；`label_loc` 横线取 `left/right`（默认 right）、竖线取 `top/bottom`（默认 top），传另一方向的值报错 | None |
| `text_color` | `(color)`；**文字着色的唯一入口**。nature 档返黑字（Nature 明文 "Avoid coloured text"，彩色文字是硬拒），cn 档走 `ink()` 压暗。recipe 里裸 `ax.annotate(color=…)` 一律用它——用 `ink()` 在 nature 档会被硬拒 | 色号 |
| `ink` | `(color, min_ratio=3.3)`；把语义色压暗到对白底 ≥3.3:1（比 QA 的 3.0 门槛留余量）。**只在确定不进 nature 档时直接用**，否则走 `text_color()` | 色号 |
| `ptx` | `(v, kind="font"\|"lw"\|"pt")`；把**按 cn 档调好的**点值换算到当前档。`font` 按字号比例缩放并夹进该档包线，`lw` 按该档线宽上限封顶，`pt` 纯比例（markersize 等）。kind 拼错直接报错——此前会静默按 font 缩放，`ptx(1.6,"linewidth")` 在 nature 档返回 5.0 而不是 1.0 | 点值 |
| `smart_legend` | `(ax, *args, **kw)`；按占用探测选位，避开数据与直标 | Legend |
| `dot_interval` | `(ax, labels, est, lo, hi, threshold=None, thr_label="", better="high", sizes=None, sort=True, value_col=True, return_order=False)`；点区间/森林图：排序+判据线+达标着色+轴外数值列。**判定按区间靠判据的那一侧**（下界过线才算达标）。`better` 只认 `"high"`/`"low"`，`value_col` 只认 `True`/`False`/`"outside"`/`"inside"`，非法值直接报错（此前 `better="higher"` 会**静默反转达标判定**）。`value_col="inside"` 用于多面板（无轴外空间）；`value_col=True` 会自动收缩本轴给数值列让位（右界取画布右缘与右邻面板左缘中更靠左者），不必手动 `subplots_adjust` | `ok`，或 `(ok, order)` |
| `slope_lines` | `(ax, labels, before, after, cond_names=("前","后"), unit="", highlight=(), higher_is_better=True, mode="emphasis", verdict="", label_ends=True)`；斜率图：**每条线按变化方向着色**；`mode="cohort"` 用于大 N（群体压灰、只高亮个体）；`verdict` 标在面板内顶部。长度不一致/highlight 越界直接抛 ValueError | `dict(up,down,flat,invalid)` |
| `figure` | `(spec, width, height=None, row_heights=None, hspace=, wspace=)`；按内容分配面板尺寸（非等分），自动 a/b/c 编号 | `fig, {name: ax}` |
| `share_colorbar` | `(fig, mappable, axes, label="", loc="right", size=0.018, pad=0.015, shrink=1.0)`；多面板共享色标 | Colorbar |
| `marginal_grid` | `(width, ratio=0.85, right=True, top=True, size=0.22)` | `fig, ax_main, ax_top, ax_right` |
| `small_multiples` | `(n, ncols=4, width="double", ratio=1.0)` | `fig, [axes]` |
| `panel_label` | `(ax, "a", dx=-0.08, dy=1.04)` | None |
| `inset_zoom` | `(ax, bounds, xlim, ylim)`；bounds 为轴分数 (x0,y0,w,h) | inset 轴 |
| `PALETTE` / `OKABE_ITO` | 低饱和序列（主用）/ 色盲安全 8 色 | — |
| `semantic` | `("data"/"fit"/"baseline"/"highlight"/"band"/"good"/"bad")` | 色号 |
| `cmap_for` | `("diverging"/"sequential"/"sequential2"/"surface"/"heatmap")` | cmap |
| `categorical` | `(n, muted=False)`；返回 n 组「颜色 + 配套 marker + 线型」，>4 类时冗余编码必备 | `cols, marks, lss` |
| `emphasis` | `(color, level="focus"\|"context"\|"background")`；按重要性压饱和度 | 色号 |
| `check_accessibility` | `(colors, min_dist=0.12, min_grey_gap=0.10, redundant=False)`；三类色盲 + 灰度复查 | 问题列表 |
| `simulate_cvd` | `(color, kind="deuteranopia"\|"protanopia"\|"tritanopia")` | 色号 |
| `PALETTE_MUTED` / `MARKERS` / `LINESTYLES` | 低饱和序列 / 配套标记 / 配套线型 | — |
| `MAX_PANELS` / `MAX_HEIGHT_MM` / `COLUMN_WIDTHS` | 6 / 170mm / 栏宽表 | — |
| `truncate_cmap` | `(cmap, lo=0.12, hi=0.88)`；大面积铺色前截断高饱和两端 | cmap |

## recipes（复制构图用；`info` 字典用于图题 f-string）

| 文件 | 函数 | 签名要点 | 返回 |
|---|---|---|---|
| pipeline_diagram | `pipeline` | `(lanes, flows, feedbacks, width="double")`；lanes=[(泳道名,[模块文本])]，flows=[((i,j),(i,j))]；跨泳道箭头锚在实测框缘 | `fig, ax, centers` |
| comparison_rank | `sorted_lollipop` | `(labels, values, unit, highlight=None, title)`；highlight=None 自动强调最大值，或传原始索引/标签名（与 slopegraph 一致） | `fig, ax` |
| | `dumbbell` | `(labels, before, after, cond_names, unit, xlabel, higher_is_better=True)` | `fig, ax` |
| | `slopegraph` | `(labels, before, after, cond_names, unit, highlight=(), higher_is_better=True, mode="emphasis"\|"cohort", verdict="", ratio=0.85)`；**返回三元组**（旧版是 `fig, ax`，照旧写法解包会 ValueError） | `fig, ax, counts` |
| | `dot_interval`（recipe） | 见 `recipes/dot_interval.py`：离散档位+区间+判据的完整用法与 `wilson()` | — |
| | `butterfly` | `(labels, left, right, left_name, right_name, unit)` | `fig, ax` |
| | `facet_metrics` | `(cat_labels, [(标题, 值, "log"/"linear")], width="double")` | `fig, axes` |
| composition | `share_bars` | `(labels, counts, unit="", width="single", highlight=None)` | `fig, ax` |
| | `waffle` | `(labels, counts, width="single", n=10)`；≤4 类 | `fig, ax` |
| | `stacked_share` | `(group_labels, cat_labels, matrix, width="onehalf", emphasize=None)` | `fig, ax` |
| | `parallel_coords` | `(names, data, dims, highlight_idx, better=["↑","↓",...])` | `fig, ax` |
| raincloud | `raincloud` | `(groups, labels, ylabel)` | `fig, ax` |
| | `ridgeline` | `(groups, labels, xlabel="值", width="single", cmap_colors=None)` | `fig, ax` |
| joint_marginal | `joint_hexbin` | `(x, y, xlabel, ylabel, effective_r=None, quantiles=(0.5,0.9), center=(0,0), unit="")`；center 默认原点仅适用偏差坐标，真实坐标需显式传或 `center=None` 用中位数 | `fig, ax, stats` |
| fit_residual | `fit_residual_pair` | `(x, y, xfit, yfit, resid, band, threshold, x_at_threshold, ...)` | `fig, (ax1,ax2), inside` |
| parity | `parity` | `(y_true, y_pred, band=("relative",0.10) 或 ("absolute",δ))`；数据跨 0 用绝对带 | `fig, ax, info(r2/rmse/mape/mae/inside)` |
| timeseries_forecast | `forecast_fan` | `(t_hist, y_hist, t_fore, y_fore, bands={level:(lo,hi)}, split, y_test)`；统计框内置 | `fig, ax, info(mape/coverage/half_w_last)` |
| roc_pr | `roc_pr` | `([(名, y_true, score), ...])`；ROC+PR 双联，AUC/AP 内部计算 | `fig, axes, info{名:{auc,ap}}` |
| cluster_scatter | `cluster_scatter` | `(X(n,2), labels, xlabel="特征 1", ylabel="特征 2", cluster_names=None, width="onehalf", n_std=2.0)`；-1 为噪声；轮廓系数内置（n ≲ 数千） | `fig, ax, info(silhouette/n_clusters/sizes)` |
| route_map | `route_map` | `(nodes(n,2), routes=[[索引...]], depot=0, labels=None)`；TSP/VRP 路线+序号 | `fig, ax, info(dists/total/imbalance)` |
| annotated_heatmap | `confusion_matrix` | `(M, class_names)` | `fig, ax, info(acc/macro_recall/share)` |
| | `corr_matrix` | `(R, names, emph_thresh=0.7)` | `fig, ax, info(n_strong)` |
| convergence_ci | `convergence_pair` | `(n, est, half, true=None)` | `fig, (ax1,ax2)` |
| algo_convergence | `convergence_curves` | `([(名, 每代原始值, 色)], xlabel="迭代代数", ylabel="目标函数值", logy=False, conv_tol=1e-3, mode="min"/"max"/"raw", width="onehalf")`；自动转 best-so-far | `fig, ax, info{名:(收敛代,终值)}` |
| scan_curve | `scan_curve` | `(x, [(名, y, 色)], xlabel, ylabel, logy=True, refine_span, yticks)` | `fig, ax` |
| contour_field | `masked_diverging_field` | `(X, Y, Z, R, ...)` | `fig, ax, info(r_max_frac/r_min_frac)` |
| | `polar_field` | `(theta, r, Z, zlabel)` | `fig, ax` |
| feasible_zoom | `feasible_contour_zoom` | `(X, Y, cost, frontier_xy, best, zoom_xlim, zoom_ylim)` | `fig, ax` |
| front_overlay | `response_overlay` | `(X, Y, P, fronts, best, levels=(0.5,0.9))` | `fig, ax` |
| | `pareto_front` | `(f1, f2, labels, knee=None, minimize=(True,True))` | `fig, ax, info(fx/fy/knee)` |
| surface3d_project | `surface_with_projection` | `(X, Y, Z, best, stat_lines)`；保存用 `tight=False` | `fig, ax` |
| small_multiples_frames | `frame_snapshots` | `(frames, titles, stats, xy=None, R=None, zlabel="偏差（m）", ncols=2, width="onehalf")` | `fig, axes` |
| tornado | `tornado` | `(factors, low, high, baseline, xlabel)` | `fig, ax` |
| | `spider_cartesian` | `(pct, outputs, names, ylabel)` | `fig, ax` |
| phase_transition | `phase_density_orderparam` | `(phi, samples, order_param, phi_c)` | `fig, (ax1,ax2)` |
| network_percolation | `spatial_networks` | `([(nodes, edges, spanning_mask)], titles, xlim, ylim)` | `fig, axes` |

## tools

| 文件 | 用法 |
|---|---|
| contact_sheet.py | `python tools/contact_sheet.py <图目录> [--cols 3]` → `_contact_sheet_N.png`，一次 Read 批量目测 |
