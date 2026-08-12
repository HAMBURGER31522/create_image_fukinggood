# API 速查表

复用前先看这页，不必逐个读源码。约定：`width` 取 `single(89mm) / onehalf(136mm) / double(183mm) / cn(150mm)` 或 mm 数值。

## core（`from core import ...`）

| 函数 | 签名要点 | 返回 |
|---|---|---|
| `apply_style` | `(base_size=9.0, draft=False)`；必须最先调用；draft 降 dpi 只出 PNG | None |
| `new_figure` | `(width="onehalf", ratio=0.62, **subplots_kw)` | `fig, ax` |
| `save_figure` | `(fig, path_no_ext, formats=("png","svg"), tight=True)`；3D 图传 `tight=False`；自动建目录 | 输出文件列表 |
| `run_qa` | `(fig, expect_width=None, strict=True)`；字号/色图/硬拒绝构图/注释层/豆腐块/图例遮挡 | 问题列表 |
| `stat_box` | `(ax, lines, loc="upper left", fontsize=7.0)`；lines 为字符串列表 | Text |
| `callout` | `(ax, xy, text, xytext, color, rad=0.25, textcoords="data", mark=False)`；目标点无标记时开 `mark=True` | Annotation |
| `end_label` | `(ax, x, y, text, color, dx_pt=4.0)`；线端直标替代图例 | Annotation |
| `ref_line` | `(ax, value, orientation="h", label=None, label_loc="right")` | None |
| `marginal_grid` | `(width, ratio=0.85, right=True, top=True, size=0.22)` | `fig, ax_main, ax_top, ax_right` |
| `small_multiples` | `(n, ncols=4, width="double", ratio=1.0)` | `fig, [axes]` |
| `panel_label` | `(ax, "a", dx=-0.08, dy=1.04)` | None |
| `inset_zoom` | `(ax, bounds, xlim, ylim)`；bounds 为轴分数 (x0,y0,w,h) | inset 轴 |
| `PALETTE` / `OKABE_ITO` | 低饱和序列（主用）/ 色盲安全 8 色 | — |
| `semantic` | `("data"/"fit"/"baseline"/"highlight"/"band"/"good"/"bad")` | 色号 |
| `cmap_for` | `("diverging"/"sequential"/"sequential2"/"surface"/"heatmap")` | cmap |
| `truncate_cmap` | `(cmap, lo=0.12, hi=0.88)`；大面积铺色前截断高饱和两端 | cmap |

## recipes（复制构图用；`info` 字典用于图题 f-string）

| 文件 | 函数 | 签名要点 | 返回 |
|---|---|---|---|
| pipeline_diagram | `pipeline` | `(lanes, flows, feedbacks, width="double")`；lanes=[(泳道名,[模块文本])]，flows=[((i,j),(i,j))] | `fig, ax, centers` |
| comparison_rank | `sorted_lollipop` | `(labels, values, unit, highlight=0, title)` | `fig, ax` |
| | `dumbbell` | `(labels, before, after, cond_names, unit, xlabel, higher_is_better=True)` | `fig, ax` |
| | `slopegraph` | `(labels, before, after, cond_names, unit, highlight=(), higher_is_better=True)` | `fig, ax` |
| | `butterfly` | `(labels, left, right, left_name, right_name, unit)` | `fig, ax` |
| | `facet_metrics` | `(cat_labels, [(标题, 值, "log"/"linear")], width="double")` | `fig, axes` |
| composition | `share_bars` | `(labels, counts, unit, highlight=None)` | `fig, ax` |
| | `waffle` | `(labels, counts, n=10)`；≤4 类 | `fig, ax` |
| | `stacked_share` | `(group_labels, cat_labels, matrix, emphasize=None)` | `fig, ax` |
| | `parallel_coords` | `(names, data, dims, highlight_idx, better=["↑","↓",...])` | `fig, ax` |
| raincloud | `raincloud` | `(groups, labels, ylabel)` | `fig, ax` |
| | `ridgeline` | `(groups, labels, xlabel, cmap_colors=None)` | `fig, ax` |
| joint_marginal | `joint_hexbin` | `(x, y, xlabel, ylabel, effective_r=None, quantiles=(0.5,0.9))` | `fig, ax, stats` |
| fit_residual | `fit_residual_pair` | `(x, y, xfit, yfit, resid, band, threshold, x_at_threshold, ...)` | `fig, (ax1,ax2), inside` |
| parity | `parity` | `(y_true, y_pred, band_pct=0.10)` | `fig, ax, info(r2/rmse/mape/inside)` |
| timeseries_forecast | `forecast_fan` | `(t_hist, y_hist, t_fore, y_fore, bands={level:(lo,hi)}, split, y_test)` | `fig, ax` |
| annotated_heatmap | `confusion_matrix` | `(M, class_names)` | `fig, ax, info(acc/macro_recall/share)` |
| | `corr_matrix` | `(R, names, emph_thresh=0.7)` | `fig, ax, info(n_strong)` |
| convergence_ci | `convergence_pair` | `(n, est, half, true=None)` | `fig, (ax1,ax2)` |
| algo_convergence | `convergence_curves` | `([(名, best_so_far, 色)], logy=False, conv_tol=1e-3)` | `fig, ax, info{名:(收敛代,终值)}` |
| scan_curve | `scan_curve` | `(x, [(名, y, 色)], xlabel, ylabel, logy=True, refine_span, yticks)` | `fig, ax` |
| contour_field | `masked_diverging_field` | `(X, Y, Z, R, ...)` | `fig, ax, info(r_max_frac/r_min_frac)` |
| | `polar_field` | `(theta, r, Z, zlabel)` | `fig, ax` |
| feasible_zoom | `feasible_contour_zoom` | `(X, Y, cost, frontier_xy, best, zoom_xlim, zoom_ylim)` | `fig, ax` |
| front_overlay | `response_overlay` | `(X, Y, P, fronts, best, levels=(0.5,0.9))` | `fig, ax` |
| | `pareto_front` | `(f1, f2, labels, knee=None, minimize=(True,True))` | `fig, ax, info(fx/fy/knee)` |
| surface3d_project | `surface_with_projection` | `(X, Y, Z, best, stat_lines)`；保存用 `tight=False` | `fig, ax` |
| small_multiples_frames | `frame_snapshots` | `(frames, titles, stats, xy, R, ncols=2)` | `fig, axes` |
| tornado | `tornado` | `(factors, low, high, baseline, xlabel)` | `fig, ax` |
| | `spider_cartesian` | `(pct, outputs, names, ylabel)` | `fig, ax` |
| phase_transition | `phase_density_orderparam` | `(phi, samples, order_param, phi_c)` | `fig, (ax1,ax2)` |
| network_percolation | `spatial_networks` | `([(nodes, edges, spanning_mask)], titles, xlim, ylim)` | `fig, axes` |

## tools

| 文件 | 用法 |
|---|---|
| contact_sheet.py | `python tools/contact_sheet.py <图目录> [--cols 3]` → `_contact_sheet_N.png`，一次 Read 批量目测 |
