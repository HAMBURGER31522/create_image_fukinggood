"""优化算法收敛曲线：迭代-目标值，多算法对照 + 收敛代标注 + 末值直标。

论点合同示例：
- 结论：改进 GA 比标准 GA 提前约 25 代收敛，终值更优且更稳定。
  （具体代数由数据算出，见图题的 f-string——写死在这里必然与图漂移）
- 证据链：best-so-far 单调线 → 收敛代竖标 → 末端直标终值 → 统计框给设置。
替代：把每代种群均值画成杂乱多折线（平庸）。
"""
from _common import GALLERY
import numpy as np

from core.style import _one_of
from core import (text_color, ptx, ink, apply_style, new_figure, save_figure, run_qa,
                  stat_box, end_label, PALETTE)


def convergence_curves(curves, xlabel="迭代代数", ylabel="目标函数值",
                       logy=False, conv_tol=1e-3, mode="min",
                       width="onehalf"):
    """curves: [(名称, 每代目标值数组, 颜色), ...]。

    mode: "min"/"max" 自动转 best-so-far 单调线（喂原始每代值即可），
          "raw" 按原样画（已是 best-so-far 时用）。
    conv_tol: 相对终值变化 < conv_tol 视为收敛，自动标注收敛代。
    返回 (fig, ax, info)，info[名称] = (收敛代, 终值)。
    """
    # 裸 KeyError('minimum') 不列合法值，与 stat_box(loc=) 修前同一个
    # 毛病：能拦住，但用户只能去读源码才知道该写什么。
    _one_of("convergence_curves(mode=)", mode, ("min", "max", "raw"))
    acc = {"min": np.minimum.accumulate, "max": np.maximum.accumulate,
           "raw": lambda y: y}[mode]
    curves = [(name, acc(np.asarray(y, dtype=float)), c)
              for name, y, c in curves]
    fig, ax = new_figure(width, ratio=0.55)
    if logy:
        ax.set_yscale("log")
    info = {}
    n_max = max(len(y) for _, y, _ in curves)
    # 终值接近的曲线，线端标签上下错开避免相压
    finals = sorted(range(len(curves)),
                    key=lambda k: curves[k][1][-1], reverse=True)
    va_of = {idx: ("bottom" if r % 2 == 0 else "top")
             for r, idx in enumerate(finals)}
    _below: list[bool] = []
    for k, (name, y, c) in enumerate(curves):
        y = np.asarray(y, dtype=float)
        it = np.arange(len(y))
        ax.plot(it, y, color=c, linewidth=ptx(1.4, "lw"), zorder=3)
        # 收敛代：此后所有值都在终值 (1±tol) 内的最早代
        final = y[-1]
        ok = np.abs(y - final) <= conv_tol * abs(final)
        i_conv = int(np.argmax(np.cumprod(ok[::-1])[::-1] > 0))
        info[name] = (i_conv, float(final))
        ax.plot([i_conv], [y[i_conv]], "o", color=c, markersize=ptx(5, "pt"),
                markeredgecolor="white", markeredgewidth=ptx(0.8, "lw"), zorder=4)
        # 偏移方向按**几何**定，不按序号。原写法 xytext=(0, 9 + 9*k) 把
        # "第二条曲线"的标签一律推得更高——第二条在下方时，它的标签会穿过
        # 上面那条曲线，落到比上面那条的标签**还高**的位置。cn 档靠彩色
        # 勉强救回绑定，nature 档 text_color() 返黑字，两串黑字挤进同一条
        # 视觉带，读者按"标签越高 = 曲线越高"读正好读反——而图题写的正是
        # 谁比谁早收敛。不要靠"把 k 的顺序倒过来"绕：那只对这组数据成立。
        _oth = [np.asarray(o[1], dtype=float)[min(i_conv, len(o[1]) - 1)]
                for j, o in enumerate(curves) if j != k]
        _dy = 10 if all(y[i_conv] >= v for v in _oth) else -12
        _below.append(_dy < 0)
        ax.annotate(f"{i_conv} 代收敛", xy=(i_conv, y[i_conv]),
                    xytext=(0, _dy), textcoords="offset points",
                    ha="center", va="bottom" if _dy > 0 else "top",
                    fontsize=ptx(6.5), color=text_color(c))
        lab = end_label(ax, it[-1], final, f" {name} {final:.4g}", c)
        lab.set_va(va_of[k])
    if any(_below):
        # 最低那条曲线的收敛标签放在它下方，而它本来就贴着轴底——不腾地方
        # 的话标签会压在 x 刻度上（实测「84 代收敛」正好盖住刻度「80」）。
        # 往下扩一档轴限腾真空带，而不是把标签压上去：这与 stat_box 的
        # expand_axes 是同一条政策。
        _lo, _hi = ax.get_ylim()
        ax.set_ylim(_lo - 0.10 * (_hi - _lo), _hi)
    ax.set_xlim(-0.02 * n_max, 1.2 * n_max)   # 右侧留线端标签位，左不出负代
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig._ff_stats = {f"{k}_conv": v[0] for k, v in info.items()} | \
                    {f"{k}_final": v[1] for k, v in info.items()}
    return fig, ax, info


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(14)
    it = np.arange(120)
    raw1 = 5.2 + 8 * np.exp(-it / 18) + rng.normal(0, 0.06, len(it))
    raw2 = 5.0 + 8 * np.exp(-it / 9) + rng.normal(0, 0.04, len(it))
    # 直接喂每代原始值，mode="min" 自动转 best-so-far
    fig, ax, info = convergence_curves(
        [("标准 GA", raw1, PALETTE[0]), ("改进 GA", raw2, "#C97B84")],
        ylabel="总成本（万元）")
    gain = (info["标准 GA"][1] - info["改进 GA"][1]) / info["标准 GA"][1]
    ax.set_title(f"改进 GA 提前 {info['标准 GA'][0] - info['改进 GA'][0]} 代收敛，"
                 f"终值优 {gain:.1%}", fontsize=ptx(9.5))
    stat_box(ax, ["种群 100，交叉 0.8 / 变异 0.05",
                  f"收敛判据：相对变化 < 0.1%",
                  f"终值 {info['标准 GA'][1]:.3f} vs {info['改进 GA'][1]:.3f}"],
             loc="upper right", fontsize=ptx(6.5))
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "algo_convergence"))
    print("algo_convergence: OK")
