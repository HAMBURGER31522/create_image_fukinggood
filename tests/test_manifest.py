"""core/manifest.py 持久 figure 台账的回归测试。

对齐规格 P2 验收标准逐条落测试：
1) QA 通过保存后恰 1 行、字段非空、sha256 与实算一致；
2) 同 id 幂等 upsert、异 id 确定排序；
3) 未过 QA / QA 失败 / 单格式落盘失败 → 台账不新增不更新且保持可解析；
4) 不传 record/manifest_path 时 save_figure 行为与返回值逐项不变；
5) 以华数杯 A 题真实 8 个图 ID 做临时目录集成测试。
"""
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core import apply_style, current_preset, new_figure, run_qa, save_figure  # noqa: E402
from core import FigureRecord                       # noqa: E402  (core 公开导出)
from core.manifest import FIELDS, complete_record, write_manifest  # noqa: E402

# journal 与 preset 是两件事：preset 管语言/纹理，journal 管交付约束。
# 少了 journal 列，同一论点在 IEEE 档与 PNAS 档下出的两张图（单栏
# 88.9mm vs 90mm）在台账里一模一样，读账的人复现不出当时的约束集。
MANIFEST_COLS = ["id", "path", "formats", "claim", "source_data",
                 "generation_script", "preset", "journal", "qa_status",
                 "sha256"]


@pytest.fixture(autouse=True)
def _styled():
    apply_style()
    yield
    plt.close("all")


def _clean_fig():
    """最小但能全绿通过 run_qa 的图：散点 + 结论句图题 + 带 bbox 注释框。"""
    rng = np.random.default_rng(7)
    fig, ax = new_figure("single")
    ax.scatter(rng.normal(size=40), rng.normal(size=40), s=18)
    ax.set_title("两组分布差异可见")
    ax.text(0.05, 0.95, "样本充分", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.4"))
    run_qa(fig)          # strict：夹具图若退化，这里直接炸
    return fig


def _record(fid, claim="证据链可复核", src=("源数据/表1.csv",)):
    return FigureRecord(id=fid, claim=claim, source_data=src,
                        generation_script=__file__)


def _read(manifest_path):
    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# --- 验收 1：QA 通过保存后恰 1 行，字段非空，哈希与实算一致 -------------

def test_qa_passed_save_writes_one_verified_row(tmp_path):
    fig = _clean_fig()
    stem = tmp_path / "figs" / "t1"
    mpath = tmp_path / "figure_manifest.csv"
    out = save_figure(fig, str(stem), record=_record("t1"),
                      manifest_path=str(mpath))
    assert len(out) == 3 and all(pathlib.Path(p).exists() for p in out)

    rows = _read(mpath)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "t1"
    assert row["claim"] == "证据链可复核"
    assert json.loads(row["source_data"]) == ["源数据/表1.csv"]
    assert row["generation_script"] == __file__
    assert row["preset"] == current_preset() != ""
    assert row["qa_status"] == "passed"

    # path：相对 POSIX 路径，指向主交付件 pdf，且可定位
    assert "\\" not in row["path"] and not pathlib.PurePosixPath(row["path"]).is_absolute()
    assert row["path"].endswith(".pdf")
    assert (mpath.parent / row["path"]).exists()

    # formats 与 sha256：逐格式与文件实算一致，不得只 hash 路径串
    assert json.loads(row["formats"]) == ["png", "svg", "pdf"]
    hashes = json.loads(row["sha256"])
    assert set(hashes) == {"png", "svg", "pdf"}
    for ext, h in hashes.items():
        assert h == _sha(tmp_path / "figs" / f"t1.{ext}")


# --- 验收 2：同 id 幂等 upsert；异 id 确定排序 ---------------------------

def test_same_id_upserts_not_appends(tmp_path):
    fig = _clean_fig()
    mpath = tmp_path / "figure_manifest.csv"
    save_figure(fig, str(tmp_path / "f" / "t1"),
                record=_record("t1", claim="旧结论"), manifest_path=str(mpath))
    save_figure(fig, str(tmp_path / "f2" / "t1"),
                record=_record("t1", claim="新结论"), manifest_path=str(mpath))
    rows = _read(mpath)
    assert len(rows) == 1
    assert rows[0]["claim"] == "新结论"
    assert rows[0]["path"].startswith("f2/")


def test_two_ids_get_deterministic_order(tmp_path):
    fig = _clean_fig()
    mpath = tmp_path / "figure_manifest.csv"
    save_figure(fig, str(tmp_path / "b"), record=_record("b_id"),
                manifest_path=str(mpath))
    save_figure(fig, str(tmp_path / "a"), record=_record("a_id"),
                manifest_path=str(mpath))
    rows = _read(mpath)
    assert len(rows) == 2
    assert [r["id"] for r in rows] == ["a_id", "b_id"]   # 插入顺序相反仍有序


# --- 验收 3：未过 QA / QA 失败 / 格式失败，台账不动且保持可解析 ---------

def test_force_save_without_qa_writes_no_row(tmp_path, monkeypatch, capsys):
    # 图从未 run_qa：force 落盘但台账必须不新增（id 复用误放行也要防，
    # 故用全新 _QA_PASSED 保证确定性）
    import core.style as style
    monkeypatch.setattr(style, "_QA_PASSED", set())
    fig, ax = new_figure("single")
    ax.scatter([0.2, 0.5, 0.8], [0.3, 0.6, 0.4], s=18)
    mpath = tmp_path / "figure_manifest.csv"
    out = save_figure(fig, str(tmp_path / "w"), force=True,
                      record=_record("bad"), manifest_path=str(mpath))
    assert all(pathlib.Path(p).exists() for p in out)
    assert not mpath.exists()          # 台账未创建
    assert "不入台账" in capsys.readouterr().out


def test_qa_pass_registry_releases_collected_figures(monkeypatch):
    """QA 资格随 Figure 生命周期结束，不能靠可复用的 id 长存。"""
    import gc
    import weakref
    import core.style as style

    style._QA_PASSED.clear()
    fig, ax = new_figure("single")
    style.mark_qa_passed(fig)
    assert len(style._QA_PASSED) == 1

    ref = weakref.ref(fig)
    plt.close(fig)
    del fig, ax
    gc.collect()

    assert ref() is None
    assert len(style._QA_PASSED) == 0


def test_force_save_keeps_existing_rows_parseable(tmp_path, monkeypatch):
    fig = _clean_fig()
    mpath = tmp_path / "figure_manifest.csv"
    save_figure(fig, str(tmp_path / "good"), record=_record("good"),
                manifest_path=str(mpath))
    before = _read(mpath)

    import core.style as style
    monkeypatch.setattr(style, "_QA_PASSED", set())
    fig2, ax2 = new_figure("single")
    ax2.scatter([0.2, 0.5, 0.8], [0.3, 0.6, 0.4], s=18)
    save_figure(fig2, str(tmp_path / "bad"), force=True,
                record=_record("bad"), manifest_path=str(mpath))
    assert _read(mpath) == before      # 一行未增、内容未变


def test_qa_failure_never_reaches_manifest(tmp_path, monkeypatch):
    import core.style as style
    monkeypatch.setattr(style, "_QA_PASSED", set())   # 防 id 复用误放行
    fig, ax = new_figure("single")
    ax.plot([0, 1, 2, 3], [0, 1, 4, 9])       # 离散少点折线 → QA 硬拒
    mpath = tmp_path / "figure_manifest.csv"
    with pytest.raises(Exception):
        run_qa(fig, strict=True)
    save_figure(fig, str(tmp_path / "x"), force=True,
                record=_record("bad"), manifest_path=str(mpath))
    assert not mpath.exists()


def test_format_failure_leaves_manifest_untouched(tmp_path, monkeypatch):
    fig = _clean_fig()
    mpath = tmp_path / "figure_manifest.csv"
    save_figure(fig, str(tmp_path / "good"), record=_record("good"),
                manifest_path=str(mpath))
    before = _read(mpath)

    import matplotlib.figure
    orig = matplotlib.figure.Figure.savefig

    def boom(self, path, **kw):
        if str(path).endswith(".pdf"):
            raise OSError("模拟 pdf 写盘失败")
        return orig(self, path, **kw)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", boom)
    fig2 = _clean_fig()
    with pytest.raises(OSError):
        save_figure(fig2, str(tmp_path / "half"), record=_record("half"),
                    manifest_path=str(mpath))
    rows = _read(mpath)
    assert rows == before              # 半途失败不得更新台账
    assert [r["id"] for r in rows] == ["good"]


# --- 验收 4：不传 record/manifest_path，行为逐项不变 ---------------------

def test_default_save_figure_unchanged(tmp_path):
    fig = _clean_fig()
    stem = tmp_path / "plain"
    out = save_figure(fig, str(stem))          # 旧调用，零新参数
    assert out == [str(stem) + ext for ext in (".png", ".svg", ".pdf")]
    assert all(pathlib.Path(p).exists() for p in out)
    # 目录里不应出现任何台账文件
    assert not list(tmp_path.rglob("*manifest*"))


def test_repeated_svg_and_pdf_exports_are_byte_stable(tmp_path):
    fig = _clean_fig()
    first = save_figure(fig, str(tmp_path / "first"), formats=("svg", "pdf"))
    second = save_figure(fig, str(tmp_path / "second"), formats=("svg", "pdf"))
    for left, right in zip(first, second):
        assert pathlib.Path(left).read_bytes() == pathlib.Path(right).read_bytes()


def test_manifest_records_qa_waiver_codes(tmp_path):
    rng = np.random.default_rng(11)
    fig, ax = new_figure("single")
    ax.scatter(np.full(200, 5.0), rng.normal(size=200), s=18)
    ax.set_title("常数轴的分布结论")
    ax.text(0.05, 0.95, "样本充分", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.4"))
    problems = run_qa(fig, strict=False, allow=("axis_slack",))
    assert not any("常数" in p for p in problems)
    assert any("常数" in p for p in getattr(fig, "_ff_qa_waived", []))

    mpath = tmp_path / "figure_manifest.csv"
    save_figure(fig, str(tmp_path / "waived"), formats=("png",),
                record=_record("waived"), manifest_path=str(mpath))
    row = _read(mpath)[0]
    assert row["qa_status"] == "waived:axis_slack"


@pytest.mark.parametrize("waiver, title, sourced, expected", [
    ("unsourced", "结果 999", {"known": 1.0}, "waived:unsourced"),
    ("overlap", "重叠标注结论", None, "waived:overlap"),
])
def test_manifest_records_legacy_qa_waiver_codes(
        tmp_path, waiver, title, sourced, expected):
    fig = _clean_fig()
    ax = fig.axes[0]
    ax.set_title(title)
    if sourced is not None:
        fig._ff_stats = sourced
    if waiver == "overlap":
        ax.annotate("A", xy=(0.5, 0.5))
        ax.annotate("B", xy=(0.5, 0.5))

    problems = run_qa(fig, strict=False, allow=(waiver,))
    assert not any(title[:2] in p or waiver in p for p in problems)
    assert getattr(fig, "_ff_qa_waived_codes") == [waiver]

    mpath = tmp_path / f"{waiver}.csv"
    save_figure(fig, str(tmp_path / waiver), formats=("png",),
                record=_record(waiver), manifest_path=str(mpath))
    assert _read(mpath)[0]["qa_status"] == expected


def test_failed_qa_revokes_previous_save_qualification(tmp_path):
    fig = _clean_fig()
    ax = fig.axes[0]
    ax.text(0.2, 0.2, "过小", fontsize=4.0)
    problems = run_qa(fig, strict=False)
    assert any("字号" in p for p in problems)
    with pytest.raises(RuntimeError):
        save_figure(fig, str(tmp_path / "stale"), formats=("png",))


def test_record_and_manifest_path_must_come_together(tmp_path):
    fig = _clean_fig()
    with pytest.raises(ValueError):
        save_figure(fig, str(tmp_path / "y"), record=_record("t"),
                    manifest_path=None)
    with pytest.raises(ValueError):
        save_figure(fig, str(tmp_path / "y"), record=None,
                    manifest_path=str(tmp_path / "m.csv"))


# --- FigureRecord 本体：不可变 + 必填校验（两侧用例） --------------------

def test_record_rejects_empty_required_fields():
    for kw in (dict(id="", claim="c", source_data=("a",), generation_script="s"),
               dict(id="i", claim="  ", source_data=("a",), generation_script="s"),
               dict(id="i", claim="c", source_data=(), generation_script="s"),
               dict(id="i", claim="c", source_data=("a",), generation_script="")):
        with pytest.raises(ValueError):
            FigureRecord(**kw)
    # source_data 允许单个字符串，规范化成元组
    r = FigureRecord(id="i", claim="c", source_data="数据/附件.xlsx",
                     generation_script="s.py")
    assert r.source_data == ("数据/附件.xlsx",)


def test_record_is_frozen():
    r = _record("t")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.claim = "改动"


# --- 原子替换：中断不得留下半行，旧台账完好 ------------------------------

def test_interrupted_write_keeps_old_manifest_intact(tmp_path, monkeypatch):
    fig = _clean_fig()
    mpath = tmp_path / "figure_manifest.csv"
    save_figure(fig, str(tmp_path / "good"), record=_record("good"),
                manifest_path=str(mpath))
    before = _read(mpath)

    import core.manifest as manifest
    real_replace = manifest.os.replace

    def dead_replace(a, b):
        raise OSError("模拟写台账中途被杀")

    monkeypatch.setattr(manifest.os, "replace", dead_replace)
    with pytest.raises(OSError):
        save_figure(fig, str(tmp_path / "again"), record=_record("again"),
                    manifest_path=str(mpath))
    monkeypatch.setattr(manifest.os, "replace", real_replace)
    assert _read(mpath) == before      # 旧台账完好、可解析、无半行


def test_complete_record_prefers_pdf_and_posix_relpath(tmp_path):
    fig = _clean_fig()
    out = save_figure(fig, str(tmp_path / "figs" / "t1"))
    done = complete_record(_record("t1"), out,
                           manifest_path=str(tmp_path / "m" / "manifest.csv"),
                           preset=current_preset())
    assert done.path == "../figs/t1.pdf"
    assert done.formats == ("png", "svg", "pdf")
    assert done.qa_status == "passed"


def test_complete_record_rejects_cross_drive_paths(tmp_path, monkeypatch):
    import core.manifest as manifest

    out = tmp_path / "deliver" / "fig.pdf"
    out.parent.mkdir()
    out.write_bytes(b"pdf")
    record = _record("cross-drive")

    def cross_drive(*args):
        raise ValueError("path is on a different drive")

    monkeypatch.setattr(manifest.os.path, "relpath", cross_drive)
    with pytest.raises(ValueError) as ei:
        complete_record(record, [str(out)],
                        manifest_path=str(tmp_path / "ledger" / "manifest.csv"),
                        preset="cn")
    msg = str(ei.value)
    assert "同一盘" in msg
    assert str(out.resolve()) in msg
    assert str((tmp_path / "ledger").resolve()) in msg


def test_two_process_manifest_writers_keep_all_rows(tmp_path):
    mpath = tmp_path / "figure_manifest.csv"
    root = pathlib.Path(__file__).resolve().parents[1]
    child = r'''
import pathlib
import sys
sys.path.insert(0, sys.argv[2])
from core.manifest import FigureRecord, write_manifest

manifest = sys.argv[1]
prefix = sys.argv[3]
for i in range(20):
    fid = f"{prefix}_{i:02d}"
    write_manifest(
        FigureRecord(id=fid, claim="并发写入", source_data=("d.csv",),
                     generation_script="child.py", path=f"{fid}.png",
                     formats=("png",), preset="cn", qa_status="passed",
                     sha256="{}"),
        manifest)
    '''
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    children = [
        subprocess.Popen([sys.executable, "-c", child, str(mpath),
                          str(root), prefix],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding="utf-8", env=env)
        for prefix in ("left", "right")
    ]
    results = [p.communicate(timeout=60) for p in children]
    assert all(p.returncode == 0 for p in children), results
    rows = _read(mpath)
    assert len(rows) == 40
    assert {row["id"] for row in rows} == {
        f"{prefix}_{i:02d}" for prefix in ("left", "right") for i in range(20)
    }


# --- 验收 5：华数杯 A 题真实 8 个 ID 的集成 ------------------------------

_EIGHT_IDS = [
    "q1_1_贯通簇几何", "q1_2_阈值扫描与稳健性", "q2_1_四档概率证据链",
    "q3_1_最小分数与拟合诊断", "q4_1_最优方案证据链", "q5_1_模型检验证据链",
    "q6_1_取向各向同性验证", "q6_2_成本场与可行域",
]


def test_eight_problem_ids_integration(tmp_path):
    # 每个图至少一个真实存在的源数据文件 + 一个真实生成脚本
    src_root = tmp_path / "求解"
    for i, fid in enumerate(_EIGHT_IDS):
        d = src_root / f"p{i}"
        d.mkdir(parents=True)
        (d / "结果.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    fig = _clean_fig()
    mpath = tmp_path / "figure_manifest.csv"
    for i, fid in enumerate(_EIGHT_IDS):
        save_figure(fig, str(tmp_path / "交付" / fid),
                    record=_record(fid, src=(str(src_root / f"p{i}" / "结果.csv"),)),
                    manifest_path=str(mpath))
    rows = _read(mpath)
    assert len(rows) == 8
    assert [r["id"] for r in rows] == _EIGHT_IDS
    for r in rows:
        assert r["claim"] and r["preset"] and r["qa_status"] == "passed"
        assert json.loads(r["formats"]) == ["png", "svg", "pdf"]
        # 源数据路径逐个可定位到真实文件
        listed = json.loads(r["source_data"])
        assert listed and all(pathlib.Path(p).exists() for p in listed)
        assert pathlib.Path(r["generation_script"]).exists()
        for ext, h in json.loads(r["sha256"]).items():
            stem = mpath.parent / "交付" / r["id"]
            assert h == _sha(f"{stem}.{ext}")


# --- 列名固定 -----------------------------------------------------------

def test_manifest_columns_are_fixed():
    assert FIELDS == tuple(MANIFEST_COLS)


# --- 台账要能区分交付约束档（journal）---------------------------------

def test_manifest_records_the_journal_profile(tmp_path):
    """台账的立身之本是可追溯。同一论点在 IEEE 档与 PNAS 档下出图，
    交付约束完全不同（单栏 88.9mm vs 90mm），而两行台账此前一模一样
    ——`preset` 都是 cn、没有 journal 列，读台账的人无法复现当时生效的
    约束集。preset 管语言/纹理，journal 管交付约束，是两件事。
    """
    import csv as _csv
    from core import apply_style, save_figure, run_qa
    from core.manifest import FigureRecord
    import matplotlib.pyplot as _plt
    mp = tmp_path / "m.csv"
    try:
        for j in ("ieee", "pnas"):
            apply_style("cn", journal=j)
            fig = _clean_fig()
            run_qa(fig)
            save_figure(fig, str(tmp_path / f"f_{j}"), formats=("png",),
                        record=FigureRecord(id=f"fig_{j}", claim="同一论点",
                                            source_data="d.csv",
                                            generation_script=__file__),
                        manifest_path=str(mp))
            _plt.close("all")
        rows = list(_csv.DictReader(mp.open(encoding="utf-8-sig")))
        assert "journal" in rows[0], f"台账没有 journal 列：{list(rows[0])}"
        got = {r["id"]: r["journal"] for r in rows}
        assert got == {"fig_ieee": "ieee", "fig_pnas": "pnas"}, got
    finally:
        apply_style("cn")


def test_manifest_journal_is_empty_when_none(tmp_path):
    """不该乱填的一侧：没指定 journal 时该列为空，不能瞎写成 preset。"""
    import csv as _csv
    from core import apply_style, save_figure, run_qa
    from core.manifest import FigureRecord
    import matplotlib.pyplot as _plt
    mp = tmp_path / "m.csv"
    apply_style("cn")
    fig = _clean_fig()
    run_qa(fig)
    save_figure(fig, str(tmp_path / "f"), formats=("png",),
                record=FigureRecord(id="fig", claim="论点",
                                    source_data="d.csv",
                                    generation_script=__file__),
                manifest_path=str(mp))
    _plt.close("all")
    row = list(_csv.DictReader(mp.open(encoding="utf-8-sig")))[0]
    assert row["journal"] == "", f"journal 该为空，实际 {row['journal']!r}"
    assert row["preset"] == "cn"
