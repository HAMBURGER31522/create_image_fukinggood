"""持久 figure manifest 台账：每张交付图一行，落盘后不开驱动即可回答
「这张图主张什么、来自哪些数据、由哪个脚本生成」。

运行时的 _ff_stats + run_qa 门禁管住"图对不对"，这里管住"图是哪来的"——
两者不是同一能力，落盘后运行时状态即消失。

设计约束（规格 P2）：

- 字段固定十列：id,path,formats,claim,source_data,generation_script,
  preset,journal,qa_status,sha256；
- 按 id 幂等 upsert，行序按 id 稳定排序，两次运行结果逐字节一致；
- 路径写相对台账的 POSIX 路径；source_data 用 JSON 数组编码；
- sha256 是「格式 → 哈希」JSON 对象，逐交付格式记录文件实算值，
  不得只 hash 路径字符串；
- 先写同目录临时文件再 os.replace 原子替换，写台账中断不留半行 CSV。
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

# journal 与 preset 是两件事：preset 管语言/纹理/字体族，journal 管交付
# 约束（栏宽、字号区间、图高）。少了它，同一论点在 IEEE 档与 PNAS 档下出的
# 两张图（单栏 88.9mm vs 90mm）在台账里一模一样，读账的人无法复现当时生效
# 的约束集——而台账的立身之本就是可追溯。
FIELDS = ("id", "path", "formats", "claim", "source_data",
          "generation_script", "preset", "journal", "qa_status", "sha256")

# 带 BOM 写出：Excel 双击打开不乱码；读取端 utf-8-sig 兼容有无 BOM
_CSV_ENCODING = "utf-8-sig"


@dataclasses.dataclass(frozen=True)
class FigureRecord:
    """一张交付图的溯源声明。

    用户只填前四项（id/claim/source_data/generation_script）；
    path/formats/preset/qa_status/sha256 由 save_figure 在全部格式
    落盘成功后回填。四个人工字段都强制非空——纯示意图也必须写明
    构造依据，留空的溯源列等于没有台账。
    """
    id: str
    claim: str
    source_data: tuple[str, ...] | str
    generation_script: str
    path: str = ""
    formats: tuple[str, ...] = ()
    preset: str = ""
    journal: str = ""
    qa_status: str = ""
    sha256: str = ""

    def __post_init__(self):
        for name in ("id", "claim", "generation_script"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"FigureRecord.{name} 不能为空")
        if isinstance(self.source_data, str):
            object.__setattr__(self, "source_data", (self.source_data,))
        src = tuple(str(s) for s in self.source_data)
        if not any(s.strip() for s in src):
            raise ValueError("FigureRecord.source_data 不能为空")
        object.__setattr__(self, "source_data", src)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def complete_record(record: FigureRecord, out_files, *,
                    manifest_path, preset: str,
                    journal: str | None = None,
                    qa_status: str = "passed") -> FigureRecord:
    """全部格式落盘成功后由 save_figure 调用：回填机器可验证的五项。

    path 指主交付件（项目约定 pdf 交付、svg 可编辑、png 供目测复核，
    故优先取 pdf）；相对台账目录、POSIX 斜杠。sha256 对每个交付格式
    分别记录文件实算值。
    """
    out = [Path(p) for p in out_files]
    primary = next((p for p in out if p.suffix == ".pdf"), out[0])
    mroot = Path(manifest_path).resolve().parent
    # relpath（而非 Path.relative_to）：图与台账常是兄弟目录，需要 ../ 语义
    try:
        rel = Path(os.path.relpath(primary.resolve(), mroot))
    except ValueError as exc:
        raise ValueError(
            "交付文件与台账必须位于同一盘符，才能写相对路径："
            f"交付={primary.resolve()}，台账目录={mroot}") from exc
    hashes = {p.suffix.lstrip("."): _sha256_file(p) for p in out}
    return dataclasses.replace(
        record,
        path=rel.as_posix(),
        formats=tuple(p.suffix.lstrip(".") for p in out),
        preset=preset,
        journal=journal or "",
        qa_status=qa_status,
        sha256=json.dumps(hashes, sort_keys=True, ensure_ascii=False),
    )


def _to_row(record: FigureRecord) -> dict:
    return {
        "id": record.id,
        "path": record.path,
        "formats": json.dumps(list(record.formats), ensure_ascii=False),
        "claim": record.claim,
        "source_data": json.dumps(list(record.source_data),
                                  ensure_ascii=False),
        "generation_script": record.generation_script,
        "preset": record.preset,
        "journal": record.journal,
        "qa_status": record.qa_status,
        "sha256": record.sha256,
    }


@contextmanager
def _manifest_lock(mpath: Path):
    """用同台账旁的独占文件锁住跨进程 read-modify-write。"""
    lock = mpath.with_name(mpath.name + ".lock")
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            time.sleep(0.002)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.unlink(lock)
        except FileNotFoundError:
            pass


def write_manifest(record: FigureRecord, manifest_path) -> Path:
    """把一条完整记录 upsert 进台账并原子落盘，返回台账路径。

    幂等：同 id 只保留最新一行。整文件重写 + os.replace，
    中断只会丢本次更新，旧台账完好可解析。
    """
    mpath = Path(manifest_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock(mpath):
        rows: list[dict] = []
        if mpath.exists():
            with open(mpath, encoding=_CSV_ENCODING, newline="") as f:
                rows = [r for r in csv.DictReader(f) if r.get("id")]
        rows = [r for r in rows if r["id"] != record.id] + [_to_row(record)]
        rows.sort(key=lambda r: r["id"])
        fd, tmp = tempfile.mkstemp(dir=str(mpath.parent),
                                   prefix=mpath.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding=_CSV_ENCODING, newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(FIELDS), restval="",
                                   extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, mpath)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return mpath
