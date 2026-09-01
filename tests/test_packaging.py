"""P1 打包验收：figure_forge 公共命名空间、版本单源、wheel、安装态 import。

关键点：安装态验证绝不能只把 repo root 塞进 PYTHONPATH——那会误用源码树。
这里真实构建 wheel、隔离安装到临时目录，再从**仓库外**的工作目录起
子进程做 import 冒烟和最小出图管线（apply_style -> run_qa -> save_figure）。
"""
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import figure_forge as ff        # noqa: E402  源码树导入（repo root 在 path）
import core                      # noqa: E402


# --- 公共命名空间与版本单源 ---------------------------------------------

def test_public_reexport_matches_core():
    assert ff.__all__ == [*core.__all__, "__version__"]
    for name in core.__all__:
        assert getattr(ff, name) is getattr(core, name), name


def test_version_is_single_sourced_semver():
    from figure_forge import _version
    assert ff.__version__ == _version.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", ff.__version__)


# --- wheel 构建与安装态（模块级夹具，只构建一次） ------------------------

@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("wheel")
    target = tmp_path_factory.mktemp("install")
    smoke = tmp_path_factory.mktemp("smoke")
    # --no-build-isolation：用本环境 setuptools（>=77），离线可复现；
    # CI 在 pytest 前单独 `pip install "setuptools>=77"` 保证同一前提
    r = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "--no-build-isolation", ".", "-w", str(out_dir)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    wheels = list(out_dir.glob("figure_forge-*.whl"))
    assert len(wheels) == 1, wheels
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps",
         "--no-cache-dir", "--target", str(target), str(wheels[0])],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    # 就地构建会掉 build/ 与 egg-info，清掉别弄脏工作树
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    for egg in ROOT.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)
    porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                               capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    return {"wheel": wheels[0], "target": target, "smoke": smoke,
            "porcelain": porcelain}


def test_wheel_contains_public_and_compat_packages(installed):
    with zipfile.ZipFile(installed["wheel"]) as z:
        names = z.namelist()
    for member in ("figure_forge/__init__.py", "figure_forge/_version.py",
                   "core/__init__.py", "core/style.py", "core/qa.py"):
        assert member in names, member


def _run_installed(installed, code):
    env = dict(os.environ, PYTHONPATH=str(installed["target"]))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(installed["smoke"]),
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_installed_import_smoke_outside_repo(installed):
    out = _run_installed(
        installed,
        "import figure_forge as ff; print(ff.__version__); "
        "print(ff.apply_style.__module__)")
    version, module, *_ = out.strip().splitlines()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert module == "core.style"      # 确实解析到安装副本，不是 repo root


def test_installed_compat_import_core(installed):
    out = _run_installed(installed, "from core import apply_style; "
                                    "print(apply_style.__module__)")
    assert out.strip() == "core.style"


def test_installed_minimal_figure_pipeline_and_clean_source_tree(installed):
    # 全 ASCII：CI 无中文字体也能走通同一管线
    code = """
from figure_forge import apply_style, new_figure, run_qa, save_figure
apply_style()
fig, ax = new_figure("single")
ax.scatter([(i % 10) / 9 for i in range(40)],
           [((i * 7) % 23) / 22 for i in range(40)], s=18)
ax.set_title("smoke: two groups differ")
ax.text(0.05, 0.95, "ok", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.4"))
run_qa(fig)
out = save_figure(fig, "smoke_fig")
print("FILES")
print("\\n".join(out))
"""
    lines = _run_installed(installed, code).strip().splitlines()
    files = lines[lines.index("FILES") + 1:]
    assert files == [f"smoke_fig{ext}" for ext in (".png", ".svg", ".pdf")]
    for p in files:
        assert (installed["smoke"] / p).exists()
    # 源码树 0 个新增文件
    now = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    assert now == installed["porcelain"]


# --- CI 工作流：矩阵覆盖声明版本区间，且做安装态冒烟 ---------------------

def test_ci_matrix_covers_declared_python_floor():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'requires-python\s*=\s*">=([\d.]+)"', pyproject)
    assert m, "pyproject.toml 必须声明 requires-python 下限"
    floor = m.group(1)
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert floor in ci                      # 最低版本进矩阵
    assert "pip install ." in ci            # 先安装，不是 PYTHONPATH 指源码树
    assert "import figure_forge" in ci      # 仓库外 import 冒烟
    assert "pytest" in ci                   # 全测试
