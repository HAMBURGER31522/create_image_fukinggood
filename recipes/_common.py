"""recipes 公共入口：路径注入 + 出图目录。每个 recipe 以

    from _common import GALLERY
    from core import ...

开头；直接运行任意 recipe 会把 demo 图写进 gallery/。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GALLERY = ROOT / "gallery"
GALLERY.mkdir(exist_ok=True)


# 样式档可由环境变量切换：`FF_PRESET=nature python recipes/run_all.py`
# 让 nature 档进回归网。此前 run_all 只跑 cn，于是"25 个模板在 nature 下
# 全部硬拒"这件事项目自身完全看不见。
import os as _os
PRESET = _os.environ.get("FF_PRESET", "cn")
