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
