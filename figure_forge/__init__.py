"""figure-forge 公共命名空间：稳定导入名。

新代码统一写 ``from figure_forge import ...``；``from core import ...``
是兼容期旧导入，core 一并打包、继续可用。这里**不复制**公开名清单，
直接重导出 core.__all__——core 增删公开名时自动跟随，
tests/test_packaging.py 钉住两边一致。
"""
from ._version import __version__
from core import *                     # noqa: F401,F403
from core import __all__ as _core_all

__all__ = [*_core_all, "__version__"]
