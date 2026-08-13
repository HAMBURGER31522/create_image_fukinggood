"""赛时数据入口：Excel/CSV → ndarray 的最短路径。

数模数据常见坑：GBK 编码 CSV、xlsx 多表头、列名带单位。
这里只做胶水：读表 + 取列转 float 数组，清洗仍归求解代码管。
"""
from __future__ import annotations

import numpy as np


def load_table(path, sheet=0, **kwargs):
    """读 Excel(.xlsx/.xls) 或 CSV/TXT 为 DataFrame。

    CSV 依次尝试 utf-8-sig / gbk / utf-8 编码（覆盖 Excel 导出与
    国内平台下载的两大编码族）。kwargs 透传 pandas。
    """
    import pandas as pd
    p = str(path)
    if p.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p, sheet_name=sheet, **kwargs)
    last_err = None
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(p, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
    raise UnicodeDecodeError(
        last_err.encoding, last_err.object, last_err.start, last_err.end,
        f"{p} 不是 utf-8/gbk 编码，请确认导出格式")


def as_1d(x, col=None) -> np.ndarray:
    """任意来源 → 一维 float ndarray。

    x 可为 DataFrame（配 col 取列，col 支持列名或位置）、Series、
    ndarray、list。非数值单元格转 NaN 不抛错，由调用方决定处理。
    """
    if col is not None:
        x = x[col] if not isinstance(col, int) else x.iloc[:, col]
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    arr = np.asarray(x)
    if arr.dtype == object or arr.dtype.kind in "US":
        import pandas as pd
        arr = pd.to_numeric(pd.Series(arr.ravel()), errors="coerce").to_numpy()
    return np.asarray(arr, dtype=float).ravel()
