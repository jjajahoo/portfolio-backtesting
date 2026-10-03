import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 테스트용 가짜 시세: 실제 Yahoo Finance에 접속하지 않는다.
# 코스닥 종목처럼 .KQ로만 찾을 수 있는 코드, 2015년에 상장한 종목, 없는 종목을 흉내 낸다.
KOSDAQ_ONLY = {"247540"}
LISTED_LATE = {"LATE": pd.Timestamp("2015-06-01")}
UNKNOWN = {"NOPE"}


def fake_close(symbol: str, start, end) -> pd.Series:
    base = symbol.split(".")[0]
    if base in UNKNOWN or (base in KOSDAQ_ONLY and symbol.endswith(".KS")):
        return pd.Series(dtype=float)
    if base.isdigit() and base not in KOSDAQ_ONLY and symbol.endswith(".KQ"):
        return pd.Series(dtype=float)
    dates = pd.bdate_range(start, end)
    if base in LISTED_LATE:
        dates = dates[dates >= LISTED_LATE[base]]
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    if symbol == "KRW=X":
        steps = rng.normal(0, 0.004, len(dates))
        return pd.Series(1200 * np.exp(np.cumsum(steps)), index=dates)
    steps = rng.normal(0.0003, 0.01, len(dates))
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=dates)
