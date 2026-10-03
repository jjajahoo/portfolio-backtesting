from datetime import date

import pandas as pd
import pytest

from backtester.market_data import candidates, currency_of, load_prices
from conftest import fake_close


def test_candidates():
    assert candidates("069500") == ["069500.KS", "069500.KQ"]
    assert candidates(" 0080g0 ") == ["0080G0.KS", "0080G0.KQ"]  # 영문이 섞인 새 한국 코드
    assert candidates("spy") == ["SPY"]
    assert candidates("BRK-B") == ["BRK-B"]


def test_currency_of():
    assert currency_of("069500.KS") == "KRW"
    assert currency_of("247540.KQ") == "KRW"
    assert currency_of("SPY") == "USD"


def test_kosdaq_code_falls_back_to_kq_and_reports_missing():
    data = load_prices(["069500", "247540", "nope"], date(2020, 1, 1), date(2020, 12, 31), "KRW", fetch=fake_close)
    assert data.symbols == {"069500": "069500.KS", "247540": "247540.KQ"}
    assert data.missing == ["NOPE"]
    assert data.fx_first_date is None  # 모두 원화라 환율이 필요 없다
    assert list(data.prices.columns) == ["069500", "247540"]


def test_usd_asset_converted_to_krw():
    start, end = date(2020, 1, 1), date(2020, 3, 31)
    data = load_prices(["SPY", "069500"], start, end, "KRW", fetch=fake_close)
    fx = fake_close("KRW=X", start, end)
    spy = fake_close("SPY", start, end)
    day = data.prices.index[10]
    assert data.prices.loc[day, "SPY"] == pytest.approx(spy[day] * fx[day])
    assert data.fx_first_date == fx.index[0]

    in_usd = load_prices(["SPY", "069500"], start, end, "USD", fetch=fake_close)
    kr = fake_close("069500.KS", start, end)
    assert in_usd.prices.loc[day, "069500"] == pytest.approx(kr[day] / fx[day])
    assert in_usd.prices.loc[day, "SPY"] == pytest.approx(spy[day])


def test_common_period_starts_when_every_asset_has_data():
    data = load_prices(["SPY", "LATE"], date(2014, 1, 1), date(2016, 1, 1), "USD", fetch=fake_close)
    assert data.prices.index[0] == pd.Timestamp("2015-06-01")
    assert data.first_dates["SPY"] < data.first_dates["LATE"]
    assert not data.prices.isna().any().any()


def test_mismatched_holidays_are_forward_filled():
    def fetch(symbol, start, end):
        s = fake_close(symbol, start, end)
        return s.drop(pd.Timestamp("2020-01-15")) if symbol == "069500.KS" else s

    data = load_prices(["SPY", "069500"], date(2020, 1, 1), date(2020, 1, 31), "USD", fetch=fetch)
    assert pd.Timestamp("2020-01-15") in data.prices.index
    assert not data.prices.isna().any().any()
