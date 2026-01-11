"""
strategy_funcs.py

Funkcje do backtestu prostych strategii na jednej serii cen:
- Buy & Hold
- Momentum 12–1 (12 miesięcy z pominięciem ostatniego miesiąca)

Interfejs jest dopasowany do Twojej aplikacji Streamlit:
    - equity_buy_hold_from_prices(prices_df) -> DataFrame[Date, Wealth]
    - equity_momentum_12_1_from_prices(prices_df) -> DataFrame[Date, Wealth]
    - returns_from_wealth(equity_df) -> DataFrame[Date, Return]
    - summarize_strategy(name, equity_df, returns_df, rf_annual=...) -> dict
    - ttest_mean_diff_paired(ret_a, ret_b, alternative="two-sided") -> dict

Wymagania: pandas, numpy. SciPy jest opcjonalne (dla p-value w t-teście).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

import numpy as np
import pandas as pd


Alternative = Literal["two-sided", "greater", "less"]


def _coerce_price_frame(
    prices_df: pd.DataFrame,
    date_col: str,
    price_col: str,
) -> pd.DataFrame:
    """Walidacja i standaryzacja wejścia: zwraca DataFrame z indexem Date i jedną kolumną Price."""
    if date_col not in prices_df.columns:
        raise KeyError(f"Brak kolumny '{date_col}' w prices_df. Dostępne: {list(prices_df.columns)}")
    if price_col not in prices_df.columns:
        raise KeyError(f"Brak kolumny '{price_col}' w prices_df. Dostępne: {list(prices_df.columns)}")

    df = prices_df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=[date_col, price_col]).sort_values(date_col)
    df = df.set_index(date_col)

    # Usuwamy duplikaty dat (zostawiamy ostatnią obserwację)
    df = df[~df.index.duplicated(keep="last")]

    return df


def equity_buy_hold_from_prices(
    prices_df: pd.DataFrame,
    date_col: str = "Date",
    price_col: str = "Price",
    start_value: float = 1.0,
) -> pd.DataFrame:
    """
    Krzywa kapitału (Wealth) dla strategii Buy & Hold na podstawie cen.

    Zwraca DataFrame z kolumnami:
      - Date
      - Wealth
    """
    df = _coerce_price_frame(prices_df, date_col, price_col)
    prices = df[price_col].astype(float)

    rets = prices.pct_change().fillna(0.0)
    wealth = float(start_value) * (1.0 + rets).cumprod()

    return wealth.rename("Wealth").reset_index().rename(columns={date_col: "Date"})


def equity_momentum_12_1_from_prices(
    prices_df: pd.DataFrame,
    date_col: str = "Date",
    price_col: str = "Price",
    start_value: float = 1.0,
) -> pd.DataFrame:
    """
    Krzywa kapitału (Wealth) dla momentum 12–1.

    Definicja sygnału momentum w chwili t:
        M_t = (P_{t-1} / P_{t-12}) - 1

    Reguła:
      - jeśli M_t > 0 -> 100% long (pozycja=1)
      - w przeciwnym razie -> gotówka (pozycja=0)

    Uwaga: strategia działa sensownie na danych miesięcznych (EOM).
    """
    df = _coerce_price_frame(prices_df, date_col, price_col)
    prices = df[price_col].astype(float)

    asset_ret = prices.pct_change()
    momentum = (prices.shift(1) / prices.shift(12)) - 1.0

    position = (momentum > 0).astype(int).fillna(0)
    strat_ret = (position * asset_ret).fillna(0.0)

    wealth = float(start_value) * (1.0 + strat_ret).cumprod()
    return wealth.rename("Wealth").reset_index().rename(columns={date_col: "Date"})


def returns_from_wealth(
    equity_df: pd.DataFrame,
    date_col: str = "Date",
    wealth_col: str = "Wealth",
    fill_first: float = 0.0,
) -> pd.DataFrame:
    """
    Zamienia krzywą kapitału (Wealth) na szereg zwrotów procentowych.

    Zwraca DataFrame: [Date, Return]
    """
    if date_col not in equity_df.columns or wealth_col not in equity_df.columns:
        raise KeyError(
            f"equity_df musi mieć kolumny '{date_col}' i '{wealth_col}'. "
            f"Dostępne: {list(equity_df.columns)}"
        )

    df = equity_df[[date_col, wealth_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[wealth_col] = pd.to_numeric(df[wealth_col], errors="coerce")

    df = df.dropna(subset=[date_col, wealth_col]).sort_values(date_col)
    df = df.set_index(date_col)
    df = df[~df.index.duplicated(keep="last")]

    rets = df[wealth_col].astype(float).pct_change()
    if fill_first is not None:
        rets = rets.fillna(float(fill_first))

    out = rets.rename("Return").reset_index().rename(columns={date_col: "Date"})
    return out


def cagr_from_wealth(
    equity_df: pd.DataFrame,
    date_col: str = "Date",
    wealth_col: str = "Wealth",
) -> float:
    """
    CAGR (Compound Annual Growth Rate) liczony z pierwszej i ostatniej wartości wealth.

    CAGR = (W_end / W_start) ** (1 / years) - 1
    """
    df = equity_df[[date_col, wealth_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[wealth_col] = pd.to_numeric(df[wealth_col], errors="coerce")
    df = df.dropna(subset=[date_col, wealth_col]).sort_values(date_col)

    if len(df) < 2:
        return np.nan

    w0 = float(df[wealth_col].iloc[0])
    w1 = float(df[wealth_col].iloc[-1])

    days = (df[date_col].iloc[-1] - df[date_col].iloc[0]).days
    years = days / 365.25

    if years <= 0 or w0 <= 0:
        return np.nan

    return (w1 / w0) ** (1.0 / years) - 1.0


def annual_vol_from_returns(
    returns_df: pd.DataFrame,
    date_col: str = "Date",
    ret_col: str = "Return",
    periods_per_year: int = 12,
    ddof: int = 1,
) -> float:
    """Roczna zmienność na podstawie zwrotów okresowych."""
    df = returns_df[[date_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.dropna(subset=[date_col, ret_col]).sort_values(date_col)

    r = df[ret_col].astype(float)
    if len(r) < 2:
        return np.nan

    return float(r.std(ddof=ddof) * np.sqrt(periods_per_year))


def sharpe_from_returns(
    returns_df: pd.DataFrame,
    date_col: str = "Date",
    ret_col: str = "Return",
    rf_annual: float = 0.0,
    periods_per_year: int = 12,
    ddof: int = 1,
) -> float:
    """
    Sharpe ratio (annualizowany) na podstawie zwrotów okresowych.

    rf_annual: roczna stopa wolna od ryzyka (np. 0.03 = 3%).
    """
    df = returns_df[[date_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.dropna(subset=[date_col, ret_col]).sort_values(date_col)

    r = df[ret_col].astype(float)
    if len(r) < 2:
        return np.nan

    rf_period = (1.0 + float(rf_annual)) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_period

    std = excess.std(ddof=ddof)
    if std == 0 or np.isnan(std):
        return np.nan

    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def max_drawdown_from_wealth(
    equity_df: pd.DataFrame,
    wealth_col: str = "Wealth",
) -> float:
    """Maksymalne obsunięcie (wartość ujemna lub 0)."""
    if wealth_col not in equity_df.columns:
        raise KeyError(f"Brak kolumny '{wealth_col}' w equity_df. Dostępne: {list(equity_df.columns)}")

    wealth = pd.to_numeric(equity_df[wealth_col], errors="coerce").astype(float)
    wealth = wealth.dropna()
    if wealth.empty:
        return np.nan

    peak = wealth.cummax()
    dd = (wealth / peak) - 1.0
    return float(dd.min())


def summarize_strategy(
    strategy_name: str,
    equity_df: pd.DataFrame,
    returns_df: Optional[pd.DataFrame] = None,
    rf_annual: float = 0.0,
    periods_per_year: int = 12,
) -> Dict[str, Any]:
    """
    Zwraca słownik metryk (pod tabelkę w Streamlit).

    Uwaga: w Twoim UI kolumna nazywa się "Sharpe (rf=0)" – zachowuję klucz,
    ale liczę Sharpe dla podanego rf_annual.
    """
    if returns_df is None:
        returns_df = returns_from_wealth(equity_df)

    final_wealth = float(pd.to_numeric(equity_df["Wealth"], errors="coerce").dropna().iloc[-1]) if len(equity_df) else np.nan

    return {
        "Strategy": strategy_name,
        "Final Wealth": final_wealth,
        "CAGR": cagr_from_wealth(equity_df),
        "Ann. Vol": annual_vol_from_returns(returns_df, periods_per_year=periods_per_year),
        "Sharpe (rf=0)": sharpe_from_returns(returns_df, rf_annual=rf_annual, periods_per_year=periods_per_year),
        "Max Drawdown": max_drawdown_from_wealth(equity_df),
    }


def _ttest_p_value_from_t(t_stat: float, dfree: int, alternative: Alternative) -> float:
    """p-value dla t-Studenta. SciPy opcjonalne; bez niego zwraca NaN."""
    try:
        from scipy.stats import t as tdist  # type: ignore
    except Exception:
        return float("nan")

    t_abs = abs(float(t_stat))

    if alternative == "two-sided":
        return float(2.0 * (1.0 - tdist.cdf(t_abs, dfree)))
    if alternative == "greater":
        # H1: mean_diff > 0
        return float(1.0 - tdist.cdf(float(t_stat), dfree))
    if alternative == "less":
        # H1: mean_diff < 0
        return float(tdist.cdf(float(t_stat), dfree))

    raise ValueError(f"Nieznane alternative={alternative!r}. Dozwolone: 'two-sided'|'greater'|'less'.")


def ttest_mean_diff_paired(
    ret_a: pd.DataFrame,
    ret_b: pd.DataFrame,
    date_col: str = "Date",
    ret_col: str = "Return",
    alternative: Alternative = "two-sided",
) -> Dict[str, Any]:
    """
    Sparowany t-test dla różnicy średnich zwrotów: d_t = rA_t - rB_t.

    ret_a, ret_b: DataFrame z kolumnami [date_col, ret_col]
    alternative:
      - "two-sided" (domyślnie)
      - "greater"   (H1: meanA-meanB > 0)
      - "less"      (H1: meanA-meanB < 0)
    """
    a = ret_a[[date_col, ret_col]].copy()
    b = ret_b[[date_col, ret_col]].copy()

    a[date_col] = pd.to_datetime(a[date_col], errors="coerce")
    b[date_col] = pd.to_datetime(b[date_col], errors="coerce")
    a[ret_col] = pd.to_numeric(a[ret_col], errors="coerce")
    b[ret_col] = pd.to_numeric(b[ret_col], errors="coerce")

    a = a.dropna(subset=[date_col, ret_col]).sort_values(date_col)
    b = b.dropna(subset=[date_col, ret_col]).sort_values(date_col)

    merged = pd.merge(
        a, b,
        on=date_col,
        how="inner",
        suffixes=("_A", "_B"),
    )

    n = int(len(merged))
    if n < 2:
        return {
            "n": n,
            "meanA": float("nan"),
            "meanB": float("nan"),
            "meanDiff(A-B)": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    d = (merged[f"{ret_col}_A"].astype(float) - merged[f"{ret_col}_B"].astype(float)).to_numpy()
    mean_diff = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    dfree = n - 1

    if sd == 0 or np.isnan(sd):
        t_stat = float("nan")
        p_value = float("nan")
    else:
        t_stat = mean_diff / (sd / np.sqrt(n))
        p_value = _ttest_p_value_from_t(t_stat, dfree, alternative=alternative)

    return {
        "n": n,
        "meanA": float(merged[f"{ret_col}_A"].astype(float).mean()),
        "meanB": float(merged[f"{ret_col}_B"].astype(float).mean()),
        "meanDiff(A-B)": mean_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }
