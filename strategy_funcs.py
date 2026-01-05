import numpy as np
import pandas as pd


def equity_buy_hold_from_prices(prices_df: pd.DataFrame,
                                date_col: str = "Date",
                                price_col: str = "Price",
                                start_value: float = 1.0) -> pd.DataFrame:
    df = prices_df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    P = df[price_col].astype(float)
    r = P.pct_change().fillna(0.0)

    wealth = start_value * (1.0 + r).cumprod()
    return wealth.rename("Wealth").reset_index().rename(columns={date_col: "Date"})


def equity_momentum_12_1_from_prices(prices_df: pd.DataFrame,
                                     date_col: str = "Date",
                                     price_col: str = "Price",
                                     start_value: float = 1.0) -> pd.DataFrame:
    df = prices_df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    P = df[price_col].astype(float)
    r = P.pct_change()

    mom = (P.shift(1) / P.shift(12)) - 1.0
    pos = (mom > 0).astype(int).fillna(0)

    strat_r = (pos * r).fillna(0.0)
    wealth = start_value * (1.0 + strat_r).cumprod()

    return wealth.rename("Wealth").reset_index().rename(columns={date_col: "Date"})


def returns_from_wealth(equity_df, date_col="Date", wealth_col="Wealth"):
    df = equity_df[[date_col, wealth_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df["Return"] = df[wealth_col].pct_change().fillna(0.0)
    return df[[date_col, "Return"]]


def cagr_from_wealth(equity_df, date_col="Date", wealth_col="Wealth"):
    df = equity_df[[date_col, wealth_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    w0 = float(df[wealth_col].iloc[0])
    w1 = float(df[wealth_col].iloc[-1])
    days = (df[date_col].iloc[-1] - df[date_col].iloc[0]).days
    years = days / 365.25

    # uwaga: jak years==0 to będzie dzielenie przez 0 -> zostawiamy jak w Twoim kodzie,
    # ale w app.py dopilnujemy, żeby nie liczyć metryk dla zbyt krótkiej próbki.
    return (w1 / w0) ** (1 / years) - 1


def annual_vol_from_returns(returns_df, date_col="Date", ret_col="Return", periods_per_year=12, ddof=1):
    df = returns_df[[date_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    r = df.sort_values(date_col)[ret_col].astype(float)
    return r.std(ddof=ddof) * np.sqrt(periods_per_year)


def sharpe_from_returns(returns_df,
                        date_col="Date",
                        ret_col="Return",
                        rf_annual=0.0,
                        periods_per_year=12,
                        ddof=1):
    df = returns_df[[date_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    r = df.sort_values(date_col)[ret_col].astype(float)

    rf_p = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_p

    std = excess.std(ddof=ddof)
    if std == 0 or np.isnan(std):
        return np.nan

    return (excess.mean() / std) * np.sqrt(periods_per_year)


def max_drawdown_from_wealth(equity_df, wealth_col="Wealth"):
    w = equity_df[wealth_col].astype(float)
    peak = w.cummax()
    dd = (w / peak) - 1.0
    return dd.min()


def ttest_mean_diff_paired(ret_a: pd.DataFrame,
                           ret_b: pd.DataFrame,
                           date_col="Date",
                           ret_col="Return",
                           alternative="two-sided"):
    a = ret_a[[date_col, ret_col]].copy()
    b = ret_b[[date_col, ret_col]].copy()
    a[date_col] = pd.to_datetime(a[date_col])
    b[date_col] = pd.to_datetime(b[date_col])

    m = a.merge(b, on=date_col, how="inner", suffixes=("_A", "_B")).sort_values(date_col)
    d = (m[f"{ret_col}_A"].astype(float) - m[f"{ret_col}_B"].astype(float)).dropna()

    n = int(d.shape[0])
    if n < 2:
        return {"n": n, "t_stat": np.nan, "p_value": np.nan}

    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    t_stat = mean_d / (sd_d / np.sqrt(n)) if sd_d != 0 else np.inf

    dfree = n - 1

    try:
        from scipy.stats import t as tdist
        if alternative == "two-sided":
            p = 2 * (1 - tdist.cdf(abs(t_stat), dfree))
        elif alternative == "greater":
            p = 1 - tdist.cdf(t_stat, dfree)
        elif alternative == "less":
            p = tdist.cdf(t_stat, dfree)
        else:
            raise ValueError("alternative must be: two-sided / greater / less")
    except Exception:
        p = np.nan

    return {
        "n": n,
        "meanA": float(m[f"{ret_col}_A"].astype(float).mean()),
        "meanB": float(m[f"{ret_col}_B"].astype(float).mean()),
        "meanDiff(A-B)": mean_d,
        "t_stat": float(t_stat),
        "p_value": float(p),
    }


def summarize_strategy(name, eq_df, ret_df, rf_annual=0.0):
    return {
        "Strategy": name,
        "Final Wealth": float(eq_df["Wealth"].iloc[-1]),
        "CAGR": float(cagr_from_wealth(eq_df)),
        "Ann. Vol": float(annual_vol_from_returns(ret_df)),
        "Sharpe (rf=0)": float(sharpe_from_returns(ret_df, rf_annual=rf_annual)),
        "Max Drawdown": float(max_drawdown_from_wealth(eq_df)),
    }
