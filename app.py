import streamlit as st
import pandas as pd

from data_loader import load_datasets
from strategy_funcs import (
    equity_buy_hold_from_prices,
    equity_momentum_12_1_from_prices,
    returns_from_wealth,
    summarize_strategy,
    ttest_mean_diff_paired,
)

st.set_page_config(page_title="Piaskownica", layout="wide")
st.title("Piaskownica")

@st.cache_data(show_spinner=False)
def get_data():
    return load_datasets()

datasets = get_data()
names = list(datasets.keys())

# ---------------------------
# 1) TABELKA: wybór zakresów + checkbox
# ---------------------------
st.subheader("Wybór datasetów i zakresów")

h1, h2, h3 = st.columns([2.6, 5.4, 1.0])
h1.markdown("**Dataset**")
h2.markdown("**Zakres (od / do)**")
h3.markdown("**Pokaż**")

selected = []
ranges = {}

for name in names:
    df = datasets[name]
    dmin = df["Date"].min().date()
    dmax = df["Date"].max().date()

    c1, c2, c3 = st.columns([2.6, 5.4, 1.0])
    c1.write(name)

    dcol1, dcol2 = c2.columns(2)
    start_d = dcol1.date_input(
        "Od",
        value=st.session_state.get(f"start_{name}", dmin),
        min_value=dmin,
        max_value=dmax,
        key=f"start_{name}",
    )
    end_d = dcol2.date_input(
        "Do",
        value=st.session_state.get(f"end_{name}", dmax),
        min_value=dmin,
        max_value=dmax,
        key=f"end_{name}",
    )

    if start_d > end_d:
        end_d = start_d
        st.session_state[f"end_{name}"] = end_d

    c2.caption(f"Dostępny zakres: {dmin.isoformat()} → {dmax.isoformat()}")

    show = c3.checkbox("", value=False, key=f"show_{name}")

    ranges[name] = (start_d, end_d)
    if show:
        selected.append(name)

st.divider()

# ---------------------------
# 2) WYKRES: equity curves dla zaznaczonych
# ---------------------------
st.subheader("Krzywe kapitału (zaznaczone datasety)")

if not selected:
    st.info("Zaznacz checkbox przy co najmniej jednym datasety.")
    st.stop()

# Zostawiam tylko kontrolę ile linii rysować (to realnie pomaga),
# a krótkie etykiety są zawsze.
colA, _ = st.columns(2)
show_bh = colA.checkbox("Pokaż Buy&Hold", value=True)
show_mom = colA.checkbox("Pokaż Momentum 12–1", value=True)

if not show_bh and not show_mom:
    st.info("Zaznacz przynajmniej jedną strategię (Buy&Hold lub Momentum), żeby narysować wykres.")
    st.stop()

wide_equity = None
prepared = []   # (name, bh_eq, mom_eq, bh_ret, mom_ret)
warnings = []

for name in selected:
    prices = datasets[name].copy()

    start_d, end_d = ranges[name]
    start_ts = pd.Timestamp(start_d)
    end_ts = pd.Timestamp(end_d)

    prices = prices[(prices["Date"] >= start_ts) & (prices["Date"] <= end_ts)].copy()
    prices = prices.sort_values("Date")

    if len(prices) < 2:
        warnings.append(f"{name}: za mało danych po filtrze (min 2 obserwacje). Pomijam.")
        continue

    bh_eq = equity_buy_hold_from_prices(prices)          # Date + Wealth
    mom_eq = equity_momentum_12_1_from_prices(prices)    # Date + Wealth

    bh_ret = returns_from_wealth(bh_eq)
    mom_ret = returns_from_wealth(mom_eq)

    prepared.append((name, bh_eq, mom_eq, bh_ret, mom_ret))

    parts = []
    if show_bh:
        parts.append(bh_eq.set_index("Date")["Wealth"].rename(f"{name}|BH"))
    if show_mom:
        parts.append(mom_eq.set_index("Date")["Wealth"].rename(f"{name}|MOM"))

    if not parts:
        continue

    block = pd.concat(parts, axis=1)

    if wide_equity is None:
        wide_equity = block
    else:
        wide_equity = wide_equity.join(block, how="outer")

if warnings:
    st.warning("\n".join(warnings))

if wide_equity is None or wide_equity.shape[1] == 0:
    st.info("Po filtrach nie zostało nic do narysowania.")
    st.stop()

wide_equity = wide_equity.sort_index()
st.line_chart(wide_equity)

# ---------------------------
# 3) Parametr Sharpe (pod wykresem)
# ---------------------------
rf_annual = st.number_input(
    "Sharpe: rf_annual (roczna stopa wolna od ryzyka). Przykład: 0.03 = 3% rocznie",
    value=0.0,
    step=0.01,
    format="%.4f",
)

st.divider()

# ---------------------------
# 4) Metryki (porównanie) — NUMERYCZNE, żeby sortowanie działało
# ---------------------------
st.subheader("Metryki (porównanie)")

summary_rows = []
for name, bh_eq, mom_eq, bh_ret, mom_ret in prepared:
    s_bh = summarize_strategy("Buy & Hold", bh_eq, bh_ret, rf_annual=rf_annual)
    s_mom = summarize_strategy("Momentum 12–1", mom_eq, mom_ret, rf_annual=rf_annual)
    s_bh["Dataset"] = name
    s_mom["Dataset"] = name
    summary_rows.extend([s_bh, s_mom])

summary_raw = pd.DataFrame(summary_rows)

if summary_raw.empty:
    st.info("Brak metryk do pokazania (np. zbyt krótkie zakresy).")
else:
    # ZOSTAWIAMY LICZBY -> sortowanie będzie poprawne
    disp = summary_raw.copy()
    disp["CAGR (%)"] = disp["CAGR"] * 100.0
    disp["Ann. Vol (%)"] = disp["Ann. Vol"] * 100.0
    disp["Max Drawdown (%)"] = disp["Max Drawdown"] * 100.0

    cols = [
        "Dataset",
        "Strategy",
        "Final Wealth",
        "CAGR (%)",
        "Ann. Vol (%)",
        "Sharpe (rf=0)",
        "Max Drawdown (%)",
    ]

    st.caption(
        "Kliknij w nagłówek kolumny, żeby sortować. Teraz sortuje liczbowo (nie tekstowo)."
    )

    st.dataframe(
        disp[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Wealth": st.column_config.NumberColumn(format="%.3f"),
            "CAGR (%)": st.column_config.NumberColumn(format="%.2f"),
            "Ann. Vol (%)": st.column_config.NumberColumn(format="%.2f"),
            "Sharpe (rf=0)": st.column_config.NumberColumn(format="%.2f"),
            "Max Drawdown (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

st.divider()

# ---------------------------
# 5) Test t (Student) – zawsze two-sided (też numeryczny)
# ---------------------------
st.subheader("Test t (Student): różnica średnich miesięcznych zwrotów (Momentum − Buy&Hold)")

st.caption(
    "Test sparowany i dwustronny (*two-sided*). "
    "A = Momentum, B = Buy&Hold. meanDiff = (Mom − B&H). "
    "Zwroty są w jednostkach (0.01 = 1% miesięcznie)."
)

ttest_rows = []
for name, bh_eq, mom_eq, bh_ret, mom_ret in prepared:
    tres = ttest_mean_diff_paired(mom_ret, bh_ret, alternative="two-sided")
    tres["Dataset"] = name
    ttest_rows.append(tres)

ttest_df = pd.DataFrame(ttest_rows)

if ttest_df.empty:
    st.info("Brak t-testów do pokazania.")
else:
    out = ttest_df.copy()
    out = out.rename(columns={
        "n": "n (wspólne miesiące)",
        "meanA": "Śr. zwrot Momentum",
        "meanB": "Śr. zwrot Buy&Hold",
        "meanDiff(A-B)": "Różnica (Mom − B&H)",
        "t_stat": "t-stat",
        "p_value": "p-value",
    })

    show_cols = [
        "Dataset",
        "n (wspólne miesiące)",
        "Śr. zwrot Momentum",
        "Śr. zwrot Buy&Hold",
        "Różnica (Mom − B&H)",
        "t-stat",
        "p-value",
    ]

    st.dataframe(
        out[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Śr. zwrot Momentum": st.column_config.NumberColumn(format="%.4f"),
            "Śr. zwrot Buy&Hold": st.column_config.NumberColumn(format="%.4f"),
            "Różnica (Mom − B&H)": st.column_config.NumberColumn(format="%.4f"),
            "t-stat": st.column_config.NumberColumn(format="%.4f"),
            "p-value": st.column_config.NumberColumn(format="%.4f"),
        },
    )
