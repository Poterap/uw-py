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

st.set_page_config(page_title="Raport", layout="wide")
st.title("Raport")

@st.cache_data(show_spinner=False)
def get_data():
    return load_datasets()

datasets = get_data()

# ======= TYLKO 3 KRYPTOWALUTY =======
crypto_assets = ["BTC", "ETH", "SOL"]
available = [a for a in crypto_assets if a in datasets]
missing = [a for a in crypto_assets if a not in datasets]

if missing:
    st.warning(
        "Brak części kryptowalut wymaganych w raporcie (pomijam): "
        + ", ".join(missing)
        + ".\n\nDodaj je w `data_loader.py` (np. SOL z datasetu crypto), jeśli chcesz komplet."
    )

if len(available) < 2:
    st.error("Za mało kryptowalut dostępnych do przygotowania raportu (min 2).")
    st.stop()

# =========================
# 1) Wprowadzenie i cel analizy
# =========================
st.markdown(
    """
## 1. Wprowadzenie i cel analizy

Niniejszy raport ma formę **prospektu analitycznego** i dotyczy wyłącznie rynku **kryptowalut**.
Analiza porównuje dwie strategie inwestycyjne w ujęciu miesięcznym:

- **Buy & Hold (B&H)** – stała ekspozycja 100% na daną kryptowalutę,
- **Momentum 12–1** – strategia trend-following long-only, która wchodzi w rynek tylko przy dodatnim sygnale trendu.

**Cel analizy:**
1. Porównać profil ryzyko–zwrot B&H i Momentum 12–1 na 3 wybranych kryptowalutach.
2. Sprawdzić, czy Momentum może ograniczać głębokie obsunięcia (drawdown) typowe dla krypto.
3. Dostarczyć ilościowych wyników i wykresów w stylu „prospektowym”, ułatwiających porównanie instrumentów.

**Zastrzeżenia (disclaimer):**
- Wyniki są historyczne i nie gwarantują przyszłych rezultatów.
- Brak kosztów transakcyjnych, spreadów, poślizgów i podatków.
- Strategie są **long-only** i rebalansowane **na koniec miesiąca**.
"""
)

# =========================
# 2) Opis danych
# =========================
st.markdown(
    """
## 2. Opis danych

W raporcie wykorzystano miesięczne szeregi cenowe (kolumny **Date** i **Price**) dla 3 kryptowalut:

- **BTC** – benchmark rynku krypto, najwyższa płynność,
- **ETH** – druga największa kapitalizacja, istotny ekosystem,
- **SOL** – przykład wysokobeta’owego aktywa z innym profilem cykliczności.

Dane pochodzą z plików CSV (Kaggle) i zostały sprowadzone do częstotliwości **miesięcznej**
poprzez wybór ostatniej dostępnej obserwacji w miesiącu (close/ostatnia wartość).

**Dlaczego miesięcznie?**
- zgodność z definicją momentum 12–1,
- porównywalność między aktywami,
- naturalny rebalancing „EOM” (end of month).
"""
)

# Zakresy danych
ranges_rows = []
for name in available:
    df = datasets[name]
    ranges_rows.append(
        {
            "Kryptowaluta": name,
            "Start": df["Date"].min().date().isoformat(),
            "Koniec": df["Date"].max().date().isoformat(),
            "Liczba miesięcy": int(df.shape[0]),
        }
    )
ranges_df = pd.DataFrame(ranges_rows)
st.dataframe(ranges_df, use_container_width=True, hide_index=True)

# Wspólny okres (część wspólna dat)
common_start = max(datasets[n]["Date"].min() for n in available)
common_end = min(datasets[n]["Date"].max() for n in available)

st.markdown(
    """
### 2.1 Okres analizy

Dla porównywalności wyników między kryptowalutami przyjęto **wspólny okres próby**
(część wspólna dat dla BTC/ETH/SOL). Dzięki temu metryki odnoszą się do tego samego horyzontu czasowego.

Możesz ręcznie ustawić zakres raportu poniżej.
"""
)

cA, cB, cC = st.columns([1.2, 1.2, 2.0])
start_d = cA.date_input("Zakres raportu: od", value=common_start.date())
end_d = cB.date_input("Zakres raportu: do", value=common_end.date())
rf_annual = cC.number_input(
    "Sharpe: rf_annual (roczna stopa wolna od ryzyka, np. 0.03 = 3%)",
    value=0.0,
    step=0.01,
    format="%.4f",
)

if start_d > end_d:
    st.warning("Uwaga: data 'od' > 'do' — koryguję 'do' = 'od'.")
    end_d = start_d

start_ts = pd.Timestamp(start_d)
end_ts = pd.Timestamp(end_d)

# =========================
# 3) Strategie i metodologia
# =========================
st.markdown(
    """
## 3. Opis strategii i metodologii

### 3.1 Buy & Hold (B&H)
Stała ekspozycja 100% na aktywo. Kapitał rośnie/spada zgodnie z miesięcznymi stopami zwrotu.

### 3.2 Momentum 12–1 (long-only)
Sygnał trendu (miesięczny):

**M(t) = (P(t−1) / P(t−12)) − 1**

Reguła:
- jeśli **M(t) > 0** → pozycja **100%**,
- jeśli **M(t) ≤ 0** → pozycja **0%** (gotówka).

Rebalancing: **koniec miesiąca**.

### 3.3 Metryki oceny
- **Final Wealth**: końcowy kapitał (start = 1.0),
- **CAGR**: średnioroczny wzrost,
- **Ann. Vol**: annualizowana zmienność miesięcznych zwrotów,
- **Sharpe**: relacja nadwyżkowego zwrotu do ryzyka (z rf_annual),
- **Max Drawdown**: maksymalne obsunięcie,
- **Test t**: test t różnicy średnich miesięcznych zwrotów (Momentum − Buy&Hold), **dwustronny**.
"""
)

# =========================
# 4) Wyniki ilościowe
# =========================
st.markdown(
    """
## 4. Wyniki ilościowe

Poniższa tabela przedstawia wyniki B&H i Momentum 12–1 dla każdej kryptowaluty.
Wartości pozostają **numeryczne**, aby sortowanie działało poprawnie.

**Jak czytać wyniki w prospekcie:**
- CAGR mówi o tempie wzrostu, ale nie o „komfortowości” drogi.
- Ann. Vol i Max Drawdown opisują ryzyko typowe dla krypto.
- Sharpe (z rf_annual) daje syntetyczną ocenę „zwrotu na jednostkę ryzyka”.
"""
)

summary_rows = []
ttest_rows = []
wide_equity = None
warnings = []

for asset in available:
    prices = datasets[asset].copy()
    prices = prices[(prices["Date"] >= start_ts) & (prices["Date"] <= end_ts)].sort_values("Date")

    if prices.shape[0] < 13:
        # Momentum 12–1 sensownie działa dopiero przy >= 13 miesiącach historii (12 lookback + 1)
        warnings.append(
            f"{asset}: zbyt krótki zakres dla Momentum 12–1 (zalecane min ~13 miesięcy). "
            "Wyniki Momentum mogą być trywialne (np. same zera)."
        )

    if prices.shape[0] < 2:
        warnings.append(f"{asset}: za mało danych w wybranym zakresie (min 2 miesiące). Pomijam.")
        continue

    bh_eq = equity_buy_hold_from_prices(prices)
    mom_eq = equity_momentum_12_1_from_prices(prices)

    bh_ret = returns_from_wealth(bh_eq)
    mom_ret = returns_from_wealth(mom_eq)

    s_bh = summarize_strategy("Buy & Hold", bh_eq, bh_ret, rf_annual=rf_annual)
    s_mom = summarize_strategy("Momentum 12–1", mom_eq, mom_ret, rf_annual=rf_annual)
    s_bh["Kryptowaluta"] = asset
    s_mom["Kryptowaluta"] = asset
    summary_rows.extend([s_bh, s_mom])

    tres = ttest_mean_diff_paired(mom_ret, bh_ret, alternative="two-sided")
    tres["Kryptowaluta"] = asset
    ttest_rows.append(tres)

    bh_s = bh_eq.set_index("Date")["Wealth"].rename(f"{asset}|BH")
    mom_s = mom_eq.set_index("Date")["Wealth"].rename(f"{asset}|MOM")
    block = pd.concat([bh_s, mom_s], axis=1)
    wide_equity = block if wide_equity is None else wide_equity.join(block, how="outer")

if warnings:
    st.warning("\n".join(warnings))

summary_raw = pd.DataFrame(summary_rows)
if summary_raw.empty:
    st.error("Brak wyników do zaprezentowania – sprawdź zakres dat.")
    st.stop()

disp = summary_raw.copy()
disp["CAGR (%)"] = disp["CAGR"] * 100.0
disp["Ann. Vol (%)"] = disp["Ann. Vol"] * 100.0
disp["Max Drawdown (%)"] = disp["Max Drawdown"] * 100.0

cols = ["Kryptowaluta", "Strategy", "Final Wealth", "CAGR (%)", "Ann. Vol (%)", "Sharpe (rf=0)", "Max Drawdown (%)"]

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

# =========================
# 5) Wykresy
# =========================
st.markdown(
    """
## 5. Wykresy

Poniższy wykres przedstawia krzywe kapitału dla BTC/ETH/SOL dla obu strategii.

**Wskazówki interpretacyjne (co koniecznie opisać w raporcie):**
- Porównaj, czy Momentum ogranicza obsunięcia w fazach spadkowych.
- Zwróć uwagę, czy Momentum „spóźnia się” w fazie odbicia (typowy trade-off).
- Oceń, czy różnice między kryptowalutami sugerują odmienny reżim rynku (trend vs whipsaw).
"""
)

wide_equity = wide_equity.sort_index()
st.line_chart(wide_equity)

# =========================
# 5.1 Test t (Student)
# =========================
st.markdown(
    """
### 5.1 Test t (Student): różnica średnich miesięcznych zwrotów (Momentum − Buy&Hold)

Test jest **sparowany** i **dwustronny**.  
`p-value` informuje, czy średnia różnica (Momentum − Buy&Hold) jest statystycznie istotna
w badanym zakresie.

**Prospektowo:** nawet przy niskim p-value należy ocenić „istotność ekonomiczną”
(wielkość różnicy vs ryzyko/drawdown).
"""
)

ttest_df = pd.DataFrame(ttest_rows)
out = ttest_df.rename(columns={
    "n": "n (wspólne miesiące)",
    "meanA": "Śr. zwrot Momentum",
    "meanB": "Śr. zwrot Buy&Hold",
    "meanDiff(A-B)": "Różnica (Mom − B&H)",
    "t_stat": "t-stat",
    "p_value": "p-value",
})

show_cols = ["Kryptowaluta", "n (wspólne miesiące)", "Śr. zwrot Momentum", "Śr. zwrot Buy&Hold", "Różnica (Mom − B&H)", "t-stat", "p-value"]

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

# =========================
# 6) Interpretacja + 7) Wnioski (prospekt)
# =========================
st.markdown(
    """
## 6. Interpretacja (checklista do opisania wyników)

**A) Porównanie strategii w obrębie jednej kryptowaluty**
- Czy Momentum podniosło CAGR względem B&H?
- Czy Momentum ograniczyło Max Drawdown?
- Czy Sharpe Momentum jest wyższy (z rf_annual)?

**B) Porównanie między kryptowalutami**
- Czy BTC zachowuje się „bardziej trendowo” niż altcoiny?
- Czy ETH/SOL mają większą zmienność i głębsze drawdowny?
- Czy Momentum częściej “wyłącza się” na bardziej reżimowych aktywach?

**C) Spójność wykresów z metrykami**
- Jeśli w metrykach widać przewagę Momentum: czy wykres pokazuje łagodniejsze spadki?
- Jeśli Momentum ma słabe wyniki: czy wykres wskazuje whipsaw (częste odwrócenia trendu)?

## 7. Wnioski (styl prospektu inwestycyjnego)

W podsumowaniu napisz (na podstawie tabel i wykresu):

1) **Wniosek ilościowy**: które krypto i która strategia ma najlepszy profil ryzyko–zwrot?
2) **Ryzyko**: gdzie drawdown jest największy i czy Momentum to redukuje?
3) **Stabilność**: czy Momentum działa podobnie na BTC/ETH/SOL, czy selektywnie?
4) **Zastrzeżenia**: koszty transakcyjne, dobór okresu, brak short, rebalancing EOM.
5) **Rekomendacje dalszej analizy**: test podokresów, rolling Sharpe, koszty, inne parametry momentum.
"""
)

st.markdown("---")
st.caption("Raport generowany dynamicznie w Streamlit dla 3 kryptowalut (BTC/ETH/SOL).")
