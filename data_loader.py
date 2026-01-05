import pandas as pd
from pathlib import Path
import kagglehub


def load_datasets():
    datasets = {}

    # --- S&P500 (equal-weight) ---
    path = kagglehub.dataset_download("wmcginn/sp500-csv")
    returns = pd.read_csv(f"{path}/returns.csv")
    returns["Date"] = pd.to_datetime(returns["Date"], errors="coerce")
    returns = returns.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    sp500_ret = returns.mean(axis=1)                  # equal-weight return
    sp500_price = (1.0 + sp500_ret).cumprod()         # wealth index od 1.0
    sp500_prices_df = sp500_price.reset_index(name="Price")
    sp500_prices_df["Date"] = pd.to_datetime(sp500_prices_df["Date"])
    datasets["S&P 500 (equal-weight)"] = sp500_prices_df.sort_values("Date").reset_index(drop=True)

    # --- Nikkei 225 (daily -> uzupełnij -> monthly last) ---
    path = kagglehub.dataset_download("ylmzasel/nikkei-225-daily-index-20202025")
    csv_path = next(Path(path).rglob("NIKKEI225.csv"))
    nik = pd.read_csv(csv_path)

    nik["observation_date"] = pd.to_datetime(nik["observation_date"], errors="coerce")
    nik = (nik.dropna(subset=["observation_date"])
              .sort_values("observation_date")
              .set_index("observation_date"))

    nik = nik.asfreq("B")  # business days
    nik["NIKKEI225"] = (nik["NIKKEI225"]
                        .interpolate(method="time", limit_direction="both")
                        .ffill()
                        .bfill())

    nik_m = (nik.resample("M")["NIKKEI225"]
                .last()
                .dropna()
                .reset_index()
                .rename(columns={"observation_date": "Date", "NIKKEI225": "Price"}))

    datasets["Nikkei 225"] = nik_m.sort_values("Date").reset_index(drop=True)

    # --- Crypto (BTC, ETH) daily -> monthly last ---
    path = kagglehub.dataset_download("sudalairajkumar/cryptocurrencypricehistory")
    data_dir = Path(path)

    for name, fname in [("AAVE", "coin_Aave.csv"),
            ("BNB",  "coin_BinanceCoin.csv"),
            ("BTC",  "coin_Bitcoin.csv"),
            ("ADA",  "coin_Cardano.csv"),
            ("LINK", "coin_ChainLink.csv"),
            ("ATOM", "coin_Cosmos.csv"),
            ("CRO",  "coin_CryptocomCoin.csv"),
            ("DOGE", "coin_Dogecoin.csv"),
            ("EOS",  "coin_EOS.csv"),
            ("ETH",  "coin_Ethereum.csv"),
            ("MIOTA","coin_Iota.csv"),
            ("LTC",  "coin_Litecoin.csv"),
            ("XMR",  "coin_Monero.csv"),
            ("XEM",  "coin_NEM.csv"),
            ("DOT",  "coin_Polkadot.csv"),
            ("SOL",  "coin_Solana.csv"),
            ("XLM",  "coin_Stellar.csv"),
            ("USDT", "coin_Tether.csv"),
            ("TRX",  "coin_Tron.csv"),
            ("USDC", "coin_USDCoin.csv"),
            ("UNI",  "coin_Uniswap.csv"),
            ("WBTC", "coin_WrappedBitcoin.csv"),
            ("XRP",  "coin_XRP.csv"),
        ]:
        csvp = next(data_dir.rglob(fname))
        df = (pd.read_csv(csvp, usecols=["Date", "Close"])
                .rename(columns={"Close": "Price"}))
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

        df_m = (df.dropna(subset=["Date", "Price"])
                  .sort_values("Date")
                  .set_index("Date")
                  .resample("M")["Price"]
                  .last()
                  .dropna()
                  .reset_index())

        datasets[name] = df_m.sort_values("Date").reset_index(drop=True)

    # --- EURUSD monthly last (close) ---
    path = kagglehub.dataset_download("konradb/foreign-exchange-rates-2003-2021")
    csv_path = next(Path(path).rglob("EURUSD_series.csv"))
    raw = pd.read_csv(csv_path)

    eurusd = (raw.assign(Date=pd.to_datetime(raw["Date"], utc=True, errors="coerce").dt.tz_convert(None))
                .dropna(subset=["Date"])
                .set_index("Date")["EURUSD_Close"]
                .resample("M").last()
                .dropna()
                .reset_index()
                .rename(columns={"EURUSD_Close": "Price"}))

    datasets["EURUSD"] = eurusd.sort_values("Date").reset_index(drop=True)

    # --- GOLD monthly (jak trzeba -> monthly last) ---
    path = kagglehub.dataset_download("tunguz/gold-prices")
    csv_path = next(Path(path).rglob("monthly_csv.csv"))
    gold = pd.read_csv(csv_path)

    date_col = "Date" if "Date" in gold.columns else next(c for c in gold.columns if "date" in c.lower())
    price_col = "Price" if "Price" in gold.columns else (
        next((c for c in gold.columns if "price" in c.lower()), gold.columns[1])
    )

    gold = gold[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "Price"})
    gold["Date"] = pd.to_datetime(gold["Date"], errors="coerce")
    gold["Price"] = pd.to_numeric(gold["Price"], errors="coerce")

    gold_m = (gold.dropna(subset=["Date", "Price"])
                  .sort_values("Date")
                  .set_index("Date")
                  .resample("M")["Price"]
                  .last()
                  .dropna()
                  .reset_index())

    datasets["GOLD"] = gold_m.sort_values("Date").reset_index(drop=True)

    return datasets
