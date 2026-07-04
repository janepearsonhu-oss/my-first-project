from __future__ import annotations

import os
from pathlib import Path
import sqlite3

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-us-tech")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.vq import kmeans2


FOCUS_TICKER = "AVGO"
START_DATE = "2023-01-01"
END_DATE = "2026-05-19"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ORCL", "CRM", "ADBE"]


TARGET_COMPANIES = [
    {"ticker": "AAPL", "company_name": "Apple Inc.", "sector": "Information Technology", "industry": "Technology Hardware", "focus_flag": 0},
    {"ticker": "MSFT", "company_name": "Microsoft Corporation", "sector": "Information Technology", "industry": "Systems Software", "focus_flag": 0},
    {"ticker": "NVDA", "company_name": "NVIDIA Corporation", "sector": "Information Technology", "industry": "Semiconductors", "focus_flag": 0},
    {"ticker": "AMD", "company_name": "Advanced Micro Devices, Inc.", "sector": "Information Technology", "industry": "Semiconductors", "focus_flag": 0},
    {"ticker": "AVGO", "company_name": "Broadcom Inc.", "sector": "Information Technology", "industry": "Semiconductors", "focus_flag": 1},
    {"ticker": "ORCL", "company_name": "Oracle Corporation", "sector": "Information Technology", "industry": "Application Software", "focus_flag": 0},
    {"ticker": "CRM", "company_name": "Salesforce, Inc.", "sector": "Information Technology", "industry": "Application Software", "focus_flag": 0},
    {"ticker": "ADBE", "company_name": "Adobe Inc.", "sector": "Information Technology", "industry": "Application Software", "focus_flag": 0},
]


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_CANDIDATES = [
    SCRIPT_DIR,
    SCRIPT_DIR / " ",
    Path(" "),
]

PROJECT_ROOT = next(
    (
        path
        for path in PROJECT_CANDIDATES
        if (path / "data" / "processed").exists() or (path / "data" / "raw").exists()
    ),
    SCRIPT_DIR,
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = SCRIPT_DIR / "us_tech_stock_analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = OUTPUT_DIR / "us_tech_stock_analysis.sqlite"


def save_csv(frame: pd.DataFrame, file_name: str) -> None:
    frame.to_csv(OUTPUT_DIR / file_name, index=False)


def load_csv_if_exists(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=parse_dates)
    return pd.DataFrame()


def clean_companies(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame = pd.DataFrame(TARGET_COMPANIES)

    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["company_name"] = data["company_name"].astype(str).str.strip()
    data["focus_flag"] = (data["ticker"] == FOCUS_TICKER).astype(int)
    return (
        data[["ticker", "company_name", "sector", "industry", "focus_flag"]]
        .drop_duplicates(subset=["ticker"])
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def clean_prices(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.tz_localize(None)

    for column in [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividend_amount",
        "split_coefficient",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = (
        data.drop_duplicates(subset=["ticker", "trade_date"])
        .sort_values(["ticker", "trade_date"])
        .reset_index(drop=True)
    )

    grouped = data.groupby("ticker", group_keys=False)
    if "daily_return" not in data.columns:
        data["daily_return"] = grouped["adjusted_close"].pct_change()
    if "cumulative_return" not in data.columns:
        data["cumulative_return"] = grouped["adjusted_close"].transform(lambda s: (s / s.iloc[0]) - 1)
    if "rolling_20d_volatility" not in data.columns:
        data["rolling_20d_volatility"] = grouped["daily_return"].transform(
            lambda s: s.rolling(window=20, min_periods=20).std() * np.sqrt(252.0)
        )
    if "volume_change" not in data.columns:
        data["volume_change"] = grouped["volume"].pct_change()
    if "dollar_volume" not in data.columns:
        data["dollar_volume"] = data["adjusted_close"] * data["volume"]

    return data


def clean_macro_observations(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["series_id"] = data["series_id"].astype(str).str.upper().str.strip()
    data["observation_date"] = pd.to_datetime(data["observation_date"])
    data["observation_value"] = pd.to_numeric(data["observation_value"], errors="coerce")
    return data.drop_duplicates(subset=["series_id", "observation_date"]).reset_index(drop=True)


def load_project_data() -> dict[str, pd.DataFrame]:
    companies = load_csv_if_exists(PROCESSED_DATA_DIR / "companies.csv")
    if companies.empty:
        companies = load_csv_if_exists(RAW_DATA_DIR / "us_tech_stock_company_reference.csv")

    prices = load_csv_if_exists(PROCESSED_DATA_DIR / "daily_prices_processed.csv", parse_dates=["trade_date"])
    if prices.empty:
        prices = load_csv_if_exists(RAW_DATA_DIR / "us_tech_stock_prices_raw.csv", parse_dates=["trade_date"])

    macro_series = load_csv_if_exists(PROCESSED_DATA_DIR / "macro_series.csv")
    macro_observations = load_csv_if_exists(
        PROCESSED_DATA_DIR / "macro_observations.csv",
        parse_dates=["observation_date"],
    )
    if macro_observations.empty:
        macro_observations = load_csv_if_exists(
            RAW_DATA_DIR / "us_tech_stock_macro_raw.csv",
            parse_dates=["observation_date"],
        )

    optional_tables = {
        "filings": load_csv_if_exists(PROCESSED_DATA_DIR / "filings.csv"),
        "fundamentals": load_csv_if_exists(PROCESSED_DATA_DIR / "fundamentals.csv"),
        "scraped_quote_metrics": load_csv_if_exists(PROCESSED_DATA_DIR / "scraped_quote_metrics.csv"),
        "scraped_news": load_csv_if_exists(PROCESSED_DATA_DIR / "scraped_news.csv"),
    }

    companies = clean_companies(companies)
    prices = clean_prices(prices)
    macro_observations = clean_macro_observations(macro_observations)

    if macro_series.empty:
        macro_series = pd.DataFrame(
            [
                {"series_id": "FEDFUNDS", "series_name": "Federal Funds Rate", "frequency": "Monthly", "source": "FRED"},
                {"series_id": "VIXCLS", "series_name": "CBOE Volatility Index", "frequency": "Daily", "source": "FRED"},
            ]
        )

    data = {
        "companies": companies,
        "daily_prices": prices,
        "macro_series": macro_series,
        "macro_observations": macro_observations,
    }
    data.update(optional_tables)
    return data


def create_sqlite_database(data: dict[str, pd.DataFrame]) -> None:
    schema_sql = """
    PRAGMA foreign_keys = OFF;

    DROP VIEW IF EXISTS vw_avgo_vs_peers;
    DROP VIEW IF EXISTS vw_monthly_returns;

    DROP TABLE IF EXISTS scraped_news;
    DROP TABLE IF EXISTS scraped_quote_metrics;
    DROP TABLE IF EXISTS fundamentals;
    DROP TABLE IF EXISTS filings;
    DROP TABLE IF EXISTS macro_observations;
    DROP TABLE IF EXISTS macro_series;
    DROP TABLE IF EXISTS daily_prices;
    DROP TABLE IF EXISTS companies;

    PRAGMA foreign_keys = ON;

    CREATE TABLE companies (
        ticker TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        sector TEXT,
        industry TEXT,
        focus_flag INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE daily_prices (
        ticker TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adjusted_close REAL,
        volume INTEGER,
        dividend_amount REAL,
        split_coefficient REAL,
        daily_return REAL,
        cumulative_return REAL,
        rolling_20d_volatility REAL,
        volume_change REAL,
        dollar_volume REAL,
        PRIMARY KEY (ticker, trade_date),
        FOREIGN KEY (ticker) REFERENCES companies (ticker)
    );

    CREATE TABLE macro_series (
        series_id TEXT PRIMARY KEY,
        series_name TEXT,
        frequency TEXT,
        source TEXT
    );

    CREATE TABLE macro_observations (
        series_id TEXT NOT NULL,
        observation_date TEXT NOT NULL,
        observation_value REAL,
        PRIMARY KEY (series_id, observation_date),
        FOREIGN KEY (series_id) REFERENCES macro_series (series_id)
    );
    """

    with sqlite3.connect(DB_PATH) as connection:
        connection.executescript(schema_sql)
        data["companies"].to_sql("companies", connection, if_exists="append", index=False)
        data["daily_prices"].to_sql("daily_prices", connection, if_exists="append", index=False)
        data["macro_series"].to_sql("macro_series", connection, if_exists="append", index=False)
        data["macro_observations"].to_sql("macro_observations", connection, if_exists="append", index=False)

        for table_name in ["filings", "fundamentals", "scraped_quote_metrics", "scraped_news"]:
            frame = data.get(table_name, pd.DataFrame())
            if not frame.empty:
                frame.to_sql(table_name, connection, if_exists="replace", index=False)

        connection.executescript(
            """
            CREATE VIEW vw_avgo_vs_peers AS
            SELECT
                c.ticker,
                c.company_name,
                c.focus_flag,
                MAX(p.cumulative_return) AS latest_cumulative_return,
                AVG(p.rolling_20d_volatility) AS average_volatility,
                AVG(p.volume) AS average_volume
            FROM companies AS c
            JOIN daily_prices AS p
                ON c.ticker = p.ticker
            GROUP BY c.ticker, c.company_name, c.focus_flag;

            CREATE VIEW vw_monthly_returns AS
            SELECT
                p.ticker,
                c.company_name,
                SUBSTR(p.trade_date, 1, 7) AS month,
                MIN(p.adjusted_close) AS min_adjusted_close,
                MAX(p.adjusted_close) AS max_adjusted_close,
                AVG(p.daily_return) AS average_daily_return,
                AVG(p.rolling_20d_volatility) AS average_volatility,
                AVG(p.volume) AS average_volume
            FROM daily_prices AS p
            JOIN companies AS c
                ON p.ticker = c.ticker
            GROUP BY p.ticker, c.company_name, SUBSTR(p.trade_date, 1, 7);
            """
        )


def build_analysis_outputs(data: dict[str, pd.DataFrame]) -> None:
    companies = data["companies"]
    prices = data["daily_prices"]
    macro = data["macro_observations"]

    analysis_frame = prices.merge(
        companies[["ticker", "company_name", "focus_flag"]],
        on="ticker",
        how="left",
    )
    analysis_frame["month"] = analysis_frame["trade_date"].dt.to_period("M").astype(str)

    performance_snapshot = (
        analysis_frame.sort_values(["ticker", "trade_date"])
        .groupby(["ticker", "company_name", "focus_flag"], as_index=False)
        .agg(
            start_date=("trade_date", "min"),
            end_date=("trade_date", "max"),
            start_close=("adjusted_close", "first"),
            end_close=("adjusted_close", "last"),
            average_daily_return=("daily_return", "mean"),
            average_volatility=("rolling_20d_volatility", "mean"),
            average_volume=("volume", "mean"),
        )
    )
    performance_snapshot["total_return"] = (
        performance_snapshot["end_close"] / performance_snapshot["start_close"]
    ) - 1
    performance_snapshot["return_volatility_ratio"] = (
        performance_snapshot["total_return"] / performance_snapshot["average_volatility"]
    )
    performance_snapshot = performance_snapshot.sort_values("total_return", ascending=False).reset_index(drop=True)

    monthly_summary = (
        analysis_frame.sort_values(["ticker", "trade_date"])
        .groupby(["ticker", "company_name", "month"], as_index=False)
        .agg(
            first_close=("adjusted_close", "first"),
            last_close=("adjusted_close", "last"),
            average_volume=("volume", "mean"),
            average_volatility=("rolling_20d_volatility", "mean"),
        )
    )
    monthly_summary["monthly_return"] = (monthly_summary["last_close"] / monthly_summary["first_close"]) - 1

    macro["month"] = macro["observation_date"].dt.to_period("M").astype(str)
    monthly_macro = (
        macro.groupby(["series_id", "month"], as_index=False)["observation_value"]
        .mean()
        .pivot(index="month", columns="series_id", values="observation_value")
        .reset_index()
    )

    monthly_analysis = monthly_summary.merge(monthly_macro, on="month", how="left")

    correlation_rows = []
    for ticker, group in monthly_analysis.groupby("ticker"):
        for macro_column in ["FEDFUNDS", "VIXCLS"]:
            if macro_column not in group:
                continue
            subset = group[["monthly_return", macro_column]].dropna()
            correlation_rows.append(
                {
                    "ticker": ticker,
                    "macro_variable": macro_column,
                    "correlation_with_monthly_return": subset["monthly_return"].corr(subset[macro_column])
                    if len(subset) > 1
                    else np.nan,
                    "usable_months": len(subset),
                }
            )
    correlation_summary = pd.DataFrame(correlation_rows)

    cluster_source = monthly_analysis[
        ["ticker", "company_name", "month", "monthly_return", "average_volatility", "average_volume", "FEDFUNDS", "VIXCLS"]
    ].dropna().copy()
    cluster_source["log_average_volume"] = np.log1p(cluster_source["average_volume"])
    feature_columns = ["monthly_return", "average_volatility", "log_average_volume", "FEDFUNDS", "VIXCLS"]

    if cluster_source.empty:
        cluster_plot_frame = pd.DataFrame()
        cluster_profile = pd.DataFrame()
        cluster_story = pd.DataFrame()
    else:
        features = cluster_source[feature_columns].copy()
        features = (features - features.mean()) / features.std(ddof=0).replace(0, 1)
        _, labels = kmeans2(features.to_numpy(), 3, minit="points", seed=42)

        cluster_plot_frame = cluster_source.copy()
        cluster_plot_frame["cluster"] = labels.astype(int)

        cluster_profile = (
            cluster_plot_frame.groupby("cluster", as_index=False)
            .agg(
                mean_monthly_return=("monthly_return", "mean"),
                mean_volatility=("average_volatility", "mean"),
                mean_volume=("average_volume", "mean"),
                mean_fedfunds=("FEDFUNDS", "mean"),
                mean_vix=("VIXCLS", "mean"),
            )
            .sort_values("cluster")
        )

        return_order = cluster_profile.sort_values("mean_monthly_return").reset_index(drop=True)
        defensive_cluster = int(return_order.iloc[0]["cluster"])
        growth_cluster = int(return_order.iloc[-1]["cluster"])

        story_rows = []
        for row in cluster_profile.itertuples(index=False):
            if row.cluster == growth_cluster:
                label = "growth months"
                interpretation = "Higher return, heavier volume, and a more aggressive market pattern."
            elif row.cluster == defensive_cluster:
                label = "defensive months"
                interpretation = "Softer returns and a more cautious monthly pattern."
            else:
                label = "balanced months"
                interpretation = "A middle pattern between the stronger and weaker months."
            story_rows.append({"cluster": int(row.cluster), "cluster_label": label, "interpretation": interpretation})
        cluster_story = pd.DataFrame(story_rows)

    with sqlite3.connect(DB_PATH) as connection:
        sqlite_peer_view = pd.read_sql_query("SELECT * FROM vw_avgo_vs_peers ORDER BY latest_cumulative_return DESC", connection)

    save_csv(companies, "companies_clean.csv")
    save_csv(prices, "prices_clean.csv")
    save_csv(macro, "macro_clean.csv")
    save_csv(performance_snapshot, "performance_snapshot.csv")
    save_csv(monthly_summary, "monthly_summary.csv")
    save_csv(monthly_macro, "monthly_macro.csv")
    save_csv(monthly_analysis, "monthly_stock_macro.csv")
    save_csv(correlation_summary, "macro_correlation_summary.csv")
    save_csv(cluster_profile, "kmeans_cluster_profile.csv")
    save_csv(cluster_story, "kmeans_cluster_story.csv")
    save_csv(cluster_plot_frame, "kmeans_cluster_assignments.csv")
    save_csv(sqlite_peer_view, "sqlite_peer_view.csv")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    for ticker, group in analysis_frame.groupby("ticker"):
        plt.plot(
            group["trade_date"],
            group["cumulative_return"],
            label=ticker,
            linewidth=3 if ticker == FOCUS_TICKER else 1.5,
            alpha=1.0 if ticker == FOCUS_TICKER else 0.75,
        )
    plt.title("Cumulative Returns of Selected U.S. Technology Stocks")
    plt.xlabel("Trade Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cumulative_returns.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=performance_snapshot,
        x="average_volatility",
        y="total_return",
        hue="ticker",
        s=100,
    )
    plt.title("Return Versus Volatility")
    plt.xlabel("Average 20-Day Volatility")
    plt.ylabel("Total Return")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "return_vs_volatility.png", dpi=160)
    plt.close()

    if not cluster_plot_frame.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=cluster_plot_frame,
            x="average_volatility",
            y="monthly_return",
            hue="cluster",
            style="ticker",
            palette="Set2",
            s=75,
        )
        plt.title("K-Means Clusters of Monthly Stock Observations")
        plt.xlabel("Average Monthly Volatility")
        plt.ylabel("Monthly Return")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "kmeans_cluster_scatter.png", dpi=160)
        plt.close()

    question_answers = pd.DataFrame(
        [
            {
                "question": "Which stock had the highest total return?",
                "answer": f"{performance_snapshot.iloc[0]['ticker']} had the highest total return.",
            },
            {
                "question": "How did AVGO compare with peers?",
                "answer": "AVGO is shown in both the performance table and the SQLite peer view for direct comparison.",
            },
            {
                "question": "Can the same data support SQL and Python analysis?",
                "answer": "Yes. The script creates SQLite tables and views first, then uses the cleaned data for visual and K-Means analysis.",
            },
        ]
    )
    save_csv(question_answers, "question_answers.csv")


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Output directory:", OUTPUT_DIR)
    print("SQLite database:", DB_PATH)

    data = load_project_data()
    create_sqlite_database(data)
    build_analysis_outputs(data)

    print("U.S. technology stock SQLite analysis workflow completed.")
    print("Main outputs are saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
