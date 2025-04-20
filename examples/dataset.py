import polars as pl

import pathlib
import datetime

dates = pl.select(
    pl.date_range(
        start=datetime.date(2020, 1, 1), end=datetime.date(2025, 1, 1), interval="1mo"
    ).alias("date")
)
print(dates)
dates = dates.with_columns(
    [
        pl.format(
            "https://www.emi.ea.govt.nz/Wholesale/Datasets/DispatchAndPricing/FinalEnergyPrices/ByMonth/{}_FinalEnergyPrices.csv",
            pl.col("date").dt.strftime("%Y%m"),
        ).alias("url")
    ]
)

poc = "BEN2201"

home = pathlib.Path("data")
home.mkdir(exist_ok=True)
months = dates["date"].to_list()
urls = dates["url"].to_list()

dataset = []
for month, url in zip(months, urls):
    if not (home / f"{month}.parquet").exists():
        data = pl.read_csv(url)
        print(data.head())
        data.write_parquet(home / f"{month}.parquet")

    data = pl.read_parquet(home / f"{month}.parquet")

    data = data.with_columns(
        (
            pl.col("TradingDate")
            .str.to_datetime()
            .dt.replace_time_zone("Pacific/Auckland")
            + (pl.col("TradingPeriod") - 1) * pl.duration(minutes=30)
        ).alias("datetime")
    )
    data = data.filter(pl.col("PointOfConnection") == poc)
    dataset.append(data)
    print(data.shape)

dataset = pl.concat(dataset)
dataset.write_parquet(home / "final.parquet")
