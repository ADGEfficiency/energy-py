import datetime
import pathlib

import polars as pl


def download_electricity_prices(
    start_date: datetime.date = datetime.date(2020, 1, 1),
    end_date: datetime.date = datetime.date(2025, 1, 1),
    poc: str = "BEN2201",
    data_dir: pathlib.Path = pathlib.Path("data"),
    verbose: bool = False,
) -> pathlib.Path:
    """
    Download electricity price data from the New Zealand Electricity Authority.

    Args:
        start_date: Start date for data download
        end_date: End date for data download
        poc: Point of Connection code
        data_dir: Directory to save data files
        verbose: Whether to print progress information

    Returns:
        Path to the final parquet file containing all data
    """
    data_dir.mkdir(exist_ok=True)
    final_file = data_dir / "final.parquet"

    # If the final file already exists, return its path
    if final_file.exists():
        return final_file

    # Generate dates and URLs
    dates = pl.select(
        pl.date_range(start=start_date, end=end_date, interval="1mo").alias("date")
    )

    if verbose:
        print(f"Downloading data from {start_date} to {end_date}")

    dates = dates.with_columns(
        [
            pl.format(
                "https://www.emi.ea.govt.nz/Wholesale/Datasets/DispatchAndPricing/FinalEnergyPrices/ByMonth/{}_FinalEnergyPrices.csv",
                pl.col("date").dt.strftime("%Y%m"),
            ).alias("url")
        ]
    )

    months = dates["date"].to_list()
    urls = dates["url"].to_list()

    dataset = []
    for month, url in zip(months, urls):
        month_file = data_dir / f"{month}.parquet"
        if not month_file.exists():
            if verbose:
                print(f"Downloading data for {month}...")
            try:
                data = pl.read_csv(url)
                data.write_parquet(month_file)
                if verbose:
                    print(f"Downloaded and saved data for {month}")
            except Exception as e:
                print(f"Error downloading data for {month}: {e}")
                continue

        data = pl.read_parquet(month_file)

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
        if verbose:
            print(f"Processed {month}: {data.shape[0]} rows")

    dataset = pl.concat(dataset)
    dataset.write_parquet(final_file)
    if verbose:
        print(f"Combined dataset saved to {final_file} with {dataset.shape[0]} rows")

    return final_file


def load_electricity_prices(
    data_dir: pathlib.Path = pathlib.Path("data"),
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Load electricity price data, downloading if necessary.

    Args:
        data_dir: Directory where data is stored
        verbose: Whether to print progress information

    Returns:
        DataFrame with electricity price data
    """
    final_file = data_dir / "final.parquet"

    if not final_file.exists():
        print("Data file not found. Downloading...")
        download_electricity_prices(data_dir=data_dir, verbose=verbose)

    data = pl.read_parquet(final_file)
    data = data.select(
        "datetime",
        pl.col("DollarsPerMegawattHour").alias("price"),
        pl.col("PointOfConnection").alias("point_of_connection"),
    )

    return data


if __name__ == "__main__":
    download_electricity_prices(verbose=True)
