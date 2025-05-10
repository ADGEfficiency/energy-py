import datetime

import polars as pl
import pytest

from energypy.dataset import download_electricity_prices


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path


def test_download_electricity_prices(tmp_data_dir):
    """Test that we can download electricity prices to a temporary directory."""
    # Call the download function with a small date range
    result = download_electricity_prices(
        data_dir=tmp_data_dir, verbose=True, end_date=datetime.date(2020, 2, 1)
    )

    # Check that the file was created
    assert result.exists()
    assert result == tmp_data_dir / "final.parquet"

    # Test that we can load the data
    df = pl.read_parquet(result)
    assert df.shape[0] > 0

