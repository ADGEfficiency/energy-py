import numpy as np
import pytest

from energypy.battery import Battery, Freq


def test_freq_class():
    """Test that the Freq class correctly converts between MW and MWh."""
    # Test hourly frequency (60 minutes)
    freq_60 = Freq(mins=60)
    assert freq_60.mw_to_mwh(1.0) == 1.0  # 1 MW for 1 hour = 1 MWh
    assert freq_60.mwh_to_mw(1.0) == 1.0  # 1 MWh over 1 hour = 1 MW
    
    # Test half-hourly frequency (30 minutes)
    freq_30 = Freq(mins=30)
    assert freq_30.mw_to_mwh(1.0) == 0.5  # 1 MW for 30 mins = 0.5 MWh
    assert freq_30.mwh_to_mw(0.5) == 1.0  # 0.5 MWh over 30 mins = 1 MW
    
    # Test 15-minute frequency
    freq_15 = Freq(mins=15)
    assert freq_15.mw_to_mwh(1.0) == 0.25  # 1 MW for 15 mins = 0.25 MWh
    assert freq_15.mwh_to_mw(0.25) == 1.0  # 0.25 MWh over 15 mins = 1 MW
    
    # Test string representation
    assert str(freq_60) == "Freq(mins=60)"


def test_battery_with_frequency():
    """Test that the battery correctly handles different frequencies."""
    # Create common test data
    prices = np.array([100.0] * 1000)
    features = np.ones((1000, 4))
    power_mw = 2.0
    
    # Test with 60-minute frequency (hourly)
    battery_60 = Battery(
        electricity_prices=prices,
        features=features,
        power_mw=power_mw,
        freq_mins=60
    )
    
    # When we apply 1 MW for 1 hour, we should get 1 MWh change
    initial_soc = battery_60.state_of_charge_mwh
    battery_60.step(np.array([1.0]))
    assert battery_60.state_of_charge_mwh - initial_soc == pytest.approx(1.0)
    
    # Test with 30-minute frequency
    battery_30 = Battery(
        electricity_prices=prices,
        features=features,
        power_mw=power_mw,
        freq_mins=30
    )
    
    # When we apply 1 MW for 30 mins, we should get 0.5 MWh change
    initial_soc = battery_30.state_of_charge_mwh
    battery_30.step(np.array([1.0]))
    assert battery_30.state_of_charge_mwh - initial_soc == pytest.approx(0.5)
    
    # Test with 15-minute frequency
    battery_15 = Battery(
        electricity_prices=prices,
        features=features,
        power_mw=power_mw,
        freq_mins=15
    )
    
    # When we apply 1 MW for 15 mins, we should get 0.25 MWh change
    initial_soc = battery_15.state_of_charge_mwh
    battery_15.step(np.array([1.0]))
    assert battery_15.state_of_charge_mwh - initial_soc == pytest.approx(0.25)


def test_battery_power_constraint_with_frequency():
    """Test that power constraints are respected with different frequencies."""
    # Create common test data
    prices = np.array([100.0] * 1000)
    features = np.ones((1000, 4))
    power_mw = 2.0
    
    # Test with 15-minute frequency
    battery = Battery(
        electricity_prices=prices,
        features=features,
        power_mw=power_mw,
        freq_mins=15
    )
    
    # Try to charge beyond power limit
    battery.step(np.array([3.0]))  # Should be capped at 2.0 MW
    
    # The energy change should be 2.0 MW * 0.25 h = 0.5 MWh
    assert battery.state_of_charge_mwh == pytest.approx(0.5)
    
    # Test with 30-minute frequency
    battery = Battery(
        electricity_prices=prices,
        features=features,
        power_mw=power_mw,
        freq_mins=30
    )
    
    # Try to discharge beyond power limit
    battery.step(np.array([-3.0]))  # Should be capped at -2.0 MW
    
    # The energy change should be -2.0 MW * 0.5 h = -1.0 MWh,
    # but since we started at 0, we should be at 0 (can't go negative)
    assert battery.state_of_charge_mwh == pytest.approx(0.0)


def test_backward_compatibility():
    """Test that the battery still works with default hourly frequency."""
    # Create common test data
    prices = np.array([100.0] * 1000)
    features = np.ones((1000, 4))
    power_mw = 2.0
    
    # Test with default frequency (hourly)
    battery = Battery(
        electricity_prices=prices,
        features=features,
        power_mw=power_mw,
    )
    
    # Apply 1 MW for default interval (1 hour)
    initial_soc = battery.state_of_charge_mwh
    battery.step(np.array([1.0]))
    
    # Expect 1 MWh change
    assert battery.state_of_charge_mwh - initial_soc == pytest.approx(1.0)