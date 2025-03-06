import pickle
from datetime import date, timedelta

import jax
from jax import numpy as jnp

import pandas as pd

from epidem_opt.src.epidata import EpiData, JaxFriendlyEpiData
from epidem_opt.src.simulator import simulate_trajectories, simulate_cost, date_to_ordinal_set
from epidem_opt.test.conftest import get_test_root


def _compare_trajectories_(ref_trajectory, actual_trajectory):
    assert len(ref_trajectory) == len(actual_trajectory), "Error, different simulation lengths."
    for day_nr, (ref_day, actual_day) in enumerate(zip(ref_trajectory, actual_trajectory)):
        assert len(ref_day) == len(actual_day), f"Error, different number of compartments on day {day_nr}."
        for i, (ref_compartment, actual_compartment) in enumerate(zip(ref_day, actual_day)):
            assert (ref_compartment == actual_compartment).all(), f"Error, {i}-th compartments differ on day {day_nr}."
    print("All correct.")


def test_regression_against_pickled_data():
    """
        Regression test that verifies that "simulate()" produces results that are exactly the same
        as those stored in a pickle file. The pickle file has been created earlier from data produced by
        "simulate()", before any refactorings.
    """

    regina_reference_data_folder = get_test_root() / "test_files" / "regina_reference"
    reference_pickle_path = regina_reference_data_folder / "exact_trajectories.pickle"

    epi_data = EpiData(config_path=regina_reference_data_folder / "config.ini",
                       epidem_data_path=regina_reference_data_folder / "epidem_data",
                       econ_data_path=regina_reference_data_folder / "econ_data",
                       qaly_data_path=regina_reference_data_folder / "qaly_data",
                       vaccination_rates_path=regina_reference_data_folder / "vaccination_rates", )
    end_date = date(year=2021, month=12, day=31)
    actual = simulate_trajectories(epi_data=epi_data,
                                   begin_date=epi_data.start_date,
                                   end_date=end_date,
                                   drop_before=date(year=2017, month=8, day=27),
                                   start_state=epi_data.start_state(saved_state_file=None, saved_date=None),
                                   enforce_invariant=True)

    # assert against previously stored exact output
    with open(reference_pickle_path, "rb") as f:
        pickled_reference = pickle.load(f)

    assert len(pickled_reference) == len(actual), "Error, different simulation lengths."
    for day_nr, (ref_day, actual_day) in enumerate(zip(pickled_reference, actual)):
        assert len(ref_day) == len(actual_day), f"Error, different number of compartments on day {day_nr}."
        for i, (ref_compartment, actual_compartment) in enumerate(zip(ref_day, actual_day)):
            assert (ref_compartment == actual_compartment).all(), f"Error, {i}-th compartments differ on day {day_nr}."


def _get_max_compartment_difference_(actual, expected):
    """
        This is the infinity norm of the difference between the two specified vectors.
        Each vector is expected to be a compartment of 100 age groups.
    """
    expected = jnp.asarray(expected)
    diff = jnp.max(jnp.abs(actual - expected))
    return diff


def test_regression_against_regina_data():
    """
        Regression test that verifies that "simulate()" produces results that differ at most
        20 units from the data produced by Regina's tool.

        More specifically, for all days, for every compartment, the value is less than 20 units away from Regina's.
    """
    regina_reference_data_folder = get_test_root() / "test_files" / "regina_reference"
    reference_csv_path = regina_reference_data_folder / "output_regina_code.csv"

    epi_data = EpiData(config_path=regina_reference_data_folder / "config.ini",
                       epidem_data_path=regina_reference_data_folder / "epidem_data",
                       econ_data_path=regina_reference_data_folder / "econ_data",
                       qaly_data_path=regina_reference_data_folder / "qaly_data",
                       vaccination_rates_path=regina_reference_data_folder / "vaccination_rates", )
    end_date = date(year=2021, month=12, day=31)
    actual = simulate_trajectories(epi_data=epi_data,
                                   begin_date=epi_data.start_date,
                                   end_date=end_date,
                                   drop_before=date(year=2017, month=8, day=27),
                                   start_state=epi_data.start_state(saved_state_file=None, saved_date=None),
                                   enforce_invariant=True)

    df = pd.read_csv(reference_csv_path, header=None)
    # NOTE: each row is a compartment. There are 100 cols per row.
    #   this means that every day in the simulation comprises 5 rows.
    print("Comparing compartment values with Reg's data")
    for (S, E, Inf, R, V, *_) in actual:
        # each row is a compartment of 100 age groups
        assert _get_max_compartment_difference_(actual=S, expected=df.iloc[0]) <= 20
        assert _get_max_compartment_difference_(actual=E, expected=df.iloc[1]) <= 20
        assert _get_max_compartment_difference_(actual=Inf, expected=df.iloc[2]) <= 20
        assert _get_max_compartment_difference_(actual=R, expected=df.iloc[3]) <= 20
        assert _get_max_compartment_difference_(actual=V, expected=df.iloc[4]) <= 20

        # advance to the next 5 rows
        df = df.iloc[5:]


def test_regression_against_expected_cost_no_jax():
    """
        Test the computed cost. This does not use Jax.
    """
    regina_reference_data_folder = get_test_root() / "test_files" / "regina_reference"

    epi_data = EpiData(config_path=regina_reference_data_folder / "config.ini",
                       epidem_data_path=regina_reference_data_folder / "epidem_data",
                       econ_data_path=regina_reference_data_folder / "econ_data",
                       qaly_data_path=regina_reference_data_folder / "qaly_data",
                       vaccination_rates_path=regina_reference_data_folder / "vaccination_rates", )

    begin_date = epi_data.start_date
    end_date = epi_data.last_burnt_date
    cost = simulate_cost(
        epi_data.vacc_rates,
        epi_data=epi_data,
        epi_state=epi_data.start_state(),
        start_date=begin_date.toordinal(),
        end_date=end_date.toordinal(),
        vacc_dates=lambda x: x in date_to_ordinal_set(epi_data.vacc_date, begin_date, end_date),
        peak_dates=lambda x: x in date_to_ordinal_set(epi_data.peak_date, begin_date, end_date),
        seed_dates=lambda x: x in date_to_ordinal_set(epi_data.seed_date, begin_date, end_date),
        birth_dates=lambda x: x in date_to_ordinal_set(epi_data.birthday, begin_date, end_date)
    )
    assert cost == 2774434816.0


def test_regression_against_expected_cost_jax():
    regina_reference_data_folder = get_test_root() / "test_files" / "regina_reference"
    reference_pickle_path = regina_reference_data_folder / "expected_gradients.pickle"

    epi_data = EpiData(config_path=regina_reference_data_folder / "config.ini",
                       epidem_data_path=regina_reference_data_folder / "epidem_data",
                       econ_data_path=regina_reference_data_folder / "econ_data",
                       qaly_data_path=regina_reference_data_folder / "qaly_data",
                       vaccination_rates_path=regina_reference_data_folder / "vaccination_rates", )

    grad_cost = jax.value_and_grad(simulate_cost)

    begin_date = epi_data.start_date
    end_date = epi_data.last_burnt_date
    actual_cost, actual_grad = grad_cost(
        epi_data.vacc_rates,
        epi_data=epi_data,
        epi_state=epi_data.start_state(),
        start_date=begin_date.toordinal(),
        end_date=end_date.toordinal(),
        vacc_dates=lambda x: x in date_to_ordinal_set(epi_data.vacc_date, begin_date, end_date),
        peak_dates=lambda x: x in date_to_ordinal_set(epi_data.peak_date, begin_date, end_date),
        seed_dates=lambda x: x in date_to_ordinal_set(epi_data.seed_date, begin_date, end_date),
        birth_dates=lambda x: x in date_to_ordinal_set(epi_data.birthday, begin_date, end_date)
    )
    assert actual_cost == 2774434816.0


def test_regression_against_expected_gradient():
    regina_reference_data_folder = get_test_root() / "test_files" / "regina_reference"
    reference_pickle_path = regina_reference_data_folder / "expected_gradients.pickle"

    epi_data = EpiData(config_path=regina_reference_data_folder / "config.ini",
                       epidem_data_path=regina_reference_data_folder / "epidem_data",
                       econ_data_path=regina_reference_data_folder / "econ_data",
                       qaly_data_path=regina_reference_data_folder / "qaly_data",
                       vaccination_rates_path=regina_reference_data_folder / "vaccination_rates", )

    grad_cost = jax.value_and_grad(simulate_cost)

    begin_date = epi_data.start_date
    end_date = epi_data.last_burnt_date
    actual_cost, actual_grad = grad_cost(
        epi_data.vacc_rates,
        epi_data=epi_data,
        epi_state=epi_data.start_state(),
        start_date=begin_date.toordinal(),
        end_date=end_date.toordinal(),
        vacc_dates=lambda x: x in date_to_ordinal_set(epi_data.vacc_date, begin_date, end_date),
        peak_dates=lambda x: x in date_to_ordinal_set(epi_data.peak_date, begin_date, end_date),
        seed_dates=lambda x: x in date_to_ordinal_set(epi_data.seed_date, begin_date, end_date),
        birth_dates=lambda x: x in date_to_ordinal_set(epi_data.birthday, begin_date, end_date)
    )

    with open(reference_pickle_path, 'rb') as ref_grad_file:
        ref_grad = pickle.load(ref_grad_file)

    assert jnp.allclose(ref_grad, actual_grad), "Error, grad mismatch"
