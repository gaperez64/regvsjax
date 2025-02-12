import pickle
from datetime import date
from pathlib import Path
import pytest

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate
from epidem_opt.test.conftest import get_test_root


def _compare_trajectories_(ref_trajectory, actual_trajectory):
    assert len(ref_trajectory) == len(actual_trajectory), "Error, different simulation lengths."
    for day_nr, (ref_day, actual_day) in enumerate(zip(ref_trajectory, actual_trajectory)):
        assert len(ref_day) == len(actual_day), f"Error, different number of compartments on day {day_nr}."
        for i, (ref_compartment, actual_compartment) in enumerate(zip(ref_day, actual_day)):
            assert (ref_compartment == actual_compartment).all(), f"Error, {i}-th compartments differ on day {day_nr}."
    print("All correct.")


def test_regression():
    """
        TODO: end-to-end regression test, where we check that the output is unchanged w.r.t. the reference.
            - which executables
            - which reference data

    """

    # TODO: create assert to verify that our simulator only deviates 20 units from Regina's data at most
    # TODO: create assert that is exactly aligns with the pickled reference data

    regina_reference_data_folder = get_test_root() / "test_files" / "regina_reference"
    reference_pickle_path = regina_reference_data_folder / "exact_output.pickle"
    reference_csv_path = regina_reference_data_folder / "output_regina_code.csv"

    epi_data = EpiData(config_path=regina_reference_data_folder / "config.ini",
                       epidem_data_path=regina_reference_data_folder / "epidem_data",
                       econ_data_path=regina_reference_data_folder / "econ_data",
                       qaly_data_path=regina_reference_data_folder / "qaly_data",
                       vaccination_rates_path=regina_reference_data_folder / "vaccination_rates",)
    end_data = date(year=2021, month=12, day=31)  # TODO: move into ini
    actual = simulate(m=epi_data, endDate=end_data, dropBefore=date(year=2017, month=8, day=27))

    # assert against previously stored exact output
    with open(reference_pickle_path, "rb") as f:
        pickled_reference = pickle.load(f)

    assert len(pickled_reference) == len(actual), "Error, different simulation lengths."
    for day_nr, (ref_day, actual_day) in enumerate(zip(pickled_reference, actual)):
        assert len(ref_day) == len(actual_day), f"Error, different number of compartments on day {day_nr}."
        for i, (ref_compartment, actual_compartment) in enumerate(zip(ref_day, actual_day)):
            assert (ref_compartment == actual_compartment).all(), f"Error, {i}-th compartments differ on day {day_nr}."


