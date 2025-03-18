from pathlib import Path

import numpy as np
import pandas as pd


class VaccinationCube:
    """
        A hyper-cube that specifies the minimum and maximum vaccination rates
        for each age group.
    """

    def __init__(self, min_rates: dict[int, float], max_rates: dict[int, float]):
        self._min_rates = min_rates
        self._max_rates = max_rates

        # sanity check
        assert self._min_rates.keys() == self._max_rates.keys()

    def get_age_groups(self) -> set[int]:
        """
            Retrieve a set of ages for which we have vaccination rates.
        """
        return set(self._min_rates.keys())

    def get_range(self, age_group: int) -> tuple[float, float]:
        """
            Retrieve the (minimum, maximum) vaccination rate for a given age group.
        """
        return self._min_rates[age_group], self._max_rates[age_group]

    def sample(self) -> float:
        pass


def read_cube(vacc_prog_summary: Path) -> VaccinationCube:
    """
        Read the vaccination min/max-rates for all age groups.

        The specified file must have three columns: age_group, min_rate, max_rate.
    """
    df = pd.read_csv(vacc_prog_summary)

    age_col = [col for col in df.columns if col.lower() == "age group"][0]
    min_rate_col = [col for col in df.columns if col.lower() == "minimum rate"][0]
    max_rate_col = [col for col in df.columns if col.lower() == "maximum rate"][0]

    ages = set(int(age) for age in df[age_col].unique())

    # sanity check: no age group has been specified twice
    assert len(df) == len(ages)

    min_rates = dict()
    max_rates = dict()

    # collect all the rates
    for index, row in df.iterrows():
        age = int(row[age_col])
        min_vaccination_rate = float(row[min_rate_col])
        max_vaccination_rate = float(row[max_rate_col])

        min_rates[age] = min_vaccination_rate
        max_rates[age] = max_vaccination_rate

    return VaccinationCube(min_rates=min_rates, max_rates=max_rates)


def read_min_max_rates(vacc_dir: Path) -> dict:
    """
        Given a directory with vaccination programs, this will the minimum and maximum rates for
        each age group.

        Resulting dict keys:
        "Age Group", "Minimum Rate", "Maximum Rate", "CSV File Name (Minimum Rate)", "CSV File Name (Maximum Rate)"
        Each key points to a dict that maps an age {0, 1, ..., 99} onto a value.
    """
    # dictionaries to store min and max rates
    min_rates = {age: float("inf") for age in range(100)}
    max_rates = {age: float("-inf") for age in range(100)}
    min_files = {age: "" for age in range(100)}
    max_files = {age: "" for age in range(100)}
    # iterate over all CSVs
    # for filename in os.listdir(directory):
    for vacc_program_path in vacc_dir.glob("*.csv"):
        # read CSV
        df = pd.read_csv(vacc_program_path)

        # extract age group and vaccination rate columns
        try:
            age_col = [col for col in df.columns if 'age' in col.lower()][0]
            rate_col = [col for col in df.columns if 'rate' in col.lower()][0]
        except IndexError as e:
            print(f"Error when reading in vaccination program '{vacc_program_path.name}':", str(vacc_program_path))
            print(str(e))
            exit(1)

        # sanity check: every vaccination program needs to specify rates for all the age groups
        ages = set(int(age) for age in df[age_col].unique())
        assert ages == set(min_rates.keys())

        # iterate through each row to extract age and rate
        for index, row in df.iterrows():
            age = int(row[age_col])
            vaccination_rate = float(row[rate_col])

            # Obtain min rate and corresponding CSV file
            if vaccination_rate < min_rates[age]:
                min_rates[age] = vaccination_rate
                min_files[age] = vacc_program_path.name

            # Obtain max rate and corresponding CSV file
            if vaccination_rate > max_rates[age]:
                max_rates[age] = vaccination_rate
                max_files[age] = vacc_program_path.name
    # sanity check
    assert set(min_files.keys()) == set(max_files.keys()) == set(min_rates.keys()) == set(max_rates.keys())
    # create output CSV
    data = {"Age Group": list(range(100)),
            "Minimum Rate": [min_rates[age] for age in range(100)],
            "Maximum Rate": [max_rates[age] for age in range(100)],
            "CSV File Name (Minimum Rate)": [min_files[age] for age in range(100)],
            "CSV File Name (Maximum Rate)": [max_files[age] for age in range(100)]}
    return data


def get_all_vacc_programs_from_dir(vacc_dir: Path) -> dict[str, list[float]]:
    """
        Reads a directory of CSV files with vaccination programs and returns a dict that maps
        the name of the vaccination program onto the rates for each age group {0, ..., 99}.
    """
    programs = dict()
    for vacc_program_path in vacc_dir.glob("*.csv"):
        # read CSV
        df = pd.read_csv(vacc_program_path)
        vacc_program_name = vacc_program_path.stem

        # extract age group and vaccination rate columns
        try:
            age_col = [col for col in df.columns if 'age' in col.lower()][0]
            rate_col = [col for col in df.columns if 'rate' in col.lower()][0]
            efficacy_col = [col for col in df.columns if 'efficacy' in col.lower()][0]
        except IndexError as e:
            print(f"Error when reading in vaccination program '{vacc_program_path.name}':", str(vacc_program_path))
            print(str(e))
            exit(1)

        # Sanity check: all ages need to be correctly specified
        ages = set(int(age) for age in df[age_col].unique())
        assert ages == set(range(100))

        rates = np.array([rate for rate in df[rate_col]])
        assert np.all(rates >= 0) and np.all(rates <= 1)

        # Sanity check: all vaccination programs use the same vaccine.
        efficacy_vals = np.array([rate for rate in df[efficacy_col]])
        expected_efficacy_vals = np.array(get_efficacy_vector())
        # TODO: why are differences of 0.01 allowed??
        assert np.allclose(efficacy_vals, expected_efficacy_vals, atol=0.01), \
            f"Error when reading program '{vacc_program_path}'."


        programs[vacc_program_name] = rates

    return programs


def get_efficacy_vector() -> list[float]:
    """
        Get the efficacy of each vaccination program per age group.
    """
    return [
        0.5241, 0.5241, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57,
        0.57, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616,
        0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616,
        0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616, 0.616,
        0.616, 0.616, 0.616, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58,
        0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58,
        0.58, 0.58, 0.58
    ]


def get_all_vaccination_programs_from_file(vacc_programs: Path) -> dict[str, list[float]]:
    """
        Reads a CSV file with vaccination programs and returns a dict that maps
        the name of the vaccination program onto the rates for each age group {0, ..., 99}.
    """

    df = pd.read_csv(vacc_programs)
    programs = dict()
    for col in df.columns:
        programs[col] = np.array(df[col])

    return programs
