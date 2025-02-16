from pathlib import Path

import pandas as pd


class VaccinationCube:
    """
        A hyper-cube that specifies the minimum and maximum vaccination rates
        for each age group.

        TODO: add the discrete points that were present in the vaccination programs.
               these are vectors with 100 dimensions that allow use to explore the efficacy
               of each manually-specified vaccination program.
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
        Read the vaccination rates for all age groups.
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
