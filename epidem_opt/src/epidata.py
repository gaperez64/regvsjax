import configparser
from datetime import date
from pathlib import Path

import jax.numpy as jnp
import pandas as pd


def load_from_csv(fname):
    df = pd.read_csv(fname, header=None)
    assert df.shape[0] == 100
    return jnp.asarray(df[0].values, dtype=jnp.float64)


class EpiData:
    def switch_program(self, program: str):
        # TODO: factor out filename.
        df = pd.read_csv(self.vaccination_rates_path / f"program_{program}.csv")
        df["CovXEff"] = df.apply(lambda row: row.iloc[1] * row.iloc[2],
                                 axis=1)
        self.vacc_rates = jnp.asarray(df["CovXEff"].values)
        return self.vacc_rates

    def __init__(self, config_path: Path,
                 epidem_data_path: Path,
                 econ_data_path: Path,
                 qaly_data_path: Path,
                 vaccination_rates_path: Path,
                 start_date: date=None):
        """
            Constructor.
        Parameters
        ----------
        start_date
            The start date of the epidemic simulation. If None, it will be retrieved from the config: Defaults/startDate
        """
        # Load parameters and constants from configuration file
        config = configparser.ConfigParser()
        config.read(config_path)
        cpars = config["Cont.Pars"]
        self.q = cpars.getfloat("q")
        self.sigma = cpars.getfloat("sigma")
        self.gamma = cpars.getfloat("gamma")
        self.omega_imm = cpars.getfloat("omegaImm")
        self.omega_vacc = cpars.getfloat("omegaVacc")
        self.delta = cpars.getfloat("delta")

        dpars = config["Disc.Pars"]
        self.seed_inf = dpars.getint("seedInf")
        self.seed_ages = jnp.asarray(range(dpars.getint("seedAgeMin"),
                                           dpars.getint("seedAgeMax")))
        self.seed_date = (dpars.getint("seedMonth"),
                          dpars.getint("seedDay"))
        self.birthday = (dpars.getint("birthMonth"),
                         dpars.getint("birthDay"))
        self.vacc_date = (dpars.getint("vaccMonth"),
                          dpars.getint("vaccDay"))
        self.peak_date = (dpars.getint("peakMonth"),
                          dpars.getint("peakDay"))

        self.no_med_care = config.getfloat("Cost.Pars", "noMedCare")

        # Defaults
        if start_date is None:
            self.start_date = date.fromisoformat(
                config.get("Defaults", "startDate"))
        else:
            self.start_date = start_date

        self.last_burnt_date = date.fromisoformat(config.get("Defaults", "lastBurntDate"))

        # contact matrix
        df = pd.read_csv(epidem_data_path / config.get("EpidemFiles", "contactMatrix"))
        assert df.shape[0] == 100
        self.contact = jnp.asarray(df.values)
        # mortality rates
        df = pd.read_csv(epidem_data_path / config.get("EpidemFiles", "mortalityRates"))
        df.drop(df.tail(1).index, inplace=True)  # Ignore last value > 100
        assert df.shape[0] == 100
        self.dailyMort = jnp.asarray(df["Value"].values) / 365
        # initial population
        df = pd.read_csv(epidem_data_path / config.get("EpidemFiles", "startPopulation"))
        df.drop(df.tail(1).index, inplace=True)  # Ignore last value > 100
        assert df.shape[0] == 100
        self.init_pop = jnp.asarray(df["Population"].values, dtype=jnp.float64)
        self.tot_pop = self.init_pop.sum()
        print(f" ****** Initial total population {self.tot_pop}")

        # Detail rates
        df = pd.read_csv(epidem_data_path / config.get("EpidemFiles", "influenzaRate"), header=None)
        assert df.shape[0] == 100
        self.influenzaRate = jnp.asarray(df[0].values, dtype=jnp.float64)
        df = pd.read_csv(epidem_data_path / config.get("EpidemFiles", "hospRate"))
        assert df.shape[0] == 100
        self.hospRate = jnp.asarray(df["Rate"].values, dtype=jnp.float64)
        df = pd.read_csv(epidem_data_path / config.get("EpidemFiles", "caseFatalityRate"))
        assert df.shape[0] == 100
        self.caseFatalityRate = jnp.asarray(df["Rate"].values,
                                            dtype=jnp.float64)
        # Costs
        self.ambulatory_costs = load_from_csv(
            econ_data_path / config.get("EconFiles", "ambulatoryCosts"))
        self.vacc_costs = load_from_csv(
            econ_data_path / config.get("EconFiles", "vaccineCost"))
        self.nomedCosts = load_from_csv(
            econ_data_path / config.get("EconFiles", "noMedicalCareCosts"))
        self.hospCosts = load_from_csv(
            econ_data_path / config.get("EconFiles", "hospCosts"))
        self.hospAmbCosts = load_from_csv(
            econ_data_path / config.get("EconFiles", "hospAmbulatoryCosts"))

        # Qalys
        self.ambulatory_qalys = load_from_csv(
            qaly_data_path / config.get("QalyFiles", "ambulatoryQalyLoss"))
        self.nomed_qalys = load_from_csv(
            qaly_data_path / config.get("QalyFiles", "noMedicalCareQalyLoss"))
        self.hosp_qalys = load_from_csv(
            qaly_data_path / config.get("QalyFiles", "hospitalizedQalyLoss"))
        self.disc_life_ex = load_from_csv(
            qaly_data_path / config.get("QalyFiles", "discountedLifeExpectancy"))

        # vaccination stats
        self.vaccination_rates_path = vaccination_rates_path
        self.switch_program(program="baseline")

    def start_state(self, saved_state: str = None, saved_date: date = None) -> tuple:
        """
            Retrieve the start state.

            The "saved state" is a filename that points to a CSV with initial conditions. The "saved date" is a date
            is ISO format.
        """
        if saved_state is None:
            start_date = self.start_date
            S = self.init_pop
            E = jnp.zeros(S.size)
            inf = jnp.zeros(S.size)
            R = jnp.zeros(S.size)
            V = jnp.zeros(S.size)
        else:
            df = pd.read_csv(saved_state)
            assert df.shape[0] == 100
            S = jnp.asarray(df["S"].values, dtype=jnp.float64)
            E = jnp.asarray(df["E"].values, dtype=jnp.float64)
            inf = jnp.asarray(df["I"].values, dtype=jnp.float64)
            R = jnp.asarray(df["R"].values, dtype=jnp.float64)
            V = jnp.asarray(df["V"].values, dtype=jnp.float64)
            tot_pop = S.sum() + E.sum() + inf.sum() + R.sum() + V.sum()
            assert tot_pop == self.tot_pop
            start_date = saved_date

        # Special computation for day
        onsame = date(year=start_date.year,
                      month=self.peak_date[0],
                      day=self.peak_date[1])
        d = (start_date - onsame).days
        if d < 0:
            onprev = date(year=start_date.year - 1,
                          month=self.peak_date[0],
                          day=self.peak_date[1])
            d = (start_date - onprev).days
        return S, E, inf, R, V, d
