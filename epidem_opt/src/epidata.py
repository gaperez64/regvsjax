import configparser
from datetime import date
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pandas as pd
from flax import struct
from jax import Array

from epidem_opt.src.vacc_programs import get_all_vaccination_programs_from_file


def load_from_csv(fname):
    df = pd.read_csv(fname, header=None)
    assert df.shape[0] == 100
    return jnp.asarray(df[0].values, dtype=jnp.float64)


class EpiData:
    # def switch_program(self, program: str):
    #     df = pd.read_csv(self.vaccination_rates_path / f"program_{program}.csv")
    #     df["CovXEff"] = df.apply(lambda row: row.iloc[1] * row.iloc[2],
    #                              axis=1)
    #     self.vacc_rates = jnp.asarray(df["CovXEff"].values)
    #     return self.vacc_rates

    def __init__(self, config_path: Path,
                 epidem_data_path: Path,
                 econ_data_path: Path,
                 qaly_data_path: Path,
                 vaccination_rates_path: Path = None,
                 baseline_program_name: str = "baseline"
                 ):
        """
            Constructor.
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

        self.start_date = date.fromisoformat(config.get("Defaults", "startDate"))
        self.last_burnt_date = date.fromisoformat(config.get("Defaults", "lastBurntDate"))
        self.end_date = date.fromisoformat(config.get("Defaults", "endDate"))

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
        # amount of people per age group at time zero
        self.init_pop = jnp.asarray(df["Population"].values, dtype=jnp.float64)
        self.tot_pop = self.init_pop.sum()  # total amount of people at the start
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
        vacc_rates = get_all_vaccination_programs_from_file(vacc_programs=vaccination_rates_path)
        if baseline_program_name not in vacc_rates:
            print(f"Error, invalid vaccination program '{baseline_program_name}'. "
                  f"Please specify the name of an existing vaccination program.")
        self.vacc_rates = vacc_rates[baseline_program_name]

    def start_state(self, saved_state_file: Path = None, saved_date: date = None
                    ) -> tuple[Array, Array, Array, Array, Array, int]:
        """
            Retrieve the start state.

            The "saved state" is a filename that points to a CSV with initial conditions. The "saved date" is a date
            is ISO format.
        """
        if saved_state_file is None:
            start_date = self.start_date
            S = self.init_pop
            E = jnp.zeros(S.size)
            I = jnp.zeros(S.size)
            R = jnp.zeros(S.size)
            V = jnp.zeros(S.size)
        else:
            df = pd.read_csv(saved_state_file)
            assert df.shape[0] == 100  # Sanity check: we have 100 age grounds
            S = jnp.asarray(df["S"].values, dtype=jnp.float64)
            E = jnp.asarray(df["E"].values, dtype=jnp.float64)
            I = jnp.asarray(df["I"].values, dtype=jnp.float64)
            R = jnp.asarray(df["R"].values, dtype=jnp.float64)
            V = jnp.asarray(df["V"].values, dtype=jnp.float64)
            tot_pop = S.sum() + E.sum() + I.sum() + R.sum() + V.sum()

            # self.tot_pop = sum over all ages
            # tot_pop = sum over all compartments, over all ages in each compartment


            # Sanity check: the population of the file we read is the same as at the beginning

            #self.tot_pop = self.init_pop.sum()  # total amount of people at the start
            # print(tot_pop - self.tot_pop)
            # assert tot_pop == self.tot_pop  # TODO:the error here is -117717.0, which is too much.
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
        return S, E, I, R, V, d


@struct.dataclass
class JaxFriendlyEpiData:
    """
        A JAX-friendly container for epidemiological data.
    """
    tot_pop: Any
    vacc_date: Any
    seed_date: Any
    peak_date: Any
    birthday: Any
    vacc_rates: Any
    vacc_costs: Any
    seed_inf: Any
    seed_ages: Any
    sigma: Any
    gamma: Any
    omega_imm: Any
    omega_vacc: Any
    delta: Any
    contact: Any
    q: Any
    dailyMort: Any
    influenzaRate: Any
    no_med_care: Any
    hospRate: Any
    caseFatalityRate: Any
    ambulatory_costs: Any
    nomedCosts: Any
    hospCosts: Any
    hospAmbCosts: Any
    ambulatory_qalys: Any
    nomed_qalys: Any
    hosp_qalys: Any
    disc_life_ex: Any

    @classmethod
    def create(cls, epi_data: EpiData):
        # I used kw arguments here to prevent any accidents.
        return cls(
            tot_pop=epi_data.tot_pop,
            vacc_date=epi_data.vacc_date,
            seed_date=epi_data.seed_date,
            peak_date=epi_data.peak_date,
            birthday=epi_data.birthday,
            vacc_costs=epi_data.vacc_costs,
            vacc_rates=epi_data.vacc_rates,
            seed_inf=epi_data.seed_inf,
            seed_ages=epi_data.seed_ages,
            sigma=epi_data.sigma,
            gamma=epi_data.gamma,
            omega_imm=epi_data.omega_imm,
            omega_vacc=epi_data.omega_vacc,
            delta=epi_data.delta,
            contact=epi_data.contact,
            q=epi_data.q,
            dailyMort=epi_data.dailyMort,
            influenzaRate=epi_data.influenzaRate,
            no_med_care=epi_data.no_med_care,
            hospRate=epi_data.hospRate,
            caseFatalityRate=epi_data.caseFatalityRate,
            ambulatory_costs=epi_data.ambulatory_costs,
            nomedCosts=epi_data.nomedCosts,
            hospCosts=epi_data.hospCosts,
            hospAmbCosts=epi_data.hospAmbCosts,
            nomed_qalys=epi_data.nomed_qalys,
            hosp_qalys=epi_data.hosp_qalys,
            disc_life_ex=epi_data.disc_life_ex,
            ambulatory_qalys=epi_data.ambulatory_qalys
        )
