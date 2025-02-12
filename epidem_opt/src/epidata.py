import configparser
from datetime import date
from pathlib import Path

import jax.numpy as jnp
import pandas as pd


class EpiData:
    def switchProgram(self, program: str):
        # TODO: factor out filename.
        df = pd.read_csv(self.vaccination_rates_path / f"program_{program}.csv")
        df["CovXEff"] = df.apply(lambda row: row.iloc[1] * row.iloc[2],
                                 axis=1)
        self.vaccRates = jnp.asarray(df["CovXEff"].values)
        return self.vaccRates

    def loadFromCSV(self, fname):
        df = pd.read_csv(fname, header=None)
        assert df.shape[0] == 100
        return jnp.asarray(df[0].values, dtype=jnp.float64)

    def __init__(self, config_path: Path,
                 epidem_data_path: Path,
                 econ_data_path: Path,
                 qaly_data_path: Path,
                 vaccination_rates_path: Path,
                 startDate: date=None):
        """
            Constructor.
        Parameters
        ----------
        startDate
            The start date of the epidemic simulation. If None, it will be retrieved from the config: Defaults/startDate
        """
        # Load parameters and constants from configuration file
        config = configparser.ConfigParser()
        config.read(config_path)
        cpars = config["Cont.Pars"]
        self.q = cpars.getfloat("q")
        self.sigma = cpars.getfloat("sigma")
        self.gamma = cpars.getfloat("gamma")
        self.omegaImm = cpars.getfloat("omegaImm")
        self.omegaVacc = cpars.getfloat("omegaVacc")
        self.delta = cpars.getfloat("delta")

        dpars = config["Disc.Pars"]
        self.seedInf = dpars.getint("seedInf")
        self.seedAges = jnp.asarray(range(dpars.getint("seedAgeMin"),
                                          dpars.getint("seedAgeMax")))
        self.seedDate = (dpars.getint("seedMonth"),
                         dpars.getint("seedDay"))
        self.birthday = (dpars.getint("birthMonth"),
                         dpars.getint("birthDay"))
        self.vaccDate = (dpars.getint("vaccMonth"),
                         dpars.getint("vaccDay"))
        self.peakDate = (dpars.getint("peakMonth"),
                         dpars.getint("peakDay"))

        self.noMedCare = config.getfloat("Cost.Pars", "noMedCare")

        # Defaults
        if startDate is None:
            self.startDate = date.fromisoformat(
                config.get("Defaults", "startDate"))
        else:
            self.startDate = startDate

        # FIXME: Factor out hardcoded data manipulations
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
        self.initPop = jnp.asarray(df["Population"].values, dtype=jnp.float64)
        self.totPop = self.initPop.sum()
        print(f" ****** Initial total population {self.totPop}")

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
        self.ambulatoryCosts = self.loadFromCSV(
            econ_data_path / config.get("EconFiles", "ambulatoryCosts"))
        self.vaccCosts = self.loadFromCSV(
            econ_data_path / config.get("EconFiles", "vaccineCost"))
        self.nomedCosts = self.loadFromCSV(
            econ_data_path / config.get("EconFiles", "noMedicalCareCosts"))
        self.hospCosts = self.loadFromCSV(
            econ_data_path / config.get("EconFiles", "hospCosts"))
        self.hospAmbCosts = self.loadFromCSV(
            econ_data_path / config.get("EconFiles", "hospAmbulatoryCosts"))

        # Qalys
        self.ambulatoryQalys = self.loadFromCSV(
            qaly_data_path / config.get("QalyFiles", "ambulatoryQalyLoss"))
        self.nomedQalys = self.loadFromCSV(
            qaly_data_path / config.get("QalyFiles", "noMedicalCareQalyLoss"))
        self.hospQalys = self.loadFromCSV(
            qaly_data_path / config.get("QalyFiles", "hospitalizedQalyLoss"))
        self.discLifeEx = self.loadFromCSV(
            qaly_data_path / config.get("QalyFiles", "discountedLifeExpectancy"))

        # vaccination stats
        self.vaccination_rates_path = vaccination_rates_path
        self.switchProgram(program="baseline")

    def startState(self, savedState: str = None, savedDate: date = None):
        """
            Retrieve the start state.

            The "saved state" is a filename that points to a CSV with initial conditions. The "saved date" is a date
            is ISO format.
        """
        if savedState is None:
            startDate = self.startDate
            S = self.initPop
            E = jnp.zeros(S.size)
            Inf = jnp.zeros(S.size)
            R = jnp.zeros(S.size)
            V = jnp.zeros(S.size)
        else:
            df = pd.read_csv(savedState)
            assert df.shape[0] == 100
            S = jnp.asarray(df["S"].values, dtype=jnp.float64)
            E = jnp.asarray(df["E"].values, dtype=jnp.float64)
            Inf = jnp.asarray(df["I"].values, dtype=jnp.float64)
            R = jnp.asarray(df["R"].values, dtype=jnp.float64)
            V = jnp.asarray(df["V"].values, dtype=jnp.float64)
            totPop = S.sum() + E.sum() + Inf.sum() + R.sum() + V.sum()
            assert totPop == self.totPop
            startDate = savedDate

        # Special computation for day
        onsame = date(year=startDate.year,
                      month=self.peakDate[0],
                      day=self.peakDate[1])
        d = (startDate - onsame).days
        if d < 0:
            onprev = date(year=startDate.year - 1,
                          month=self.peakDate[0],
                          day=self.peakDate[1])
            d = (startDate - onprev).days
        return S, E, Inf, R, V, d
