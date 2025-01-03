import configparser
from datetime import date
import jax.numpy as jnp
import pandas as pd


class EpiData:
    def switchProgram(self, prog="baseline"):
        df = pd.read_csv(f"vaccination_rates/program_{prog}.csv")
        df["CovXEff"] = df.apply(lambda row: row.iloc[1] * row.iloc[2],
                                 axis=1)
        self.vaccRates = jnp.asarray(df["CovXEff"].values)
        return self.vaccRates

    def loadFromCSV(self, fname):
        df = pd.read_csv(fname, header=None)
        assert df.shape[0] == 100
        return jnp.asarray(df[0].values, dtype=jnp.float64)

    def __init__(self, startDate=None):
        # FIXME: Factor out hardcoded data manipulations
        # contact matrix
        df = pd.read_csv("data/contact_matrix.csv")
        assert df.shape[0] == 100
        self.contact = jnp.asarray(df.values)
        # mortality rates
        df = pd.read_csv("data/rateMort.csv")
        df.drop(df.tail(1).index, inplace=True)  # Ignore last value > 100
        assert df.shape[0] == 100
        self.dailyMort = jnp.asarray(df["Value"].values) / 365
        # initial population
        df = pd.read_csv("data/startPop.csv")
        df.drop(df.tail(1).index, inplace=True)  # Ignore last value > 100
        assert df.shape[0] == 100
        self.initPop = jnp.asarray(df["Population"].values, dtype=jnp.float64)
        self.totPop = self.initPop.sum()
        print(f" ****** Initial total population {self.totPop}")

        # Detail rates
        df = pd.read_csv("data/influenzaRate.csv", header=None)
        assert df.shape[0] == 100
        self.influenzaRate = jnp.asarray(df[0].values, dtype=jnp.float64)
        df = pd.read_csv("data/hospRate.csv")
        assert df.shape[0] == 100
        self.hospRate = jnp.asarray(df["Rate"].values, dtype=jnp.float64)
        df = pd.read_csv("data/caseFatalityRate.csv")
        assert df.shape[0] == 100
        self.caseFatalityRate = jnp.asarray(df["Rate"].values,
                                            dtype=jnp.float64)
        # Costs
        self.ambulatoryCosts =\
            self.loadFromCSV("econ_data/ambulatory_costs.csv")
        self.vaccCosts = self.loadFromCSV("econ_data/vaccine_cost.csv")
        self.nomedCosts =\
            self.loadFromCSV("econ_data/no_medical_care_costs.csv")
        self.hospCosts =\
            self.loadFromCSV("econ_data/hospitalization_costs.csv")
        self.hospAmbCosts =\
            self.loadFromCSV("econ_data/hosp_ambulatory_costs.csv")

        # Qalys
        self.ambulatoryQalys =\
            self.loadFromCSV("econ_data/ambulatory_qaly_loss.csv")
        self.nomedQalys =\
            self.loadFromCSV("econ_data/no_medical_care_qaly_loss.csv")
        self.hospQalys =\
            self.loadFromCSV("econ_data/hospitalized_qaly_loss.csv")
        self.discLifeEx =\
            self.loadFromCSV("econ_data/discounted_life_expectancy.csv")

        # vaccination stats
        self.switchProgram()

        # Load parameters and constants from configuration file
        config = configparser.ConfigParser()
        config.read("config.ini")
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

    def startState(self, savedState=None, savedDate=None):
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
