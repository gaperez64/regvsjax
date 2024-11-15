import configparser
from datetime import date
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


class Model:
    def switchProgram(self, prog="baseline"):
        self.vaccRates = _vaccRates(prog)

    def loadFromCSV(self, fname):
        df = pd.read_csv(fname, header=None)
        assert df.shape[0] == 100
        return jnp.asarray(df[0].values, dtype=jnp.float64)

    def __init__(self, startDate=None, peakDate=None):
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
        self.vaccineCosts = self.loadFromCSV("econ_data/vaccine_cost.csv")
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

        self.noMedCare = config.getfloat("Cost.Pars", "noMedCare")

        # Defaults
        if startDate is None:
            self.startDate = date.fromisoformat(
                config.get("Defaults", "startDate"))
        else:
            self.startDate = startDate
        if peakDate is None:
            self.peakDate = date.fromisoformat(
                config.get("Defaults", "peakDate"))
        else:
            self.peakDate = peakDate

    def init(self, savedState=None):
        if savedState is None:
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
        return S, E, Inf, R, V, 0

    @partial(jax.jit, static_argnums=0)
    def step(self, S, E, Inf, R, V, day):
        """
            We simulate one step of (forward) Newton integration
            with h = 1 (day). For efficiency, the method is JIT compiled
            NOTE: if any object attributes change after the first call,
            this will result in incorrect results as we assume self
            to be static
        """
        daterng = (self.startDate - self.peakDate).days + day
        z = 1 + self.delta * jnp.sin(2 * np.pi * (daterng / 365))
        beta = self.contact * self.q
        force = z * jnp.dot(beta, Inf)

        # daily transitions
        S2E = S * force  # element-wise product
        E2I = E * self.sigma
        I2R = Inf * self.gamma
        R2S = R * self.omegaImm
        V2S = V * self.omegaVacc

        # daily mortality = element-wise product with mortality
        # rates
        S2D = S * self.dailyMort
        E2D = E * self.dailyMort
        I2D = Inf * self.dailyMort
        R2D = R * self.dailyMort
        V2D = V * self.dailyMort

        # new values for all components
        newS = S - S2E + R2S + V2S - S2D
        newE = E + S2E - E2I - E2D
        newInf = Inf + E2I - I2R - I2D
        newR = R + I2R - R2S - R2D
        newV = V - V2S - V2D

        # breakdown of new infections
        confirmedInf = E2I * self.influenzaRate
        noMedCare = (confirmedInf / self.noMedCare) - confirmedInf
        hospd = confirmedInf * self.hospRate
        fatal = confirmedInf * self.caseFatalityRate

        # costs
        ambCost = (confirmedInf - hospd) * self.ambulatoryCosts
        noMedCost = noMedCare * self.nomedCosts
        hospCost = hospd * (self.hospCosts + self.hospAmbCosts)
        vaxCost = 0

        # qalys
        ambQaly = (confirmedInf - hospd) * self.ambulatoryQalys
        noMedQaly = noMedCare * self.nomedQalys
        hospQaly = hospd * self.hospQalys
        lifeyrsLost = fatal * self.discLifeEx

        return (newS, newE, newInf, newR, newV, day + 1,
                ambCost, noMedCost, hospCost, vaxCost,
                ambQaly, noMedQaly, hospQaly, lifeyrsLost)

    def seedInfs(self, S, E, Inf, R, V, day):
        newS = S.at[self.seedAges].add(-self.seedInf)
        newInf = Inf.at[self.seedAges].add(self.seedInf)
        return newS, E, newInf, R, V, day

    def vaccinate(self, S, E, Inf, R, V, day):
        return _vaccinate(S, E, Inf, R, V, day, self.vaccRates,
                          self.vaccineCosts)

    def age(self, S, E, Inf, R, V, day):
        return _age(S, E, Inf, R, V, day, self.totPop)


# This is NOT something that's easy to make a method of the model class because
# the vaccination rates and costs could change after switching vaccination
# strategy! To optimize via JAX compilation, this version of the function is
# externalized.
@jax.jit
def _vaccinate(S, E, Inf, R, V, day, vaccRates, vaccineCosts):
    # vaccination = element-wise product with vaccRates
    S2V = S * vaccRates
    E2V = E * vaccRates
    I2V = Inf * vaccRates
    R2V = R * vaccRates
    # updates
    newS = S - S2V
    newE = E - E2V
    newInf = Inf - I2V
    newR = R - R2V
    newV = V + S2V + E2V + I2V + R2V

    # cost
    vaxCost = (newV - V) * vaccineCosts
    return (newS, newE, newInf, newR, newV, day, vaxCost)


@jax.jit
def _age(S, E, Inf, R, V, day, totPop):
    newS = jnp.roll(S, 1).at[0].set(0)
    newE = jnp.roll(E, 1).at[0].set(0)
    newInf = jnp.roll(Inf, 1).at[0].set(0)
    newR = jnp.roll(R, 1).at[0].set(0)
    newV = jnp.roll(V, 1).at[0].set(0)
    # reincarnate dead people
    curPop = jnp.asarray([newS.sum(),
                          newE.sum(),
                          newInf.sum(),
                          newR.sum(),
                          newV.sum()]).sum()
    newS = newS.at[0].set(totPop - curPop)
    return newS, newE, newInf, newR, newV, day


def _vaccRates(prog):
    df = pd.read_csv(f"vaccination_rates/program_{prog}.csv")
    df["CovXEff"] = df.apply(lambda row: row.iloc[1] * row.iloc[2],
                             axis=1)
    return jnp.asarray(df["CovXEff"].values)
