from datetime import date
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


class Model:
    def switchProgram(self, prog="baseline"):
        self.vaccRates = _vaccRates(prog)

    def __init__(self):
        # FIXME: Factor out hardcoded data manipulations
        # contact matrix
        df = pd.read_csv("data/close+15_old.csv")
        self.contact = jnp.asarray(df.values)
        # mortality rates
        df = pd.read_csv("data/rateMor_eurostat_2021.csv")
        df.drop(df.tail(1).index, inplace=True)  # Ignore last value > 100
        self.dailyMort = jnp.asarray(df["Value"].values) / 365
        # initial population
        df = pd.read_csv("data/refPop_eurostat_2021.csv")
        df.drop(df.tail(1).index, inplace=True)  # Ignore last value > 100
        self.initPop = jnp.asarray(df["Population"].values, dtype=jnp.float64)
        self.totPop = self.initPop.sum()
        print(f" ****** Initial total population {self.totPop}")
        # Detailed infection rates
        df = pd.read_csv("data/influenzaRate.csv")
        self.influenzaRate = jnp.asarray(df["Rate"].values, dtype=jnp.float64)
        df = pd.read_csv("data/hospRate.csv")
        self.hospRate = jnp.asarray(df["Rate"].values, dtype=jnp.float64)
        df = pd.read_csv("data/caseFatalityRate.csv")
        self.caseFatalityRate = jnp.asarray(df["Rate"].values,
                                            dtype=jnp.float64)
        # vaccination stats
        self.switchProgram()

        # FIXME: All values below should be loaded from a settings/JSON file
        # other parameters
        self.q = 1.8 / 30.35392  # FIXME: the numerator is a randomized R0
        self.sigma = 1
        self.gamma = 0.26316
        self.omegaImm = 1 / (6 * 365)
        self.omegaVacc = 1 / (6 * 365)
        self.seedInf = 200
        self.seedAges = jnp.asarray(range(5, 51))
        self.delta = 0.43  # FIXME: this one is randomized too

        # other constants
        self.noMedCare = 0.492
        # FIXME: This is a random nr. of days after the (fixed)
        # FIXME: these values are not part of the model and are not used within this class. They need to be moved
        #   to simulator.
        # start of the season
        self.peak = date(year=2016, month=9, day=21)
        # FIXME: the peak/reference day above should be randomized too
        self.birthday = (8, 31)   # End of August
        self.startDate = date(year=2016, month=9, day=1)
        self.seedDate = (8, 31)   # End of August
        # FIXME: the seeding date above is also randomized
        self.vaccDate = (10, 10)  # October 10

    def init(self):
        S = self.initPop
        E = jnp.zeros(S.size)
        Inf = jnp.zeros(S.size)
        R = jnp.zeros(S.size)
        V = jnp.zeros(S.size)
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
        daterng = (self.startDate - self.peak).days + day
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
        confirmedInf = newInf * self.influenzaRate
        noMedCare = (confirmedInf / self.noMedCare) - confirmedInf
        hospd = confirmedInf * self.hospRate
        fatal = confirmedInf * self.caseFatalityRate

        return (newS, newE, newInf, newR, newV, day + 1,
                confirmedInf, noMedCare, hospd, fatal)

    def seedInfs(self, S, E, Inf, R, V, day):
        newS = S.at[self.seedAges].add(-self.seedInf)
        newE = E.at[self.seedAges].add(self.seedInf)
        return newS, newE, Inf, R, V, day

    def vaccinate(self, S, E, Inf, R, V, day):
        return _vaccinate(S, E, Inf, R, V, day, self.vaccRates)

    def age(self, S, E, Inf, R, V, day):
        return _age(S, E, Inf, R, V, day, self.totPop)


@jax.jit
def _vaccinate(S, E, Inf, R, V, day, vaccRates):
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
    return newS, newE, newInf, newR, newV, day


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
