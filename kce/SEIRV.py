import jax.numpy as jnp
import numpy as np
import pandas as pd


class Model:
    def __init__(self):
        # contact matrix
        df = pd.read_csv("data/close+15_old.csv")
        self.contact = jnp.asarray(df.values)
        # mortality rates
        df = pd.read_csv("data/rateMor_eurostat_2021.csv")
        df.drop(df.tail(1).index, inplace=True)  # FIXME
        self.dailyMort = jnp.asarray(df["Value"].values) / 365
        # initial population
        df = pd.read_csv("data/refPop_eurostat_2021.csv")
        df.drop(df.tail(1).index, inplace=True)  # FIXME
        self.initPop = jnp.asarray(df["Population"].values,
                                   dtype=float)
        # TODO: vaccination efficacy
        # other parameters
        self.q = 1.8 / 15.2153
        self.sigma = 0.5
        self.gamma = 0.5
        self.omegaImm = 0.33 / 365
        self.omegaVacc = 0.33 / 365
        self.seedInf = 200
        self.seedAges = jnp.asarray(range(5, 51))
        self.delta = 0.7
        self.peak = 1  # day of the year (out of 365)

    def init(self):
        S = self.initPop
        E = jnp.zeros(S.size)
        Inf = jnp.zeros(S.size)
        R = jnp.zeros(S.size)
        V = jnp.zeros(S.size)
        return (S, E, Inf, R, V)

    # Using Forward Newton
    def step(self, S, E, Inf, R, V, day):
        z = 1 + jnp.sin(2 * np.pi * (day - self.peak) / 365)
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
        V2D = R * self.dailyMort

        # new values for all components
        newS = S - S2E + R2S + V2S - S2D
        newE = E + S2E - E2I - E2D
        newInf = Inf + E2I - I2R - I2D
        newR = R + I2R - R2S - R2D
        newV = V - V2S - V2D
        return (newS, newE, newInf, newR, newV, day + 1)

    def seedInfs(self, S, E, Inf, R, V, day):
        newS = S.at[self.seedAges].add(-self.seedInf)
        newE = E.at[self.seedAges].add(self.seedInf)
        return (newS, newE, Inf, R, V, day)

    def vaccinate(self, S, E, Inf, R, V, day):
        pass

    def age(self, S, E, Inf, R, V, day):
        pass
