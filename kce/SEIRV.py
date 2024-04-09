import jax.numpy as jnp
import pandas as pd


class Model:
    def __init__(self):
        # contact matrix
        df = pd.read_csv("data/close+15_old.csv")
        self.contact = jnp.asarray(df.values)
        # mortality rates
        df = pd.read_csv("data/rateMor_eurostat_2021.csv")
        self.rateMor = jnp.asarray(df["Value"].values)
        # other parameters
        self.q = 1.8 / 15.2153
        self.sigma = 0.5
        self.gamma = 0.5
        self.omega_i = 0.9 / 365
        self.omega_v = 0.9 / 365
        self.mu_d = self.rateMor / 365
        self.seedInf = 200
        self.seedAge = list(range(5, 51))
        self.alpha = 0.2
        self.delta = 0.7  # FIXME

    def step(self, S, E, I, R, V, day):
        pass

    def age(self, S, E, I, R, V, day):
        pass

    def infect(self, S, E, I, R, V, day):
        pass

    def vaccinate(self, S, E, I, R, V, day):
        pass
