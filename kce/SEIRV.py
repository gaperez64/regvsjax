import jax.numpy as jnp
import pandas as pd


class Model:
    def __init__(self):
        df = pd.read_csv("data/close+15_old.csv")
        print(df.head())
