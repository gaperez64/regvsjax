from pathlib import Path

import pandas as pd

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_trajectories


# jax.config.update("jax_enable_x64", True)


def dumpCSV(S, E, Inf, R, V, day):
    named = {"Age": range(len(S)),
             "S": S,
             "E": E,
             "I": Inf,
             "R": R,
             "V": V}
    df = pd.DataFrame(named)
    df.to_csv("data/afterBurnIn.csv", index=False)


def main():
    # TODO: arguments
    epi_data = EpiData(config_path=Path("./config.ini"),
                epidem_data_path=Path("./epidem_data"),
                econ_data_path=Path("./econ_data"),
                qaly_data_path=Path("./qaly_data"),
                vaccination_rates_path=Path("./vaccination_rates"),)
    # config = configparser.ConfigParser()
    # config.read("config.ini")
    # endDate = date.fromisoformat(config.get("Defaults", "lastBurntDate"))
    last = simulate_trajectories(epi_data, epi_data.last_burnt_date)
    dumpCSV(*last)


if __name__ == "__main__":
    main()
