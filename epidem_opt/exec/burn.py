import argparse
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
    #   -> experiment folder
    #   -> output_folder

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--experiment_data", type=str, required=True,
                            help="Directory with the vaccination information. It should have sub-folders 'data', 'econ_data', 'vaccination_rates', ")
    arg_parser.add_argument('--output_file', type=str, required=True,
                            help="Output file with the minimum and maximum rates per age group.")

    args = arg_parser.parse_args()



    epi_data = EpiData(config_path=Path("./experiment_data/config.ini"),
                epidem_data_path=Path("./experiment_data/epidem_data"),
                econ_data_path=Path("./experiment_data/econ_data"),
                qaly_data_path=Path("./experiment_data/qaly_data"),
                vaccination_rates_path=Path("./experiment_data/vaccination_rates"),)
    # config = configparser.ConfigParser()
    # config.read("config.ini")
    # endDate = date.fromisoformat(config.get("Defaults", "lastBurntDate"))
    last = simulate_trajectories(epi_data, epi_data.last_burnt_date)

    # TODO: check if this is OK. Add output folder.
    dumpCSV(*last)


if __name__ == "__main__":
    main()
