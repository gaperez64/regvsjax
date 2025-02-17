import argparse
from pathlib import Path

import pandas as pd

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_trajectories


# jax.config.update("jax_enable_x64", True)


def save_to_csv(S, E, Inf, R, V, output_file):
    # NOTE: this assumes that all ages 0...n-1 are covered by the simulation.

    df = pd.DataFrame({
        "Age": range(len(S)),
        "S": S,
        "E": E,
        "I": Inf,
        "R": R,
        "V": V})
    df.to_csv(output_file, index=False)


def main():
    """
        Utility to simulate the epidemiological behaviour of a population under a certain baseline vaccination program.
    """

    arg_parser = argparse.ArgumentParser(prog="Utility to simulate the epidemiological behaviour of a population "
                                              "under a certain baseline vaccination program.")
    arg_parser.add_argument("--experiment_data", type=str, required=True,
                            help="Directory with the vaccination information. It should have sub-folders 'epi_data', "
                                 "'econ_data', 'vaccination_rates', 'qaly_data'.")
    arg_parser.add_argument('--output_file', type=str, required=True,
                            help="Output file with the results of the burn-in step.")

    args = arg_parser.parse_args()

    output_file = Path(args.output_file)
    if output_file.is_file():
        print(f"Error: output file '{output_file}' already exists.")
        exit(1)

    experiment_data = Path(args.experiment_data)
    if not experiment_data.is_dir():
        print(f"Error: directory '{experiment_data}' does not exist.")
        exit(1)

    epi_data = EpiData(
        config_path=experiment_data / "config.ini",
        epidem_data_path=experiment_data / "epidem_data",
        econ_data_path=experiment_data / "econ_data",
        qaly_data_path=experiment_data / "qaly_data",
        vaccination_rates_path=experiment_data / "vaccination_rates"
    )
    last = simulate_trajectories(epi_data, epi_data.last_burnt_date)

    # take state vector and store to CSV
    (*epidem_state, day) = last
    save_to_csv(*epidem_state, output_file=output_file)


if __name__ == "__main__":
    main()
