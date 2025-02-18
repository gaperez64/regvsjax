import argparse
from pathlib import Path

import jax
from jaxopt import GradientDescent

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_cost
from epidem_opt.src.vacc_programs import read_cube


# jax.config.update("jax_enable_x64", True)


def main():
    # TODO: args

    arg_parser = argparse.ArgumentParser(prog="Utility to simulate the epidemiological behaviour of a population "
                                              "under a certain baseline vaccination program.")
    arg_parser.add_argument("--experiment_data", type=str, required=True,
                            help="Directory with the vaccination information. It should have sub-folders 'epi_data', "
                                 "'econ_data', 'vaccination_rates', 'qaly_data'.")
    arg_parser.add_argument('--burn_in_file', type=str, required=True,
                            help="CSV file with the results of the burn-in step.")

    args = arg_parser.parse_args()
    #
    # output_file = Path(args.output_file)
    # if output_file.is_file():
    #     print(f"Error: output file '{output_file}' already exists.")
    #     exit(1)

    experiment_data = Path(args.experiment_data)
    if not experiment_data.is_dir():
        print(f"Error: directory '{experiment_data}' does not exist.")
        exit(1)

    burn_in_file = Path(args.burn_in_file)
    if not burn_in_file.is_file():
        print(f"Error: burn-in file '{burn_in_file}' does not exist.")
        exit(1)

    epi_data = EpiData(
        config_path=experiment_data / "config.ini",
        epidem_data_path=experiment_data / "epidem_data",
        econ_data_path=experiment_data / "econ_data",
        qaly_data_path=experiment_data / "qaly_data",
        vaccination_rates_path=experiment_data / "vaccination_rates"
    )

    # construct derivative. This will differential w.r.t. the vaccination rates.
    value_and_grad_func = jax.value_and_grad(simulate_cost)

    # we start the vaccination simulation with the output of the burn-in step
    # TODO: some assert is failing here...
    start_state = epi_data.start_state(
        saved_state_file=burn_in_file,
        saved_date=epi_data.last_burnt_date
    )  # (S, E, I, R, V, day) with 100 age groups.

    # TODO: check that the "day" integer and the start_state "date" objects are aligned.
    # TODO: maybe just use "integer" days somewhere, and store all dates as integers relative to the start date (=day 0)?
    # cost, gradient = value_and_grad_func(epi_data.vacc_rates, epi_data, start_state, start_date, epi_data.end_date)

    # read vaccination programs over which we optimise
    cube = read_cube(Path("./vaccination_box.csv"))

    # sample a vaccination program
    vaccination_program = cube.sample()

    # TODO: extra metadata (vaccination_program, epi_data, start_state, epi_data.end_date)
    solver = GradientDescent(fun=value_and_grad_func, value_and_grad=True, maxiter=100)

    # TODO: solver.run().params


if __name__ == "__main__":
    main()
