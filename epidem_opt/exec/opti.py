import argparse
from pathlib import Path

import jax
from jaxopt import GradientDescent

from epidem_opt.src.epidata import EpiData, JaxFriendlyEpiData
from epidem_opt.src.simulator import simulate_cost, date_to_ordinal_set
from epidem_opt.src.vacc_programs import read_cube


# jax.config.update("jax_enable_x64", True)


def main():
    arg_parser = argparse.ArgumentParser(prog="Utility to optimise the vaccination behaviour.")
    arg_parser.add_argument("--experiment_data", type=str, required=True,
                            help="Directory with the vaccination information. It should have sub-folders 'epi_data', "
                                 "'econ_data', 'vaccination_rates', 'qaly_data'.")
    arg_parser.add_argument('--burn_in_file', type=str, required=True,
                            help="CSV file with the results of the burn-in step.")

    args = arg_parser.parse_args()

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
    cube = read_cube(Path(experiment_data / "vaccination_box.csv"))

    # sample an initial vaccination program
    initial_vacc_program = cube.sample()

    solver = GradientDescent(fun=value_and_grad_func, value_and_grad=True, maxiter=100)

    # NOTE: the following two functions can be useful here:
    # "jax.lax.cond"
    # "jax.debug.print"

    # init-params is the vaccination program from which we start.
    # then, we pass additional params (epi_data, start_state, end_date)

    # TODO: the ordinal set could be a JIT-compiled predicate
    result = solver.run(
        init_params=initial_vacc_program,
        epi_data=JaxFriendlyEpiData.create(epi_data),
        state=start_state,
        start_date=epi_data.last_burnt_date.toordinal(),  # TODO: these dates should be replaced by integers.
        end_date=epi_data.end_date.toordinal(),
        vacc_dates=date_to_ordinal_set(epi_data.vacc_date, epi_data.last_burnt_date, epi_data.end_date),
        peak_dates=date_to_ordinal_set(epi_data.peak_date, epi_data.last_burnt_date, epi_data.end_date),
        seed_dates=date_to_ordinal_set(epi_data.seed_date, epi_data.last_burnt_date, epi_data.end_date),
        birth_dates=date_to_ordinal_set(epi_data.birthday, epi_data.last_burnt_date, epi_data.end_date)
    )

    print(result)

    # TODO: solver.run().params


if __name__ == "__main__":
    main()
