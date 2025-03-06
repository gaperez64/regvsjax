import argparse
from pathlib import Path

import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_cost, date_to_ordinal_set

from scipy.optimize import minimize, OptimizeResult

from epidem_opt.src.vacc_programs import get_all_vaccination_programs_from_file


# jax.config.update("jax_enable_x64", True)


def main():
    arg_parser = argparse.ArgumentParser(prog="Utility to optimise the vaccination behaviour.")
    arg_parser.add_argument("--experiment_data", type=str, required=True,
                            help="Directory with the vaccination information. It should have sub-folders 'epi_data', "
                                 "'econ_data', 'vaccination_rates', 'qaly_data'.")
    arg_parser.add_argument('--burn_in_file', type=str, required=True,
                            help="CSV file with the results of the burn-in step.")
    arg_parser.add_argument('--init_program', type=str, required=True,
                            help="The name of the vacc. program from which we start the gradient descent.")

    args = arg_parser.parse_args()

    experiment_data = Path(args.experiment_data)
    if not experiment_data.is_dir():
        print(f"Error: directory '{experiment_data}' does not exist.")
        exit(1)

    burn_in_file = Path(args.burn_in_file)
    if not burn_in_file.is_file():
        print(f"Error: burn-in file '{burn_in_file}' does not exist.")
        exit(1)

    init_program_name = args.init_program

    epi_data = EpiData(
        config_path=experiment_data / "config.ini",
        epidem_data_path=experiment_data / "epidem_data",
        econ_data_path=experiment_data / "econ_data",
        qaly_data_path=experiment_data / "qaly_data",
        vaccination_rates_path=experiment_data / "vaccination_rates"
    )

    np.set_printoptions(suppress=True)

    # construct derivative. This will differentiate w.r.t. the vaccination rates.
    value_and_grad_func = jax.value_and_grad(simulate_cost)

    # we start the vaccination simulation with the output of the burn-in step
    start_state = epi_data.start_state(
        saved_state_file=burn_in_file,
        saved_date=epi_data.last_burnt_date
    )  # (S, E, I, R, V, day) with 100 age groups.

    # maxx_vacc = jnp.ones(shape=(100,))
    # anti_vacc = jnp.zeros(shape=(100,))

    start_date = epi_data.last_burnt_date
    end_date = epi_data.end_date

    def get_value_and_grad(vacc_rates):
        """
            Wrapper that invokes the grad function for the start values and appropriate dates.
        """
        value, grad = value_and_grad_func(
            vacc_rates,
            epi_data=epi_data,
            epi_state=start_state,
            start_date=start_date.toordinal(),
            end_date=end_date.toordinal(),
            vacc_dates=lambda x: x in date_to_ordinal_set(epi_data.vacc_date,
                                                          epi_data.last_burnt_date,
                                                          epi_data.end_date),
            peak_dates=lambda x: x in date_to_ordinal_set(epi_data.peak_date,
                                                          epi_data.last_burnt_date,
                                                          epi_data.end_date),
            seed_dates=lambda x: x in date_to_ordinal_set(epi_data.seed_date,
                                                          epi_data.last_burnt_date,
                                                          epi_data.end_date),
            birth_dates=lambda x: x in date_to_ordinal_set(epi_data.birthday,
                                                           epi_data.last_burnt_date,
                                                           epi_data.end_date)
        )

        print("value:", value)
        grad_norm = grad / np.linalg.norm(grad)

        print("normalised gradient:", grad_norm)

        return jnp.array(value, dtype=jnp.float64), grad_norm

    # read vaccination programs
    vacc_rates = get_all_vaccination_programs_from_file(vacc_programs=experiment_data / "vacc_programs.csv")

    # check if the user specified a valid vaccination program
    if init_program_name not in vacc_rates:
        print(f"Error, invalid vaccination program '{init_program_name}'.")
        exit(1)

    # do gradient descent
    init_rates = vacc_rates[init_program_name]
    _gradient_descent_(val_and_grad_func=get_value_and_grad, init_rates=init_rates)


def _gradient_descent_(val_and_grad_func, init_rates):
    """
        TODO:
            - convergence
            - check if normalisation is OK
            - 
    """
    cur = init_rates
    most_recent_valid = cur
    val = None
    for i in range(50):
        print(f"ITER {i}")
        print("Rates before:", cur)
        old_val = val
        val, grad = val_and_grad_func(cur)
        # print(val, grad)
        if old_val is not None:
            print("Difference:", old_val - val)

        cur -= 0.005 * grad
        if np.all(cur <= 1) and np.all(cur >= 0):
            most_recent_valid = cur
        else:
            print("=== CONSTRAINT VIOLATION ===")
    print(most_recent_valid)


# def _debug_run_sim_no_jax_(vacc_program=None, epi_data=None, start_state=None, start_date=None, end_date=None):
#     """
#         Compute the cost of the vaccination program without touching any JAX functionality.
#
#         Something seems to go wrong after the vaccination day.
#     """
#     cost_3 = simulate_cost(
#         vacc_program,
#         epi_data=epi_data,
#         epi_state=start_state,
#         start_date=start_date.toordinal(),
#         end_date=end_date.toordinal(),
#         vacc_dates=lambda x: x in date_to_ordinal_set(epi_data.vacc_date,
#                                                       epi_data.last_burnt_date,
#                                                       epi_data.end_date),
#         peak_dates=lambda x: x in date_to_ordinal_set(epi_data.peak_date,
#                                                       epi_data.last_burnt_date,
#                                                       epi_data.end_date),
#         seed_dates=lambda x: x in date_to_ordinal_set(epi_data.seed_date,
#                                                       epi_data.last_burnt_date,
#                                                       epi_data.end_date),
#         birth_dates=lambda x: x in date_to_ordinal_set(epi_data.birthday,
#                                                        epi_data.last_burnt_date,
#                                                        epi_data.end_date)
#     )
#     print(cost_3)


def _write_vacc_program_(vacc_program, baseline_file: Path, output_path: Path):
    df_baseline = pd.read_csv(baseline_file)
    # Spelling error is intentional
    assert "Coverate rate" in df_baseline.columns
    df_baseline["Coverate rate"] = vacc_program
    df_baseline.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
