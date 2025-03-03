import argparse
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_cost, date_to_ordinal_set

from scipy.optimize import minimize, OptimizeResult


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

    def get_value(vacc_rates):
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

        print(value)

        return value

    def get_gradient(vacc_rates):
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

        return grad



    class Wrapper:
        """
            Reduces the number of calls to the solver
        """
        def __init__(self):
            self.cache = {}

        def __call__(self, x, *args):
            fun, grad = value_and_grad_func(x,
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
            print("fun", fun)
            self.cache['grad'] = grad
            return fun

        def jac(self, x, *args):
            return np.array(self.cache.pop('grad'))

    # TODO: fix bounds.
    # print(bnds)

    def callback(intermediate_result: OptimizeResult):
        print(intermediate_result)

    bounds = [(0, 1)]*100

    cons = []
    for factor in range(len(bounds)):
        lower, upper = bounds[factor]
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)

    wrapper = Wrapper()
    # value = minimize(get_value, epi_data.vacc_rates, jac=get_gradient, bounds=bnds, options={"maxiter": 2, "disp": True}, callback=callback)
    value = minimize(wrapper, np.array(epi_data.vacc_rates), jac=wrapper.jac, constraints=cons, options={"maxiter": 3, "disp": True}, callback=callback, method="COBYLA")
    print(value)


def _debug_run_sim_no_jax_(vacc_program=None, epi_data=None, start_state=None, start_date=None, end_date=None):
    """
        Compute the cost of the vaccination program without touching any JAX functionality.

        Something seems to go wrong after the vaccination day.
    """
    cost_3 = simulate_cost(
        vacc_program,
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
    print(cost_3)


def _write_vacc_program_(vacc_program, baseline_file: Path, output_path: Path):
    df_baseline = pd.read_csv(baseline_file)
    # Spelling error is intentional
    assert "Coverate rate" in df_baseline.columns
    df_baseline["Coverate rate"] = vacc_program
    df_baseline.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
