
from pathlib import Path

import jax
from jaxopt import GradientDescent

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_cost
from epidem_opt.src.vacc_programs import read_cube


# jax.config.update("jax_enable_x64", True)


def main():
    # TODO: args
    epi_data = EpiData(config_path=Path("./config.ini"),
                       epidem_data_path=Path("./data"),
                       econ_data_path=Path("./econ_data"),
                       qaly_data_path=Path("./econ_data"),
                       vaccination_rates_path=Path("./vaccination_rates"))

    # construct derivative. This will differential w.r.t. the vaccination rates.
    value_and_grad_func = jax.value_and_grad(simulate_cost)

    # we start the vaccination simulation with the output of the burn-in step
    start_state = epi_data.start_state(
        saved_state_file="burn_file",  # TODO: save burn stuff to file
        saved_date=epi_data.last_burnt_date
    )

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
