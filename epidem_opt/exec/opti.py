
from pathlib import Path

import jax
from jaxopt import GradientDescent

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_cost

# jax.config.update("jax_enable_x64", True)


def main():
    epi_data = EpiData(config_path=Path("./config.ini"),
                       epidem_data_path=Path("./data"),
                       econ_data_path=Path("./econ_data"),
                       qaly_data_path=Path("./econ_data"),
                       vaccination_rates_path=Path("./vaccination_rates"))

    value_and_grad_func = jax.value_and_grad(simulate_cost)
    # grad_cost = jax.grad(simulate_cost)
    # cost = simulate_cost(epi_data.vacc_rates, epi_data, epi_data.start_state(), epi_data.last_burnt_date)

    cost, gradient = value_and_grad_func(epi_data.vacc_rates, epi_data, epi_data.start_state(), epi_data.last_burnt_date)
    print(gradient)
    print(gradient.shape[0])
    print(f"The price of it all = {cost}")
    # TODO: it seems this gradient is with respect to the vaccination cost. How does it know this? Why did it not
    #   differentiate with respect to the other parameters?

    solver = GradientDescent(fun=value_and_grad_func, value_and_grad=True, maxiter=100)
    # TODO: solver.run().params


if __name__ == "__main__":
    main()
