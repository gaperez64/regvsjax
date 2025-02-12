
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

    grad_cost = jax.value_and_grad(simulate_cost)
    # grad_cost = jax.grad(simulate_cost)
    # cost = simulate_cost(epi_data.vacc_rates, epi_data, epi_data.start_state(), epi_data.last_burnt_date)

    cost, grad_cost = grad_cost(epi_data.vacc_rates, epi_data, epi_data.start_state(), epi_data.last_burnt_date)
    print(grad_cost)
    print(f"The price of it all = {cost}")

    solver = GradientDescent(fun=grad_cost, value_and_grad=True, maxiter=100)
    # TODO: solver.run().params

if __name__ == "__main__":
    main()
