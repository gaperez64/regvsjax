import configparser
import pickle
from datetime import date, timedelta
from pathlib import Path

import jax
from jax import numpy as jnp

from epidem_opt.src.epidata import EpiData
# from epidem_opt.src import kce as epistep
import epidem_opt.src.epistep as epistep
from epidem_opt.src.simulator import simulate_cost
from epidem_opt.test.test_regression import check_output

# jax.config.update("jax_enable_x64", True)







if __name__ == "__main__":
    m = EpiData(config_path=Path("./config.ini"),
                epidem_data_path=Path("./data"),
                econ_data_path=Path("./econ_data"),
                qaly_data_path=Path("./econ_data"),
                vaccination_rates_path=Path("./vaccination_rates"))
    config = configparser.ConfigParser()
    config.read("config.ini")
    endDate = date.fromisoformat(config.get("Defaults", "lastBurntDate"))
    grad_cost = jax.grad(simulate_cost)
    cost = simulate_cost(m.vacc_rates, m, m.start_state(), endDate)
    print(f"The price of it all = {cost}")
    # with open("./working_dir/reference_cost.pickle", 'wb') as ref_file:
    #     pickle.dump(cost, ref_file)
    grad_cost = grad_cost(m.vacc_rates, m, m.start_state(), endDate)
    print(grad_cost)
    # with open("./working_dir/reference_grad.pickle", 'wb') as ref_file:
    #     pickle.dump(grad_cost, ref_file)
    # TODO: do gradient descent with the gradient.

    with open("./working_dir/reference_cost.pickle", 'rb') as ref_cost_file:
        ref_cost = pickle.load(ref_cost_file)
    with open("./working_dir/reference_grad.pickle", 'rb') as ref_grad_file:
        ref_grad = pickle.load(ref_grad_file)
    check_output(ref_cost=ref_cost, actual_cost=cost, ref_grad=ref_grad, actual_grad=grad_cost)
    exit(0)
