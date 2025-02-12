import configparser
import pickle
from datetime import date, timedelta
from pathlib import Path

import jax
from jax import numpy as jnp

from epidem_opt.src.epidata import EpiData
# from epidem_opt.src import kce as epistep
import epidem_opt.src.epistep as epistep



# jax.config.update("jax_enable_x64", True)


@jax.jit
def update_vax_cost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)


def get_cost(vaccRates, m, state, end_date):
    cur_date = m.start_date
    total_cost = 0
    while cur_date <= end_date:
        (S, E, Inf, R, V, day) = state

        if (cur_date.month, cur_date.day) == m.peak_date:
            day = 0
            state = (S, E, Inf, R, V, day)

        if (cur_date.month, cur_date.day) == m.seed_date:
            state = epistep.seedInfs(m, *state)

        (*state,
         amb_cost, nomed_cost, hosp_cost, vax_cost,
         amb_qaly, nomed_qaly, hosp_qaly, lifeyrs_lost) = epistep.step(m, *state)

        if (cur_date.month, cur_date.day) == m.vacc_date:
            (*state, extraVaxCost) = epistep.vaccinate(m, vaccRates, *state)
            vax_cost += extraVaxCost

        total_cost += ((amb_cost.sum() +
                        nomed_cost.sum() +
                        hosp_cost.sum() +
                        vax_cost.sum()) +
                       (amb_qaly.sum() +
                        nomed_qaly.sum() +
                        hosp_qaly.sum() +
                        lifeyrs_lost.sum()) * 35000)

        if (cur_date.month, cur_date.day) == m.birthday:
            state = epistep.age(m, *state)

        cur_date = cur_date + timedelta(days=1)
    return total_cost


def check_output(ref_cost, actual_cost, ref_grad, actual_grad):
    assert ref_cost == actual_cost, "Error, cost mismatch"
    assert jnp.allclose(ref_grad, actual_grad), "Error, grad mismatch"
    print("All correct.")


if __name__ == "__main__":
    m = EpiData(config_path=Path("./config.ini"),
                epidem_data_path=Path("./data"),
                econ_data_path=Path("./econ_data"),
                qaly_data_path=Path("./econ_data"),
                vaccination_rates_path=Path("./vaccination_rates"))
    config = configparser.ConfigParser()
    config.read("config.ini")
    endDate = date.fromisoformat(config.get("Defaults", "lastBurntDate"))
    grad_cost = jax.grad(get_cost)
    cost = get_cost(m.vacc_rates, m, m.start_state(), endDate)
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
