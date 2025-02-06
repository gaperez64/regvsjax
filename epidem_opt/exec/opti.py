import configparser
import pickle
from datetime import date, timedelta
import jax
from jax import numpy as jnp

from epidem_opt.src.kce.epidata import EpiData
# from epidem_opt.src import kce as epistep
import epidem_opt.src.kce.epistep as epistep



# jax.config.update("jax_enable_x64", True)


@jax.jit
def updateVaxCost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)


def getCost(vaccRates, m, state, endDate):
    curDate = m.startDate
    total_cost = 0
    while curDate <= endDate:
        (S, E, Inf, R, V, day) = state

        if (curDate.month, curDate.day) == m.peakDate:
            day = 0
            state = (S, E, Inf, R, V, day)

        if (curDate.month, curDate.day) == m.seedDate:
            state = epistep.seedInfs(m, *state)

        (*state,
         ambCost, nomedCost, hospCost, vaxCost,
         ambQaly, nomedQaly, hospQaly, lifeyrsLost) = epistep.step(m, *state)

        if (curDate.month, curDate.day) == m.vaccDate:
            (*state, extraVaxCost) = epistep.vaccinate(m, vaccRates, *state)
            vaxCost += extraVaxCost

        total_cost += ((ambCost.sum() +
                        nomedCost.sum() +
                        hospCost.sum() +
                        vaxCost.sum()) +
                       (ambQaly.sum() +
                        nomedQaly.sum() +
                        hospQaly.sum() +
                        lifeyrsLost.sum()) * 35000)

        if (curDate.month, curDate.day) == m.birthday:
            state = epistep.age(m, *state)

        curDate = curDate + timedelta(days=1)
    return total_cost


def check_output(ref_cost, actual_cost, ref_grad, actual_grad):
    assert ref_cost == actual_cost, "Error, cost mismatch"
    assert jnp.allclose(ref_grad, actual_grad), "Error, grad mismatch"
    print("All correct.")


if __name__ == "__main__":
    m = EpiData()
    config = configparser.ConfigParser()
    config.read("config.ini")
    endDate = date.fromisoformat(config.get("Defaults", "lastBurntDate"))
    grad_cost = jax.grad(getCost)
    cost = getCost(m.vaccRates, m, m.startState(),  endDate)
    print(f"The price of it all = {cost}")
    # with open("./working_dir/reference_cost.pickle", 'wb') as ref_file:
    #     pickle.dump(cost, ref_file)
    grad_cost = grad_cost(m.vaccRates, m, m.startState(), endDate)
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
