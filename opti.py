import configparser
from datetime import date, timedelta
from functools import partial
import jax

from kce.epidata import EpiData
import kce.epistep as epistep


# jax.config.update("jax_enable_x64", True)


@jax.jit
def updateVaxCost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)


def simulate(m, state, endDate):
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
            (*state, extraVaxCost) = epistep.vaccinate(m, m.vaccRates, *state)
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


if __name__ == "__main__":
    m = EpiData()
    config = configparser.ConfigParser()
    config.read("config.ini")
    endDate = date.fromisoformat(config.get("Defaults", "lastBurntDate"))
    cost = simulate(m, m.startState(),  endDate)
    print(f"The price of it all = {cost}")
    exit(0)
