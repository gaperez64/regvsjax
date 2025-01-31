import configparser
from datetime import date, timedelta
import pandas as pd

from epidem_opt.src.kce.epidata import EpiData
from epidem_opt.src.kce import epistep


# jax.config.update("jax_enable_x64", True)


def updateVaxCost(t, vaxCost):
    """
        Update the vaccination cost.

        This takes a tuple (..., vc, aq, nmq, hq, ll):
            - ... = other parameters
            - vc = current vaccination cost
            - aq = ambulatory costs
            - nmq = no medical treatment cost
            - hq = hospital costs
            - ll = lost life years.

        The value 'vc' is increased with 'vaxCosts', and the modified tuple is returned.
    """
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return *rest, vaxCost + vc, aq, nmq, hq, ll


def simulate(m, endDate):
    """
        Simulate the epidemic from start date until end date.

        - on peak dates, the "day" is set to 0.
        - on seed dates, we call "epistep.seedInfs"
        - after adjusting for peak dates and seed dates, we call "epistep.step" on every iteration
        - on vaccination dates, we call "epistep.vaccinate"
        - on the "birth day" dates, we call "epistep.age"
    """
    state = m.startState()
    curDate = m.startDate
    idx = 1
    print(f"Start date {curDate}")
    while curDate <= endDate:
        (S, E, Inf, R, V, day) = state

        if (curDate.month, curDate.day) == m.peakDate:
            print(f"Reseting flu cycle {curDate} (day {idx}:{day})")
            day = 0
            state = (S, E, Inf, R, V, day)

        if (curDate.month, curDate.day) == m.seedDate:
            print(f"Seeding infections {curDate} (day {idx}:{day})")
            state = epistep.seedInfs(m, *state)

        extState = epistep.step(m, *state)
        state = extState[0:6]

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating {curDate} (day {idx}:{day})")
            vaxdState = epistep.vaccinate(m, m.vaccRates, *state)
            state = vaxdState[0:6]

        if (curDate.month, curDate.day) == m.birthday:
            print(f"Aging population {curDate} (day {idx}:{day})")
            state = epistep.age(m, *state)

        curDate = curDate + timedelta(days=1)
        idx += 1
    print(f"End date {curDate} (day {idx}:{day})")
    return state


def dumpCSV(S, E, Inf, R, V, day):
    named = {"Age": range(len(S)),
             "S": S,
             "E": E,
             "I": Inf,
             "R": R,
             "V": V}
    df = pd.DataFrame(named)
    df.to_csv("data/afterBurnIn.csv", index=False)


def main():
    # TODO: arguments
    m = EpiData()
    config = configparser.ConfigParser()
    config.read("config.ini")
    # TODO: move end date into the config.
    endDate = date.fromisoformat(config.get("Defaults", "lastBurntDate"))
    last = simulate(m, endDate)
    dumpCSV(*last)
    exit(0)


if __name__ == "__main__":
    main()
