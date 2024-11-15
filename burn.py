import configparser
from datetime import date, timedelta
import pandas as pd

from kce.SEIRS import Model


# jax.config.update("jax_enable_x64", True)


def updateVaxCost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)


def simulate(m, endDate):
    state = m.init()
    curDate = m.startDate
    print(f"Start date {curDate}")

    while curDate <= endDate:
        (_, _, _, _, _, day) = state
        assert (m.startDate + timedelta(days=int(day))) == curDate

        if (curDate.month, curDate.day) == m.seedDate:
            print(f"Seeding infections {curDate} (day {day})")
            state = m.seedInfs(*state)

        extState = m.step(*state)
        state = extState[0:6]

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating {curDate} (day {day})")
            vaxdState = m.vaccinate(*state)
            state = vaxdState[0:6]

        if (curDate.month, curDate.day) == m.birthday:
            print(f"Aging population {curDate} (day {day})")
            state = m.age(*state)

        curDate = curDate + timedelta(days=1)
    print(f"End date {curDate} (day {day})")
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


if __name__ == "__main__":
    m = Model()
    config = configparser.ConfigParser()
    config.read("config.ini")
    endDate = date.fromisoformat(
        config.get("Defaults", "lastBurntDate"))
    last = simulate(m, endDate)
    dumpCSV(*last)
    exit(0)
