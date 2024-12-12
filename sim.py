import configparser
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

from kce.epidata import EpiData
import kce.epistep as epistep


# jax.config.update("jax_enable_x64", True)


def updateVaxCost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)


def simulate(m, endDate, cacheFile=None, cacheDate=None):
    state = m.startState(cacheFile, cacheDate)
    trajectories = []
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

        # TODO: call m.switchProgram("prog name") after an
        # appropriate number of days
        trajectories.append(extState)

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating {curDate} (day {idx}:{day})")
            vaxdState = epistep.vaccinate(m, m.vaccRates, *state)
            state = vaxdState[0:6]
            trajectories[-1] = updateVaxCost(trajectories[-1], vaxdState[-1])

        if (curDate.month, curDate.day) == m.birthday:
            print(f"Aging population {curDate} (day {idx}:{day})")
            state = epistep.age(m, *state)

        curDate = curDate + timedelta(days=1)
        idx += 1
    print(f"End date {curDate} (day {idx}:{day})")
    return trajectories


def plot(m, trajectories):
    # We first plot dynamics
    summd = []
    d = 1
    for (S, E, Inf, R, V, day, *_) in trajectories:
        entry = ("Susceptible", float(S.sum()), d)
        summd.append(entry)
        entry = ("Exposed", float(E.sum()), d)
        summd.append(entry)
        entry = ("Infectious", float(Inf.sum()), d)
        summd.append(entry)
        entry = ("Recovered", float(R.sum()), d)
        summd.append(entry)
        entry = ("Vaccinated", float(V.sum()), d)
        summd.append(entry)
        d += 1
    df = pd.DataFrame(summd, columns=["Compartment", "Population", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="Population",
        hue="Compartment", style="Compartment"
    )
    plt.show()

    # Now we plot costs
    summd = []
    d = 1
    for (*_, day, ambCost, nomedCost, hospCost, vaxCost,
         ambQaly, nomedQaly, hospQaly, lifeyrsLost) in trajectories:
        entry = ("Ambulatory", float(ambCost.sum()), d)
        summd.append(entry)
        entry = ("No med care", float(nomedCost.sum()), d)
        summd.append(entry)
        entry = ("Hospital", float(hospCost.sum()), d)
        summd.append(entry)
        entry = ("Vaccine", float(vaxCost.sum()), d)
        summd.append(entry)
        d += 1
    df = pd.DataFrame(summd, columns=["Cost", "Euros", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="Euros",
        hue="Cost", style="Cost",
    )
    plt.yscale("log")
    plt.show()

    # Now we plot qaly
    total_cost = 0
    summd = []
    d = 1
    for (*_, day, ambCost, nomedCost, hospCost, vaxCost,
         ambQaly, nomedQaly, hospQaly, lifeyrsLost) in trajectories:
        entry = ("Ambulatory", float(ambQaly.sum()), d)
        summd.append(entry)
        entry = ("No med care", float(nomedQaly.sum()), d)
        summd.append(entry)
        entry = ("Hospital", float(hospQaly.sum()), d)
        summd.append(entry)
        entry = ("Life years", float(lifeyrsLost.sum()), d)
        summd.append(entry)
        total_cost += ((ambCost.sum() +
                        nomedCost.sum() +
                        hospCost.sum() +
                        vaxCost.sum()) +
                       (ambQaly.sum() +
                        nomedQaly.sum() +
                        hospQaly.sum() +
                        lifeyrsLost.sum()) * 35000)
        d += 1
    df = pd.DataFrame(summd, columns=["QALY", "QALY Units", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="QALY Units",
        hue="QALY", style="QALY",
    )
    plt.yscale("log")
    plt.show()

    print(f"The price of it all = {total_cost}")

    return


if __name__ == "__main__":
    m = EpiData()
    config = configparser.ConfigParser()
    config.read("config.ini")
    endDate = date.fromisoformat(sys.argv[1])
    # we also allow to be given a cached data filename and the day that
    # corresponds
    if len(sys.argv) > 2:
        cached = sys.argv[2]
        cachedDate = date.fromisoformat(sys.argv[3])
        print(f"Cached data file {cached} for date {cachedDate}")
        ts = simulate(m, endDate, cached, cachedDate)
    else:
        ts = simulate(m, endDate)
    plot(m, ts)
    exit(0)
