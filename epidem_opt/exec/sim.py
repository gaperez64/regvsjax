import configparser
import pickle
from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import jax.numpy as jnp

from epidem_opt.src.kce.epidata import EpiData
from epidem_opt.src.kce import epistep


# jax.config.update("jax_enable_x64", True)


def updateVaxCost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)


def simulate(m, endDate, cacheFile=None, cacheDate=None):
    """
        TODO: this is the same as the simulation loop in the "burn" module. Refactor.

        Simulate the epidemic from start date until end date.

        - on peak dates, the "day" is set to 0.
        - on seed dates, we call "epistep.seedInfs"
        - after adjusting for peak dates and seed dates, we call "epistep.step" on every iteration
        - on vaccination dates, we call "epistep.vaccinate"
        - on the "birth day" dates, we call "epistep.age"
    """
    state = m.startState(cacheFile, cacheDate)
    trajectory = []
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
        trajectory.append(extState)

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating {curDate} (day {idx}:{day})")
            vaxdState = epistep.vaccinate(m, m.vaccRates, *state)
            state = vaxdState[0:6]
            trajectory[-1] = updateVaxCost(trajectory[-1], vaxdState[-1])

        if (curDate.month, curDate.day) == m.birthday:
            print(f"Aging population {curDate} (day {idx}:{day})")
            state = epistep.age(m, *state)

        curDate = curDate + timedelta(days=1)
        idx += 1
    print(f"End date {curDate} (day {idx}:{day})")
    return trajectory


def plot(m, trajectory):
    # We first plot dynamics
    summd = []
    d = 1
    for (S, E, Inf, R, V, day, *_) in trajectory:
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
         ambQaly, nomedQaly, hospQaly, lifeyrsLost) in trajectory:
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
         ambQaly, nomedQaly, hospQaly, lifeyrsLost) in trajectory:
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


def main():
    m = EpiData()
    config = configparser.ConfigParser()
    config.read("config.ini")
    try:
        endDate = date.fromisoformat(sys.argv[1])
    except:
        endDate = date.fromisoformat(config.get("Defaults", "lastBurntDate"))
    # we also allow to be given a cached data filename and the day that
    # corresponds
    if len(sys.argv) > 2:
        cached = sys.argv[2]
        cachedDate = date.fromisoformat(sys.argv[3])
        print(f"Cached data file {cached} for date {cachedDate}")
        ts = simulate(m, endDate, cached, cachedDate)
    else:
        ts = simulate(m, endDate)
    # plot(m, ts)
    # with open("./working_dir/reference_sim.pickle", "wb") as reference_file:
    #     pickle.dump(obj=ts, file=reference_file)

    with open("./working_dir/reference_sim.pickle", "rb") as reference_file:
        reference = pickle.load(file=reference_file)

    compare_trajectories(ref_trajectory=reference, actual_trajectory=ts)


def compare_trajectories(ref_trajectory, actual_trajectory):
    assert len(ref_trajectory) == len(actual_trajectory), "Error, different simulation lengths."
    for day_nr, (ref_day, actual_day) in enumerate(zip(ref_trajectory, actual_trajectory)):
        assert len(ref_day) == len(actual_day), f"Error, different number of compartments on day {day_nr}."
        for i, (ref_compartment, actual_compartment) in enumerate(zip(ref_day, actual_day)):
            assert jnp.allclose(ref_compartment, actual_compartment), \
                f"Error, {i}-th compartments differ on day {day_nr}."
    print("All correct.")


if __name__ == "__main__":
    # TODO: call, save stuff
    main()
