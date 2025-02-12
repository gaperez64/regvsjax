import pickle
from datetime import date, timedelta
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from epidem_opt.src.kce.epidata import EpiData
# from epidem_opt.src import kce as epistep
import epidem_opt.src.kce.epistep as epistep


# jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def updateVaxCost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)


def simulate(m, endDate, dropBefore=date(year=2000, month=1, day=1)):
    # TODO: run this, and then refactor step by step
    #   - replace state with extState
    #   - maybe find a more elegant solution, where stuff is put in a dict or object.
    # TODO: make start state an ndarray
    state = m.startState()
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

        # (newS, newE, newInf, newR, newV, day + 1,
        #             ambCost, noMedCost, hospCost, vaxCost,
        #             ambQaly, noMedQaly, hospQaly, lifeyrsLost)
        extState = epistep.step(m, *state)

        # state = (newS, newE, newInf, newR, newV, day)
        state = extState[0:6]

        # TODO: call m.switchProgram("prog name") after an appropriate number of days

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating {curDate} (day {idx}:{day})")
            # vaxdState = (newS, newE, newInf, newR, newV, day, vaxCost)
            vaxdState = epistep.vaccinate(m, m.vaccRates, *state)
            state = vaxdState[0:6]
            if curDate >= dropBefore:
                extState = updateVaxCost(extState, vaxdState[-1])

        if curDate >= dropBefore:
            trajectories.append(extState)

        if (curDate.month, curDate.day) == m.birthday:
            print(f"Aging population {curDate} (day {idx}:{day})")
            state = epistep.age(m, *state)

        curDate = curDate + timedelta(days=1)
        idx += 1
    print(f"End date {curDate} (day {idx}:{day})")
    return trajectories


def compare(mine, regs, label, day):
    regs = jax.numpy.asarray(regs)
    diff = jax.numpy.max(jax.numpy.abs(mine - regs))
    return ("Diff " + label, float(diff), int(day))


# def aggregate(mine, regs, label, day):
#     return [("R " + label, float(regs.sum()), int(day)),
#             (label, float(mine.sum()), int(day))]


def plot(m, trajectories):
    # We first plot differences
    summd = []
    df = pd.read_csv("./data/output_4yrs.csv", header=None)
    print("Comparing compartment values with Reg's data")
    d = 1
    for (S, E, Inf, R, V, *_) in trajectories:
        summd.append(compare(S, df.iloc[0], "Susceptible", d))
        summd.append(compare(E, df.iloc[1], "Exposed", d))
        summd.append(compare(Inf, df.iloc[2], "Infectious", d))
        summd.append(compare(R, df.iloc[3], "Recovered", d))
        summd.append(compare(V, df.iloc[4], "Vaccinated", d))
        df = df.iloc[5:]
        d += 1
    df = pd.DataFrame(summd, columns=["Compartment", "Population", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="Population",
        hue="Compartment", style="Compartment"
    )
    plt.show()
    return


def compare_trajectories(ref_trajectory, actual_trajectory):
    assert len(ref_trajectory) == len(actual_trajectory), "Error, different simulation lengths."
    for day_nr, (ref_day, actual_day) in enumerate(zip(ref_trajectory, actual_trajectory)):
        assert len(ref_day) == len(actual_day), f"Error, different number of compartments on day {day_nr}."
        for i, (ref_compartment, actual_compartment) in enumerate(zip(ref_day, actual_day)):
            assert (ref_compartment == actual_compartment).all(), f"Error, {i}-th compartments differ on day {day_nr}."
    print("All correct.")


def main():
    m = EpiData(config_path=Path("./test_reference/config.ini"),
                epidem_data_path=Path("./test_reference/epidem_data"),
                econ_data_path=Path("./test_reference/econ_data"),
                qaly_data_path=Path("./test_reference/qaly_data"))
    endDate = date(year=2021, month=12, day=31)
    ts = simulate(m=m, endDate=endDate, dropBefore=date(year=2017, month=8, day=27))

    # with open("./working_dir/test_reference.pickle", "wb") as f:
    #     pickle.dump(ts, f)

    with open("./working_dir/test_reference.pickle", "rb") as f:
        ref_trajectory = pickle.load(f)

    compare_trajectories(actual_trajectory=ts, ref_trajectory=ref_trajectory)


if __name__ == "__main__":
    main()

    # plot(m, ts)
    # exit(0)
