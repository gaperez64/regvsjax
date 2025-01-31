from datetime import date, timedelta
import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from epidem_opt.src.kce.epidata import EpiData
from epidem_opt.src import kce as epistep


# jax.config.update("jax_enable_x64", True)


def updateVaxCost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)


def simulate(m, endDate, dropBefore=date(year=2000, month=1, day=1)):
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

        extState = epistep.step(m, *state)
        state = extState[0:6]

        # TODO: call m.switchProgram("prog name") after an
        # appropriate number of days
        if curDate >= dropBefore:
            trajectories.append(extState)

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating {curDate} (day {idx}:{day})")
            vaxdState = epistep.vaccinate(m, m.vaccRates, *state)
            state = vaxdState[0:6]
            if curDate >= dropBefore:
                trajectories[-1] = updateVaxCost(trajectories[-1],
                                                 vaxdState[-1])

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


def aggregate(mine, regs, label, day):
    return [("R " + label, float(regs.sum()), int(day)),
            (label, float(mine.sum()), int(day))]


def plot(m, trajectories):
    # We first plot differences
    summd = []
    df = pd.read_csv("../../data/output_4yrs.csv", header=None)
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


if __name__ == "__main__":
    m = EpiData()
    endDate = date(year=2021, month=12, day=31)
    ts = simulate(m, endDate, date(year=2017, month=8, day=27))
    plot(m, ts)
    exit(0)
