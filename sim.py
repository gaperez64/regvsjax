from datetime import date, timedelta
import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kce.SEIRS import Model


jax.config.update("jax_enable_x64", True)


def updateVaxCost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)


def simulate(m, endDate, dropBefore=date(year=2000, month=1, day=1)):
    state = m.init()
    trajectories = []
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

        # TODO: call m.switchProgram("prog name") after an
        # appropriate number of days
        if curDate >= dropBefore:
            trajectories.append(extState)

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating {curDate} (day {day})")
            vaxdState = m.vaccinate(*state)
            state = vaxdState[0:6]
            if curDate >= dropBefore:
                trajectories[-1] = updateVaxCost(trajectories[-1],
                                                 vaxdState[-1])

        if (curDate.month, curDate.day) == m.birthday:
            print(f"Aging population {curDate} (day {day})")
            state = m.age(*state)

        curDate = curDate + timedelta(days=1)
    print(f"End date {curDate} (day {day})")
    return trajectories


def plot(m, trajectories):
    # We first plot dynamics
    summd = []
    df = pd.read_csv("data/output_493days.csv", header=None)
    for (S, E, Inf, R, V, day, *_) in trajectories:
        entry = ("Susceptible", float(S.sum()), int(day))
        summd.append(entry)
        entry = ("Exposed", float(E.sum()), int(day))
        summd.append(entry)
        entry = ("Infectious", float(Inf.sum()), int(day))
        summd.append(entry)
        entry = ("Recovered", float(R.sum()), int(day))
        summd.append(entry)
        entry = ("Vaccinated", float(V.sum()), int(day))
        summd.append(entry)
        # Regina's data to compare against
        rtitles = ["R Susceptible", "R Exposed",
                   "R Infectious", "R Recovered",
                   "R Vaccinated"]
        ridcs = [0, 1, 2, 3, 4]
        for (t, i) in zip(rtitles, ridcs):
            entry = (t, float(df.iloc[i].sum()), int(day))
            summd.append(entry)
        # Drop the rows we used
        df = df.iloc[5:]
    df = pd.DataFrame(summd, columns=["Compartment", "Population", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="Population",
        hue="Compartment", style="Compartment"
    )
    plt.show()

    # Now we plot costs
    summd = []
    df = pd.read_csv("econ_data/output_493days_cost.csv", header=None)
    # Drop label column
    df = df.iloc[:, 1:]
    for (*_, day, ambCost, nomedCost, hospCost, vaxCost,
         ambQaly, nomedQaly, hospQaly, lifeyrsLost) in trajectories:
        entry = ("Ambulatory", float(ambCost.sum()), int(day))
        summd.append(entry)
        entry = ("No med care", float(nomedCost.sum()), int(day))
        summd.append(entry)
        entry = ("Hospital", float(hospCost.sum()), int(day))
        summd.append(entry)
        entry = ("Vaccine", float(vaxCost.sum()), int(day))
        summd.append(entry)

        # Regina's data to compare against
        rtitles = ["R Ambulatory", "R No med care",
                   "R Hospital", "R Vaccination"]
        ridcs = [0, 1, 2, 4]
        for (t, i) in zip(rtitles, ridcs):
            entry = (t, float(df.iloc[i].sum()), int(day))
            summd.append(entry)
        # Drop the rows we used
        df = df.iloc[5:]
    df = pd.DataFrame(summd, columns=["Cost", "Euros", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="Euros",
        hue="Cost", style="Cost",
    )
    plt.yscale("log")
    plt.show()

    # Now we plot qaly
    summd = []
    df = pd.read_csv("econ_data/output_493days_qaly.csv", header=None)
    # Drop label column
    df = df.iloc[:, 1:]
    for (*_, day, ambCost, nomedCost, hospCost, vaxCost,
         ambQaly, nomedQaly, hospQaly, lifeyrsLost) in trajectories:
        entry = ("Ambulatory", float(ambQaly.sum()), int(day))
        summd.append(entry)
        entry = ("No med care", float(nomedQaly.sum()), int(day))
        summd.append(entry)
        entry = ("Hospital", float(hospQaly.sum()), int(day))
        summd.append(entry)
        entry = ("Life years", float(lifeyrsLost.sum()), int(day))
        summd.append(entry)

        # Regina's data to compare against
        rtitles = ["R Ambulatory", "R No med care",
                   "R Hospital", "R Life years"]
        ridcs = [0, 1, 2, 3]
        for (t, i) in zip(rtitles, ridcs):
            entry = (t, float(df.iloc[i].sum()), int(day))
            summd.append(entry)
        # Drop the rows we used
        df = df.iloc[5:]
    df = pd.DataFrame(summd, columns=["QALY", "QALY Units", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="QALY Units",
        hue="QALY", style="QALY",
    )
    plt.yscale("log")
    plt.show()

    return


if __name__ == "__main__":
    m = Model()
    endDate = date(year=2019, month=1, day=1)
    ts = simulate(m, endDate, date(year=2017, month=8, day=27))
    plot(m, ts)
    exit(0)
