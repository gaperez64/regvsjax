from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kce.SEIRS import Model


# jax.config.update("jax_enable_x64", True)


def simulate(m, endDate):
    state = m.init()
    trajectories = []
    curDate = m.startDate
    print(f"Start date {curDate}")
    while curDate <= endDate:
        (_, _, _, _, _, day) = state
        assert (m.startDate + timedelta(days=int(day))) == curDate

        if (curDate.month, curDate.day) == m.seedDate:
            print(f"Seeding infections {curDate}")
            state = m.seedInfs(*state)

        extState = m.step(*state)
        state = extState[0:6]

        # TODO: call m.switchProgram("prog name") after an
        # appropriate number of days
        trajectories.append(extState)

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating {curDate}")
            state = m.vaccinate(*state)
        if (curDate.month, curDate.day) == m.birthday:
            print(f"Aging population {curDate}")
            state = m.age(*state)

        curDate = curDate + timedelta(days=1)
    print(f"End date {curDate}")
    return trajectories


def plot(m, trajectories):
    # We first plot dynamics
    summd = []
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
    day = 1
    while df.shape[0]:
        rtitles = ["R Ambulatory", "R No med care",
                   "R Hospital", "R Vaccination"]
        ridcs = [0, 1, 2, 4]
        for (t, i) in zip(rtitles, ridcs):
            entry = (t, float(df.iloc[i].sum()), day)
            summd.append(entry)
        # Drop the rows we used
        df = df.iloc[5:]
        # Increase day count
        day += 1
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
    day = 1
    while df.shape[0]:
        rtitles = ["R Ambulatory", "R No med care",
                   "R Hospital", "R Life years"]
        ridcs = [0, 1, 2, 3]
        for (t, i) in zip(rtitles, ridcs):
            entry = (t, float(df.iloc[i].sum()), day)
            summd.append(entry)
        # Drop the rows we used
        df = df.iloc[5:]
        # Increase day count
        day += 1
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
    ts = simulate(m, endDate)
    plot(m, ts)
    exit(0)
