from datetime import date, timedelta
import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kce.SEIRS import Model


# jax.config.update("jax_enable_x64", True)


def simulate(m, endDate):
    state = m.init()
    trajectories = []
    curDate = m.startDate
    while curDate <= endDate:
        (_, _, _, _, _, day) = state
        assert (m.startDate + timedelta(days=int(day))) == curDate

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating, year {curDate}")
            state = m.vaccinate(*state)
        if (curDate.month, curDate.day) == m.seedDate:
            print(f"Seeding infections, year {curDate}")
            state = m.seedInfs(*state)
        extState = m.step(*state)
        state = extState[0:6]
        if (curDate.month, curDate.day) == m.birthday:
            print(f"Aging population, year {curDate}")
            state = m.age(*state)

        # TODO: call m.switchProgram("prog name") after an
        # appropriate number of days
        trajectories.append(state)
        curDate = curDate + timedelta(days=1)
    return trajectories


def plot(m, trajectories):
    summd = []
    for (S, E, Inf, R, V, day) in trajectories:
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
        print(",".join([str(float(s)) for s in S]))
        print(",".join([str(float(e)) for e in E]))
        print(",".join([str(float(i)) for i in Inf]))
        print(",".join([str(float(r)) for r in R]))
        print(",".join([str(float(v)) for v in V]))
    df = pd.DataFrame(summd, columns=["Compartment", "Population", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="Population",
        hue="Compartment", style="Compartment"
    )
    plt.show()


if __name__ == "__main__":
    m = Model()
    endDate = date(year=2017, month=11, day=5)
    ts = simulate(m, endDate)
    plot(m, ts)
    exit(0)
