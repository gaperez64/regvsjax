import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kce.SEIRS import Model


def simulate(m, noYears):
    state = m.init()
    trajectories = [state]
    for _ in range(365 * noYears):
        (_, _, _, _, _, day) = state
        if day % 365 == m.seedDate:
            print(f"Seeding infections, year {day // 356 + 1}")
            state = m.seedInfs(*state)
        state = m.step(*state)
        if day % 365 == m.birthday:
            print(f"Aging population, year {day // 365 + 1}")
            state = m.age(*state)
        if day % 365 == m.vaccDate:
            print(f"Vaccinating, year {day // 365 + 1}")
            state = m.vaccinate(*state)
        # TODO: call m.switchProgram("prog name") after an
        # appropriate number of days
        trajectories.append(state)
    return trajectories


def plot(m, trajectories, noYears):
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
    df = pd.DataFrame(summd, columns=["Compartment", "Population", "Day"])
    # ax =
    sns.lineplot(
        data=df,
        x="Day", y="Population",
        hue="Compartment", style="Compartment"
    )
    # ax.vlines(x=[m.seedDate + i
    #              for i in range(0, 365 * noYears, 365)],
    #           ymin=0, ymax=m.totPop)
    plt.show()


if __name__ == "__main__":
    m = Model()
    noYears = 10
    ts = simulate(m, noYears)
    plot(m, ts, noYears)
    exit(0)
