import pandas as pd
import seaborn as sns

from kce.SEIRV import Model


def simulate():
    m = Model()
    (S, E, Inf, R, V) = m.init()
    day = 279  # 9 * 31 ~ End of September
    state = (S, E, Inf, R, V, day)
    state = m.seedInfs(*state)
    trajectories = [state]
    for _ in range(365):
        state = m.step(*state)
        trajectories.append(state)
    return trajectories


def plot(trajectories):
    summd = []
    for (S, E, Inf, R, V, day) in trajectories:
        entry = ("Susceptible", S.sum(), day)
        summd.append(entry)
        entry = ("Exposed", E.sum(), day)
        summd.append(entry)
        entry = ("Infectious", Inf.sum(), day)
        summd.append(entry)
        entry = ("Recovered", R.sum(), day)
        summd.append(entry)
        entry = ("Vaccinated", V.sum(), day)
        summd.append(entry)
    df = pd.DataFrame(summd, columns=["Compartment", "Population", "Day"])
    print(df)
    sns.relplot(
        data=df, x="Day", y="Population"
    )


if __name__ == "__main__":
    ts = simulate()
    print("Done with simulations, trajectories acquired!")
    plot(ts)
    exit(0)
