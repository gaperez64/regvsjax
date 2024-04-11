import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kce.SEIRS import Model


# @jax.jit
def simulate():
    m = Model()
    (S, E, Inf, R, V, day) = m.init()
    state = (S, E, Inf, R, V, day)
    trajectories = [state]
    for i in range(365 * 2):
        if i % 365 == m.seedDate - 1:
            print(f"Seeding infections, year {i // 356 + 1}")
            state = m.seedInfs(*state)
        state = m.step(*state)
        if i % 365 == m.birthday - 1:
            print(f"Aging population, year {i // 365 + 1}")
            state = m.age(*state)
        if i % 365 == m.vaccDate - 1:
            print(f"Vaccinating the pro-vaxxers, year {i // 365 + 1}")
            state = m.vaccinate(*state)
        trajectories.append(state)
    return trajectories


def plot(trajectories):
    summd = []
    for (S, E, Inf, R, V, day) in trajectories:
        entry = ("Susceptible", float(S.sum()), day)
        summd.append(entry)
        entry = ("Exposed", float(E.sum()), day)
        summd.append(entry)
        entry = ("Infectious", float(Inf.sum()), day)
        summd.append(entry)
        entry = ("Recovered", float(R.sum()), day)
        summd.append(entry)
        entry = ("Vaccinated", float(V.sum()), day)
        summd.append(entry)
    df = pd.DataFrame(summd, columns=["Compartment", "Population", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="Population",
        hue="Compartment", style="Compartment"
    )
    plt.show()


if __name__ == "__main__":
    ts = simulate()
    print("Done with simulations, trajectories acquired!")
    # plot(ts)
    exit(0)
