import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
import seaborn as sns

from kce.SEIRS import Model


def testAge():
    m = Model()
    m.totPop = 10
    S = jnp.asarray([3.0, 3.0])
    E = jnp.asarray([0.0, 0.0])
    Inf = jnp.asarray([2.0, 2.0])
    R = jnp.asarray([0.0, 0.0])
    V = jnp.asarray([0.0, 0.0])
    day = 1
    state = (S, E, Inf, R, V, day)
    print(state)
    state = m.age(*state)
    print(state)
    state = m.age(*state)
    print(state)


def simulate():
    m = Model()
    (S, E, Inf, R, V, day) = m.init()
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
    # ts = simulate()
    # print("Done with simulations, trajectories acquired!")
    # plot(ts)
    testAge()
    exit(0)
