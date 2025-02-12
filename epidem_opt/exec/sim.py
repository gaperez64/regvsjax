from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_trajectories

# jax.config.update("jax_enable_x64", True)


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
    epi_data = EpiData(config_path=Path("./config.ini"),
                epidem_data_path=Path("./epidem_data"),
                econ_data_path=Path("./econ_data"),
                qaly_data_path=Path("./qaly_data"),
                vaccination_rates_path=Path("./vaccination_rates"),)
    try:
        end_date = date.fromisoformat(sys.argv[1])
    except:
        end_date = epi_data.last_burnt_date
    # we also allow to be given a cached data filename and the day that
    # corresponds
    if len(sys.argv) > 2:
        cached = Path(sys.argv[2])
        cache_date = date.fromisoformat(sys.argv[3])
        print(f"Cached data file {cached} for date {cache_date}")
        ts = simulate_trajectories(epi_data=epi_data, end_date=end_date, cache_file=cached, cache_date=cache_date)
    else:
        ts = simulate_trajectories(epi_data, end_date)
    plot(epi_data, ts)


if __name__ == "__main__":
    # TODO: call, save stuff
    main()
