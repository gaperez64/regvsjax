from collections import namedtuple
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

# import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from kce.SEIRS import Model, EpidemParams


# jax.config.update("jax_enable_x64", True)


class TimeOfYear(namedtuple("TimeOfYear", ["month", "day"])):
    """
        Small type to represent a "time of the year", i.e. a date that occurs every year.
    """


@dataclass
class ImportantDates:
    """
        Contains key dates in the course of the simulation.
    """
    start_date: date
    peak: date
    birthday: TimeOfYear
    seed_date: TimeOfYear
    vacc_date: TimeOfYear
    end_date: date


def simulate(model: Model, important_dates: ImportantDates):
    # TODO: maybe move the "state" variable inside the model, and let it be owned by "step()"?
    model.set_vacc_program(prog="baseline")
    state = model.init()
    trajectories = []
    curDate = important_dates.start_date
    print(f"Start date {curDate}")
    while curDate <= important_dates.end_date:
        (_, _, _, _, _, day) = state
        assert (important_dates.start_date + timedelta(days=int(day))) == curDate

        if (curDate.month, curDate.day) == important_dates.seed_date:
            print(f"Seeding infections {curDate}")
            state = model.seedInfs(*state)

        extState = model.step(*state)
        state = extState[0:6]

        # TODO: call m.switchProgram("prog name") after an
        # appropriate number of days
        trajectories.append(state)

        if (curDate.month, curDate.day) == important_dates.vacc_date:
            print(f"Vaccinating {curDate}")
            state = model.vaccinate(*state)
        if (curDate.month, curDate.day) == important_dates.birthday:
            print(f"Aging population {curDate}")
            state = model.age(*state)

        curDate = curDate + timedelta(days=1)
    print(f"End date {curDate}")
    return trajectories


def plot(sim_traj: Iterable, reference_data: Path):
    """
        Plot the specified trajectories.

        Parameters
        ----------
        sim_traj : Iterable
            The trajectories produced by the simulation.
        reference_data : Path
            The reference data with which we wish to compare the simulation results.
    """
    summd = []
    df = pd.read_csv(reference_data, header=None)
    for (S, E, Inf, R, V, day) in sim_traj:
        # daterng = (m.startDate - m.peak).days + day
        # z = 1 + m.delta * np.sin(2 * np.pi * (daterng / 365))
        # print(f"z = {z}, daterng = {daterng}")

        entry = ("Susceptible", float(S.sum()), int(day))
        summd.append(entry)
        entry = ("Reg's S", float(df.iloc[0].sum()), int(day))
        summd.append(entry)
        # x = abs(summd[-1][1] - summd[-2][1])
        # if x > 99.0:
        #     pass
        # print(f"Discrepancy of {x} on day {day}")

        entry = ("Exposed", float(E.sum()), int(day))
        summd.append(entry)
        entry = ("Reg's E", float(df.iloc[1].sum()), int(day))
        summd.append(entry)

        entry = ("Infectious", float(Inf.sum()), int(day))
        summd.append(entry)
        entry = ("Reg's I", float(df.iloc[2].sum()), int(day))
        summd.append(entry)

        entry = ("Recovered", float(R.sum()), int(day))
        summd.append(entry)
        entry = ("Reg's R", float(df.iloc[3].sum()), int(day))
        summd.append(entry)

        entry = ("Vaccinated", float(V.sum()), int(day))
        summd.append(entry)
        entry = ("Reg's V", float(df.iloc[4].sum()), int(day))
        summd.append(entry)

        df = df.iloc[5:]

        # print(",".join([str(float(s)) for s in S]))
        # print(",".join([str(float(e)) for e in E]))
        # print(",".join([str(float(i)) for i in Inf]))
        # print(",".join([str(float(r)) for r in R]))
        # print(",".join([str(float(v)) for v in V]))
    df = pd.DataFrame(summd, columns=["Compartment", "Population", "Day"])
    sns.lineplot(
        data=df,
        x="Day", y="Population",
        hue="Compartment", style="Compartment"
    )
    plt.show()


def main():
    important_dates = ImportantDates(
        # FIXME: This is a random nr. of days after the (fixed)
        # FIXME: these values are arguably not part of the model
        peak=date(year=2016, month=9, day=21),
        start_date=date(year=2017, month=8, day=27),
        # FIXME:the peak/reference day above should be randomized too
        # birthday = (8, 31),  # End of August
        birthday=TimeOfYear(month=8, day=31),  # End of August
        seed_date=TimeOfYear(month=8, day=31),  # End of August
        # seed_date = (8, 31),  # End of August
        # FIXME: the seeding date above is also randomized
        vacc_date=TimeOfYear(month=10, day=10),  # October 10
        # vacc_date = (10, 10),
        end_date = date(year=2019, month=1, day=1)
    )
    epidem_params = EpidemParams()
    model = Model(epidem_params=epidem_params)
    traj = simulate(model, important_dates)
    plot(traj, reference_data=Path("./data/output_853days.csv"))
    # exit(0)


if __name__ == "__main__":
    main()
