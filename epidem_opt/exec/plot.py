import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_trajectories
from epidem_opt.src.vacc_programs import get_vacc_program, read_vacc_program


def plot(epi_data: EpiData, trajectory, output_path: Path, filename_prefix: str):
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
    plt.savefig(output_path / f"{filename_prefix}_dynamics.png")
    plt.close()
    # plt.show()

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
    plt.savefig(output_path / f"{filename_prefix}_cost.png")
    plt.close()
    # plt.show()

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
    plt.savefig(output_path / f"{filename_prefix}_qaly.png")
    plt.close()
    # plt.show()

    print(f"The price of it all = {total_cost}")

    return


def _print_diff_(trajectory):
    (S_old, E_old, Inf_old, R_old, V_old, *_) = trajectory[0]
    tot_pop_old = S_old.sum() + E_old.sum() + Inf_old.sum() + R_old.sum() + V_old.sum()
    for (S, E, Inf, R, V, *_) in trajectory:
        tot_pop = S.sum() + E.sum() + Inf.sum() + R.sum() + V.sum()
        # assert tot_pop == tot_pop_old
        print("Diff:", tot_pop - tot_pop_old)
        tot_pop_old = tot_pop


def main():
    arg_parser = argparse.ArgumentParser(prog="Utility to simulate the epidemiological behaviour of a population "
                                              "under a certain baseline vaccination program.")
    arg_parser.add_argument("--experiment_data", type=str, required=True,
                            help="Directory with the vaccination information. It should have sub-folders 'epi_data', "
                                 "'econ_data', 'vaccination_rates', 'qaly_data', as well as a file 'config.ini'.")

    vacc_program_options = arg_parser.add_mutually_exclusive_group(required=True)
    vacc_program_options.add_argument('--program_name', default=None, type=str,
                                      help="Name of the vaccination program in the summary CSV")
    vacc_program_options.add_argument('--program_path', default=None, type=str,
                                      help="Path to a CSV file with a single vaccination program.")

    args = arg_parser.parse_args()
    experiment_data = Path(args.experiment_data)
    if not experiment_data.is_dir():
        print(f"Error: directory '{experiment_data}' does not exist.")
        exit(1)

    if args.program_path is not None:
        program_path = Path(args.program_path)
        _, vacc_rates = read_vacc_program(vacc_program_path=program_path)
        plot_prefix = program_path.stem
    elif args.program_name is not None:
        init_program_name = args.program_name
        plot_prefix = init_program_name
        vacc_rates_path = experiment_data / "vacc_programs.csv"
        vacc_rates = get_vacc_program(vacc_rates_path, program_name=init_program_name)
    else:
        raise ValueError("Error, missing argument!")

    epi_data = EpiData(
        config_path=experiment_data / "config.ini",
        epidem_data_path=experiment_data / "epidem_data",
        econ_data_path=experiment_data / "econ_data",
        qaly_data_path=experiment_data / "qaly_data",
        vacc_rates=vacc_rates
    )

    trajectory = simulate_trajectories(epi_data=epi_data, begin_date=epi_data.start_date,
                                       end_date=epi_data.last_burnt_date,
                                       start_state=epi_data.start_state(saved_state_file=None, saved_date=None))

    # print the loss in total population
    _print_diff_(trajectory=trajectory)

    # save plots to PNG
    plot(epi_data=epi_data, trajectory=trajectory, output_path=Path("./working_dir"), filename_prefix=plot_prefix)


if __name__ == "__main__":
    main()
