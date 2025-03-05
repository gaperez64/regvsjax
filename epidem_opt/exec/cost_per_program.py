import argparse
from pathlib import Path

import pandas as pd

from epidem_opt.src.epidata import EpiData
from epidem_opt.src.simulator import simulate_cost, date_to_ordinal_set
from epidem_opt.src.vacc_programs import get_all_vaccination_programs_from_file


def main():
    arg_parser = argparse.ArgumentParser(prog="Utility to optimise the vaccination behaviour.")
    arg_parser.add_argument("--experiment_data", type=str, required=True,
                            help="Directory with the vaccination information. It should have sub-folders 'epi_data', "
                                 "'econ_data', 'vaccination_rates', 'qaly_data'.")
    arg_parser.add_argument('--burn_in_file', type=str, required=True,
                            help="CSV file with the results of the burn-in step.")
    arg_parser.add_argument('--output_file', type=str, required=True,
                            help="Output file with the costs per program.")

    args = arg_parser.parse_args()

    experiment_data = Path(args.experiment_data)
    if not experiment_data.is_dir():
        print(f"Error: directory '{experiment_data}' does not exist.")
        exit(1)

    burn_in_file = Path(args.burn_in_file)
    if not burn_in_file.is_file():
        print(f"Error: burn-in file '{burn_in_file}' does not exist.")
        exit(1)

    output_file = Path(args.output_file)
    if output_file.is_file():
        print("Error: Output file already exists.")
        exit(1)

    epi_data = EpiData(
        config_path=experiment_data / "config.ini",
        epidem_data_path=experiment_data / "epidem_data",
        econ_data_path=experiment_data / "econ_data",
        qaly_data_path=experiment_data / "qaly_data",
        vaccination_rates_path=experiment_data / "vaccination_rates"
    )

    # we start the vaccination simulation with the output of the burn-in step
    start_state = epi_data.start_state(
        saved_state_file=burn_in_file,
        saved_date=epi_data.last_burnt_date
    )  # (S, E, I, R, V, day) with 100 age groups.

    # maxx_vacc = jnp.ones(shape=(100,))
    # anti_vacc = jnp.zeros(shape=(100,))

    start_date = epi_data.last_burnt_date
    end_date = epi_data.end_date

    vacc_rates = get_all_vaccination_programs_from_file(vacc_programs=experiment_data / "vacc_programs.csv")

    program_costs: dict[str, int] = dict()
    for i, (program_name, vacc_program) in enumerate(vacc_rates.items()):
        print(f"[{i+1}/{len(vacc_rates)}] Simulating program '{program_name}'")
        cost = simulate_cost(
            vacc_program,
            epi_data=epi_data,
            epi_state=start_state,
            start_date=start_date.toordinal(),
            end_date=end_date.toordinal(),
            vacc_dates=lambda x: x in date_to_ordinal_set(epi_data.vacc_date,
                                                          epi_data.last_burnt_date,
                                                          epi_data.end_date),
            peak_dates=lambda x: x in date_to_ordinal_set(epi_data.peak_date,
                                                          epi_data.last_burnt_date,
                                                          epi_data.end_date),
            seed_dates=lambda x: x in date_to_ordinal_set(epi_data.seed_date,
                                                          epi_data.last_burnt_date,
                                                          epi_data.end_date),
            birth_dates=lambda x: x in date_to_ordinal_set(epi_data.birthday,
                                                           epi_data.last_burnt_date,
                                                           epi_data.end_date)
        )

        program_costs[program_name] = cost

        if i >= 4:
            break

    df = pd.DataFrame(program_costs.items(), columns=['Program Name', 'Cost'])
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
