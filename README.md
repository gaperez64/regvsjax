# Optimal Vaccination Strategy

This repository contains two main parts. First, a Python3 reimplementation of
R.  Manansala's modification of the Belgian Health Care Knowledge Centre's
SEIRS model for Influenza. Second, code to find an optimal choice of
vaccination strategy parameters.

In both parts, JAX is used.

# NEW README (cool)

## Plotting

```
python -m epidem_opt.exec.plot --experiment_data ./experiment_data --program_name program_baseline
```

```
python -m epidem_opt.exec.plot --experiment_data ./experiment_data --program_name program_a24-c10-07
```

```
python -m epidem_opt.exec.plot --experiment_data ./experiment_data --program_path ./working_dir/modified_program.csv
```

## Summarising the vaccination rates

```
python -m epidem_opt.exec.find_vaccination_box --vacc_dir ./vaccination_rates --output_file ./vaccination_box.csv
```

## Burn-in step

```
python -m epidem_opt.exec.burn --experiment_data ./experiment_data --output_file ./experiment_data/burned_in.csv
```

## Optimisation step

```
python -m epidem_opt.exec.opti --experiment_data ./experiment_data --burn_in_file ./experiment_data/burned_in.csv --init_program program_a24-c10-07

```

## Merging vaccination programs into a single CSV

```
python -m epidem_opt.exec.merge_vacc_programs --vacc_dir ./experiment_data/vaccination_rates --output_file ./experiment_data/vacc_programs.csv
```

## Computing the costs for all vaccination programs

```
python -m epidem_opt.exec.cost_per_program --experiment_data ./experiment_data --burn_in_file ./experiment_data/burned_in.csv --output_file ./experimental_data/program_costs.csv
```

# OLD README (boring)

## Dependencies

A `requirements.txt` file is provided with all required packages to run the
code. If you are not using a (Mac) MX processor, you need `reqs-linux-x86_64.txt` instead.

We recommend the usage of a virtual environment together with `pip install -r requirements.txt` to ensure you have the right version of the
packages. For instance, run:

```
python3 -m venv virtenv
pip install -r reqs-linux-x86-64.txt
source virtenv/bin/activate
python sim.py
```

in the directory where you cloned the repo.

Several configuration values can be set in the `config.ini` file. These change
the behavior of the scripts described below.

## Testing

The `test.py` utility runs a comparison of the simulated dynamics against data
obtained from R. Manansala's model. It plots the differences as a final step.
We consider a discrepancy of ~20 individuals per compartment to be reasonable.
Anything larger than this should be considered a bug or a mismatch in
dates/rates/data in general.

## Simulating

The `sim.py` utility can be used to simulate (a deterministic version of) the
baseline situation. The script requires an (ISO-format) date that will be used
as end-date for the simulation. It can also take a filename and date for
cached data.

### Caching (the burn-in) data

You can use the `burn.py` script to simulate the model from start date to
burn-in date and to save the distribution of the population in
`data/afterBurnIn.csv`.

## Lower and upper bounds of vaccination rates

The `rate_min_max.py` script goes through all vaccination programs in the
`data` directory and computes uniform upper and lower bounds for the
vaccination rates. These are output in a file named `output.csv`. Intuitively,
the file represents a hypercube containing all considered vaccination
strategies!
