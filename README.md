# Optimal Vaccination Strategy
This repository contains two main parts. First, a Python3 reimplementation of
R.  Manansala's modification of the Belgian Health Care Knowledge Centre's
SEIRS model for Influenza. Second, code to find an optimal choice of
vaccination strategy parameters.

In both parts, JAX is used.

## Dependencies
A `requirements.txt` file is provided with all required packages to run the
code. We recommend the usage of a virtual environment together with `pip
install -r requirements.txt` to ensure you have the right version of the
packages.

## Simulating
The `sim.py` utility can be used to simulate (a determinist version of) the
baseline situation.

## Lower and upper bounds of vaccination rates
The `rate_min_max.py` script goes through all vaccination programs in the
`data` directory and computes uniform upper and lower bounds for the
vaccination rates. These are output in a file named `output.csv`. Intuitively,
the file represents a hypercube containing all considered vaccination
strategies!
