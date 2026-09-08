# Bicycle Lane Planning BP/BPC Code

This repository is a minimal open-source release of the code needed to run:

- the BP/BPC algorithm used in the manuscript;
- the Bagloee, Sarvi, and Wallace (2016) inspired branch-and-bound baseline;
- the three retained benchmark network datasets: Sioux Falls, Anaheim, and Eastern Massachusetts.

## Repository Contents

```text
.
├── main_run.py                 # BP/BPC experiment entry point
├── B_and_B.py                  # Branch-and-bound search for the BP/BPC algorithm
├── OPT_HPR.py                  # HPR optimization model
├── UB_LB.py                    # Upper/lower bound routines
├── Network.py                  # Sparse graph and shortest-path utilities
├── read_.py                    # TNTP network and OD-demand reader
├── baseline_run.py             # Command-line runner for the retained baseline
├── baseline_bagloee_bb.py      # Bagloee et al. (2016) inspired baseline implementation
├── baseline_common.py          # Shared baseline parsing, grouping, and evaluator utilities
├── SiouxFalls/                 # Full Sioux Falls benchmark data retained
├── Anaheim/                    # Full Anaheim benchmark data retained
└── Eastern-Massachusetts/      # Full Eastern Massachusetts benchmark data retained
```

## Requirements

The code requires Python 3 and a working Gurobi installation/license.

Install the Python packages used by the retained code. Use a compatible NumPy/SciPy pair; for the SciPy 1.10.x stack, NumPy must stay below 2:

```bash
pip install "numpy<2" scipy matplotlib igraph gurobipy
```

`gurobipy` also requires a valid local Gurobi license.

## Run the BP/BPC Algorithm

The default script runs the three retained networks with the paper-style batch settings currently defined at the bottom of `main_run.py`:

```bash
python main_run.py
```

By default this evaluates:

- networks: `1` = Sioux Falls, `2` = Anaheim, `3` = Eastern Massachusetts;
- budget share: `0.5`;
- candidate groups: `20`;
- group length: `1`;
- demand class count: `1`;
- value cut: disabled.

The script appends generated results to `results.csv`. That output file is intentionally not included in this open-source release.

## Run the Bagloee et al. Baseline

Run the retained Bagloee, Sarvi, and Wallace (2016) inspired baseline on all three retained networks:

```bash
python baseline_run.py --networks sf,ema,anaheim --methods bagloee_bb --groups 20 --group-links 1 --budget-share 0.5
```

The baseline appends generated rows to `bagloee_bb_baseline_results_with_relative.csv`. That output file is also intentionally not included.

The baseline corresponds to:

S. A. Bagloee, M. Sarvi, and M. Wallace. Bicycle lane priority: Promoting bicycle as a green mode even in congested urban area. Transportation Research Part A: Policy and Practice, 87:102-121, 2016. doi:10.1016/j.tra.2016.03.004.

## Data

The retained network folders contain the full local copies of the three benchmark datasets used by the scripts:

- `SiouxFalls`
- `Anaheim`
- `Eastern-Massachusetts`

These network data files are third-party benchmark data, originally distributed through the Transportation Networks/TNTP collection:

https://github.com/bstabler/TransportationNetworks

The MIT license in this repository applies to the code authored for this project. Third-party data remains subject to its original source terms.

## License

This code is released under the MIT License. See `LICENSE`.
