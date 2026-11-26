# SPDX-License-Identifier: MIT
"""Source-tree entry point for :mod:`fibernet.examples.tet4_iterative_mechanoelectric_sweep`."""

from fibernet.examples.tet4_iterative_mechanoelectric_sweep import (
    main as main,
    run as run,
    run_initial_slope as run_initial_slope,
    _power_weighted_strain as _power_weighted_strain,
)


if __name__ == "__main__":
    main()
