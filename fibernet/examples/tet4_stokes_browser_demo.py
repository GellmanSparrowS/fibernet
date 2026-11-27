"""Generate and export a fixed-pressure flow example for the local browser."""

import argparse
from dataclasses import replace
from importlib.util import find_spec
from itertools import product
import json
import os
from pathlib import Path

from fibernet.examples.tet4_periodic_pore import _scaffold_voxels, _voxel_bundle
from fibernet.sim.tet4_open_stokes import (
    solve_open_pressure_stokes,
    write_open_stokes_velocity_vtu,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for velocity.vtu and summary.json")
    parser.add_argument("--tier", choices=("small", "reference"), default="small",
                        help="small verification case or refined pore")
    parser.add_argument("--reverse", action="store_true", help="apply pressure from high x")
    args = parser.parse_args(argv)
    output = args.output.expanduser()
    if os.path.lexists(output):
        parser.error("--output must not exist")
    if not output.parent.is_dir():
        parser.error("--output must have an existing parent directory")
    if find_spec("meshio") is None:
        parser.error("the example needs optional Meshio; install meshio==5.3.5")
    scale = 1 if args.tier == "small" else 2
    root = set(product(range(8), range(7), range(7))) - _scaffold_voxels()
    # Both tiers represent the same duct and voxelized internal network.
    offsets = list(product(range(scale), repeat=3))
    voxels = {tuple(scale*i+j for i,j in zip(voxel, offset))
              for voxel in root for offset in offsets}
    pore = _voxel_bundle(voxels)
    pore = replace(pore, points=pore.points / scale)
    dp = -0.0035 if args.reverse else 0.0035
    mu = 0.001
    result = solve_open_pressure_stokes(
        pore, pressure_drop_Pa=dp, viscosity_Pa_s=mu,
        solver="direct" if scale == 1 else "amg", include_quadratic_velocity=True,
    )
    output.mkdir()
    write_open_stokes_velocity_vtu(output / "velocity.vtu", result.quadratic_velocity)
    summary = {
        "case": "fixed geometry; three internal paths and two crosslinks",
        "tier": args.tier,
        "tetrahedra": len(pore.tetrahedra),
        "pressure_drop_Pa": dp,
        "viscosity_Pa_s": mu,
        "axial_flow_m3_s": result.axial_flow_m3_s,
        "cap_mass_relative_error": result.cap_mass_relative_error,
        "dissipated_power_W": result.dissipated_power_W,
        "p2_points": len(result.quadratic_velocity.points_m),
    }
    payload = json.dumps(summary, indent=2)
    (output / "summary.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
