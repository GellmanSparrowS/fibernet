"""
LAMMPS data file I/O.

The legacy FiberNetwork writer is unsupported and fails before file I/O.
The legacy reader also fails before file I/O; original inputs are untouched.
"""

import numpy as np
from typing import Optional, Dict
from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material


def to_lammps(
    network: FiberNetwork,
    filename: str,
    atom_style: str = "bond",
    bead_spacing: float = None,
    bond_type: str = "harmonic",
    units: str = "real",
) -> str:
    """Unsupported legacy FiberNetwork writer; rejects without touching output.

    Historical atom records, coefficient sections and material/unit mapping
    were not a validated LAMMPS contract. Neither unlinked inputs nor choosing
    another atom_style/bond_type establishes a supported physical model.
    Preserve source data using FiberNetwork.save_json. The separate explicit
    StructureGraph backend requires caller-defined units, masses and bonds;
    it is not an automatic conversion or equivalent replacement.
    """
    raise NotImplementedError(
        'legacy to_lammps export is unsupported: atom records, bond coefficients '
        'and units/material mapping are not validated; no file was opened. '
        'Preserve FiberNetwork with save_json. For a separately prepared '
        'StructureGraph, use the explicit LAMMPS backend with length_unit, '
        'masses_kg and bond parameters; no automatic conversion is performed.'
    )


def from_lammps(filename: str, bead_spacing: float = 1.0) -> FiberNetwork:
    """Unsupported legacy reader; rejects before opening the input file.

    Historical parsing and chain reconstruction could silently change positions,
    discard atoms/bonds and invent welded connections. Keep the original LAMMPS
    file. Native JSON preserves an original FiberNet graph only when one exists;
    it cannot recover information lost in an earlier import. A future explicit
    geometry-only reader needs its own supported-format and topology contract.
    """
    raise NotImplementedError(
        'legacy from_lammps import is unsupported: atom-style parsing and '
        'topology reconstruction can silently lose or invent data; no input '
        'file was opened. Retain the original LAMMPS file. If an original '
        'FiberNet graph exists, preserve it with native JSON; JSON cannot '
        'recover data lost by an earlier import. There is no automatic '
        'replacement importer.'
    )
