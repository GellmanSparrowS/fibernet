"""
I/O interoperability module.

Supports import/export to:
- LAMMPS data files
- VTK (Paraview/VisIt)
- CIF (crystallographic)
- PDB (protein)
- GMSH (mesh)
- XYZ (simple atomic)
- Pandas DataFrame (analysis)
"""

from fibernet.io.lammps import to_lammps, from_lammps
from fibernet.io.vtk import to_vtk
from fibernet.io.xyz import to_xyz
from fibernet.io.pdb import to_pdb, from_pdb
from fibernet.io.gmsh import to_gmsh
from fibernet.io.pandas_io import (
    to_dataframe,
    from_dataframe,
    network_summary,
    parametric_to_dataframe,
)

__all__ = [
    "to_lammps", "from_lammps",
    "to_vtk",
    "to_xyz",
    "to_pdb", "from_pdb",
    "to_gmsh",
    "to_dataframe", "from_dataframe",
    "network_summary", "parametric_to_dataframe",
]

# Mesh export
from .mesh_export import export_stl, export_obj, export_ply
__all__.extend([
    "export_stl", "export_obj", "export_ply"
])

# FEA export
from .fea_export import (
    export_to_abaqus,
    export_to_ansys,
    export_to_gmsh as export_to_gmsh_fea,
    export_to_lammps as export_to_lammps_fea,
)
__all__.extend([
    "export_to_abaqus",
    "export_to_ansys",
    "export_to_gmsh_fea",
    "export_to_lammps_fea",
])
