"""Pandas integration for FiberNet.

Provides DataFrame export/import for easy data analysis and visualization.
Requires: pandas
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Any
from ..core.network import FiberNetwork
from ..core.fiber import Fiber
from ..core.material import Material

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def _require_pandas():
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required. Install with: pip install pandas")


def to_dataframe(network: FiberNetwork, include_crosslinks: bool = True) -> 'pd.DataFrame':
    """Convert fiber network to pandas DataFrame.

    Each row represents a point on a fiber centerline.

    Parameters
    ----------
    network : FiberNetwork
        The fiber network to convert.
    include_crosslinks : bool, optional
        Whether to include crosslink information, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: fiber_id, x, y, z, radius, material_name,
        youngs_modulus, density.
    """
    _require_pandas()

    rows = []
    for fiber in network.fibers:
        cl = np.asarray(fiber.centerline)
        n_pts = len(cl)
        dim = cl.shape[1] if cl.ndim > 1 else 1

        for i in range(n_pts):
            row = {
                'fiber_id': fiber.fiber_id,
                'point_idx': i,
                'x': float(cl[i, 0]) if dim >= 1 else 0.0,
                'y': float(cl[i, 1]) if dim >= 2 else 0.0,
                'z': float(cl[i, 2]) if dim >= 3 else 0.0,
                'radius': fiber.radius,
                'material_name': fiber.material.name,
                'youngs_modulus': fiber.material.youngs_modulus,
                'density': fiber.material.density,
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if include_crosslinks and network.crosslinks:
        cl_fiber_set = set()
        for cl in network.crosslinks:
            cl_fiber_set.add(cl.fiber_i)
            cl_fiber_set.add(cl.fiber_j)
        df['has_crosslink'] = df['fiber_id'].isin(cl_fiber_set)

    return df


def from_dataframe(df: 'pd.DataFrame', material=None) -> FiberNetwork:
    """Create fiber network from pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: fiber_id, x, y (at minimum). If z is absent,
        points are padded with z=0 and the network dimension is 2; otherwise
        the network dimension is 3. Rows are sorted by point_idx when present.
    material : Material, optional
        Material for all fibers, taking precedence over table material columns.
        If omitted, material columns are used when youngs_modulus is present;
        otherwise a default material is used. The supplied object is shared.

    Returns
    -------
    FiberNetwork
        Reconstructed fiber network. Fiber IDs are reassigned in group order;
        crosslinks, box, and network metadata are not reconstructed. DataFrame
        export includes z, so a 2D network's dimension is not preserved by
        a round trip.
    """
    _require_pandas()

    supplied_material = material is not None
    if material is None:
        material = Material(name='default')

    network = FiberNetwork(dimension=3 if 'z' in df.columns else 2)
    coord_cols = [c for c in ['x', 'y', 'z'] if c in df.columns]

    for fiber_id, group in df.groupby('fiber_id'):
        if 'point_idx' in group.columns:
            group = group.sort_values('point_idx')
        points = group[coord_cols].values.astype(float)
        if coord_cols == ['x', 'y']:
            points = np.column_stack((points, np.zeros(len(points))))
        radius = float(group['radius'].iloc[0]) if 'radius' in group.columns else 1.0

        mat = material
        if not supplied_material and 'youngs_modulus' in group.columns:
            mat = Material(
                name=group['material_name'].iloc[0] if 'material_name' in group.columns else 'imported',
                youngs_modulus=float(group['youngs_modulus'].iloc[0]),
                density=float(group['density'].iloc[0]) if 'density' in group.columns else 1000.0,
            )

        fiber = Fiber(centerline=points, radius=radius, material=mat, fiber_id=int(fiber_id))
        network.add_fiber(fiber)

    return network


def network_summary(network: FiberNetwork) -> 'pd.DataFrame':
    """Summary statistics per fiber.

    Returns
    -------
    pd.DataFrame
        One row per fiber with length, radius, tortuosity, material info.
    """
    _require_pandas()

    data = []
    for fiber in network.fibers:
        data.append({
            'fiber_id': fiber.fiber_id,
            'length': fiber.length,
            'tortuosity': fiber.tortuosity(),
            'radius': fiber.radius,
            'num_points': fiber.num_points,
            'material': fiber.material.name,
            'youngs_modulus': fiber.material.youngs_modulus,
            'density': fiber.material.density,
        })

    return pd.DataFrame(data)


def parametric_to_dataframe(
    params: Dict[str, np.ndarray],
    metrics: Dict[str, np.ndarray]
) -> 'pd.DataFrame':
    """Convert parametric study results to DataFrame for easy plotting.

    Parameters
    ----------
    params : dict
        Parameter names to value arrays.
    metrics : dict
        Metric names to value arrays.

    Returns
    -------
    pd.DataFrame
        Combined parameter + metric columns.
    """
    _require_pandas()
    combined = {**params, **metrics}
    return pd.DataFrame(combined)
