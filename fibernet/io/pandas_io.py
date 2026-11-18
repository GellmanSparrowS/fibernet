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
        DataFrame with columns: fiber_id, point_idx, x, y, z, radius,
        material_name, youngs_modulus, density. Empty networks retain these
        columns. If requested and crosslinks exist, has_crosslink marks
        each point on a fiber referenced by a crosslink.
    """
    _require_pandas()

    columns = ['fiber_id', 'point_idx', 'x', 'y', 'z', 'radius',
               'material_name', 'youngs_modulus', 'density']
    linked_positions = set()
    if include_crosslinks:
        linked_positions = {position for link in network.crosslinks
                            for position in (link.fiber_i, link.fiber_j)}
    if linked_positions:
        columns.append('has_crosslink')

    rows = []
    for fiber_index, fiber in enumerate(network.fibers):
        has_crosslink = fiber_index in linked_positions
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
            if linked_positions:
                row['has_crosslink'] = has_crosslink
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def from_dataframe(df: 'pd.DataFrame', material=None) -> FiberNetwork:
    """Create fiber network from pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x and y columns and a fiber_id column or named index
        level. If z is absent, points are padded with z=0 and the network
        dimension is 2; otherwise it is 3. Every row needs a fiber_id.
        Categorical groups follow category order and skip unused labels.
        Rows are sorted by point_idx when present.
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

    group_key = 'fiber_id'
    indexed_labels = group_key in df.index.names
    if indexed_labels and group_key in df.columns:
        raise ValueError("'fiber_id' is both an index level and a column label, which is ambiguous.")
    labels = df.index.get_level_values(group_key) if indexed_labels else df[group_key]
    if labels.isna().any():
        raise ValueError('fiber_id must be present for every DataFrame row')
    # Older pandas sorts observed categoricals in category order only if
    # marked ordered. Use a temporary key so the caller's dtype is preserved.
    if isinstance(labels, pd.CategoricalIndex):
        group_key = labels.as_ordered()
    elif isinstance(labels.dtype, pd.CategoricalDtype):
        group_key = labels.cat.as_ordered()

    supplied_material = material is not None
    if material is None:
        material = Material(name='default')

    network = FiberNetwork(dimension=3 if 'z' in df.columns else 2)
    coord_cols = [c for c in ['x', 'y', 'z'] if c in df.columns]

    for fiber_id, group in df.groupby(group_key, observed=True):
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
