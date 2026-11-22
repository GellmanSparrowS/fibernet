"""Experimental direct prescribed-temperature 3D welded beam snapshots.

Importing this opt-in Python module does not load the numerical owner.
It does not dispatch saved data or provide an MCP or simulate mode.
"""
from typing import NamedTuple, Optional, Tuple

__all__ = ["ThermoelasticSnapshot3D", "solve_thermoelastic_network_snapshot_3d"]

_Row6 = Tuple[float, float, float, float, float, float]
_IdValues = Tuple[Tuple[int, float], ...]


class _ApplicabilityLimits(NamedTuple):
    endpoint_thermal_strain: float
    nodal_rotation: float
    transverse_slope_bound: float
    total_extreme_fiber_strain: float
    endpoint_elastic_extreme_fiber_strain: float
    twist_shear: float


class _EdgeApplicability(NamedTuple):
    edge_id: int
    endpoint_thermal_strain: float
    transverse_slope_bound: float
    total_extreme_fiber_strain: float
    endpoint_elastic_extreme_fiber_strain: float
    twist_shear: float
    thermal_strain_by_endpoint_id: _IdValues
    elastic_extreme_fiber_strain_by_endpoint_id: _IdValues


class _Applicability(NamedTuple):
    scope: str
    limits: _ApplicabilityLimits
    nodal_rotation_rad_by_id: _IdValues
    edge_diagnostics: Tuple[_EdgeApplicability, ...]


class ThermoelasticSnapshot3D(NamedTuple):
    """Owned immutable SI results; align node/edge rows to the identity tuples.

    The reaction row contains the six equilibrium residual components at
    each node. Fixed components are support reactions; free components
    are numerical residuals. Axial edge force/stress is positive in tension.
    """

    node_ids: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    displacement_m_and_rad: Tuple[_Row6, ...]
    reactions_N_and_Nm: Tuple[_Row6, ...]
    axial_force_N: Tuple[float, ...]
    axial_stress_Pa: Tuple[float, ...]
    mean_thermal_strain: Tuple[float, ...]
    elastic_energy_J: float
    edge_elastic_energy_J: Tuple[float, ...]
    element_energy_roundoff_bound_J: float
    scaled_stiffness_min_over_max: Optional[float]
    applicability: _Applicability


def _id_values(items):
    return tuple((int(key), float(value)) for key, value in items.items())


def _applicability(values):
    edges = []
    for eid, item in values["edge_diagnostics_by_id"].items():
        edges.append(_EdgeApplicability(
            int(eid),
            float(item["endpoint_thermal_strain"]),
            float(item["transverse_slope_bound"]),
            float(item["total_extreme_fiber_strain"]),
            float(item["endpoint_elastic_extreme_fiber_strain"]),
            float(item["twist_shear"]),
            _id_values(item["thermal_strain_by_endpoint_id"]),
            _id_values(item["elastic_extreme_fiber_strain_by_endpoint_id"]),
        ))
    return _Applicability(
        values["scope"],
        _ApplicabilityLimits(**values["limits"]),
        _id_values(values["nodal_rotation_rad_by_id"]),
        tuple(edges),
    )


def solve_thermoelastic_network_snapshot_3d(
    graph, *, temperature_K_by_node_id, stress_free_temperature_K,
    expansion_alpha_per_K_by_edge_id, constraints_by_node_id,
    loads_N_by_node_id=None, length_unit, applicability_limits
) -> ThermoelasticSnapshot3D:
    """Solve one experimental direct prescribed-temperature 3D frame snapshot.

    Use a native 3D StructureGraph with straight circular members and edge
    materials (E in Pa). Supply absolute Kelvin at every node, a positive
    reference Kelvin, and expansion per K at every edge. Only positions and
    radii use the declared geometry unit (m, mm or um). Sparse constraints
    are six SI values (m, m, m, rad, rad, rad), with None for a free
    component. Sparse applied forces are three newton components; no moments
    are accepted. Supply all six positive kinematic screening ceilings in
    applicability_limits; they are not calibrated validity limits.

    At most 96 nodes, 192 physical edges, and 384 free DOFs are accepted.
    This does not model temperature evolution, instability or feedback.
    """
    from ._thermoelastic_3d import _solve_thermoelastic_3d

    data = _solve_thermoelastic_3d(
        graph,
        temperature_K_by_node_id=temperature_K_by_node_id,
        stress_free_temperature_K=stress_free_temperature_K,
        expansion_alpha_per_K_by_edge_id=expansion_alpha_per_K_by_edge_id,
        constraints_by_node_id=constraints_by_node_id,
        loads_N_by_node_id=loads_N_by_node_id,
        length_unit=length_unit,
        applicability_limits=applicability_limits,
    )
    spectral_ratio = data["scaled_stiffness_min_over_max"]
    return ThermoelasticSnapshot3D(
        tuple(map(int, data["node_ids"])),
        tuple(map(int, data["edge_ids"])),
        tuple(tuple(map(float, row)) for row in data["displacement_m_and_rad"]),
        tuple(tuple(map(float, row)) for row in data["reactions_N_and_Nm"]),
        tuple(map(float, data["axial_force_N"])),
        tuple(map(float, data["axial_stress_Pa"])),
        tuple(map(float, data["mean_thermal_strain"])),
        float(data["elastic_energy_J"]),
        tuple(map(float, data["edge_elastic_energy_J"])),
        float(data["element_energy_roundoff_bound_J"]),
        None if spectral_ratio is None else float(spectral_ratio),
        _applicability(data["applicability"]),
    )
