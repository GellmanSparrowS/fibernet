"""Frozen 105-node example. Import/configuration do not advance a solver."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def prepare(level="fine"):
    """Return the authoritative public graph and explicit SI dynamic arguments."""
    import numpy as np
    from fibernet import StructureGraph, LinearViscoelasticMaterial
    fixture = json.loads((HERE / "fixtures/dynamic_viscoelastic_105.json").read_text())
    if level not in ("coarse", "medium", "fine"):
        raise ValueError("level must be coarse, medium, or fine")
    run = fixture["runs"][("coarse", "medium", "fine").index(level)]
    graph = StructureGraph.from_dict(fixture["graph"])
    nodes, edges = fixture["node_ids"], fixture["edge_ids"]
    return graph, dict(backend="taichi", mode="dynamics", arch="cpu", num_threads=1,
        material=LinearViscoelasticMaterial(**fixture["material"]),
        reference_areas_by_id=dict(zip(edges, fixture["areas_m2"])),
        rest_lengths_by_id=dict(zip(edges, fixture["rest_lengths_m"])),
        nodal_masses_by_id=dict(zip(nodes, fixture["nodal_masses_kg"])),
        initial_velocities_by_id=dict(zip(nodes, fixture["initial_velocities_m_per_s"])),
        fixed_nodes=fixture["fixed_rows"], external_force=np.zeros((len(nodes), 3)),
        state_units=fixture["units"], max_abs_strain=fixture["max_abs_strain"],
        dashpot=0., drag=0., dt=run["dt_s"], num_steps=run["intervals"],
        save_interval=run["save_interval"], record_state=True)


def run(level="fine"):
    """Explicit opt-in: run one retained timestep level; no installation or retry."""
    from fibernet import simulate
    graph, kwargs = prepare(level)
    return graph, simulate(graph, **kwargs)


if __name__ == "__main__":
    graph, kwargs = prepare()
    print(f"Prepared {graph.num_nodes} nodes and {graph.num_edges} edges; no simulation run.")
    print("Import run() explicitly to execute the fine level; see docs/DYNAMIC_AXIAL_VISCOELASTICITY.md.")
