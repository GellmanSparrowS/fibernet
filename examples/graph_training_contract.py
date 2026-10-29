"""Bind graph inputs, an explicit group split, and training degree statistics.

The three synthetic specimens demonstrate the metadata API, not a learning
experiment. Run from a source checkout containing graph_training_contract:
    python -m examples.graph_training_contract
"""
import json

from fibernet.core.structure_graph import StructureGraph
from fibernet.ml.graph_data import graph_batch
from fibernet.ml.graph_training_contract import (
    FrozenRecord,
    bind_graph_inputs,
    bind_explicit_group_split,
    training_degree_histogram,
    validate_pna_degree_admission,
)


def make_source(graph_id, length_m):
    graph = StructureGraph(dimension=2)
    left = graph.add_node([0.0, 0.0, 0.0])
    right = graph.add_node([length_m, 0.0, 0.0])
    graph.add_node([0.0, length_m, 0.0])  # Keep an isolated node.
    graph.add_edge(left, right, radius=1.0e-5)
    graph.add_edge(left, right, radius=2.0e-5, allow_parallel=True)

    # Each sample has its own one-graph snapshot. Physical edge IDs are retained.
    batch = graph_batch(
        [graph], graph_ids=[graph_id], length_unit="m",
        coordinate_frame="specimen_xy",
        edge_features=("radius", "chord_length"),
    )
    return {
        "batch": batch,
        "external": {
            "units": {"conductivity_S_per_m": "S/m"},
            "rows": {
                int(edge_id): {
                    "conductivity_S_per_m": {
                        "value": 1.0e4, "available": True,
                    }
                }
                for edge_id in batch.edge_ids
            },
        },
        "conditions": {
            "schema": "fibernet.electrical_conditions.v1",
            "task_rule_id": "two_terminal_1V.v1",
            "conditioning": "metadata_only_fixed_rule",
            # Ordered role slots: reference terminal, driven terminal.
            "boundary_voltages": [
                {"node_id": left, "value": 0.0, "unit": "V"},
                {"node_id": right, "value": 1.0, "unit": "V"},
            ],
            "nodal_currents": [],
        },
        # These are opaque labels, not paths that the API opens or verifies.
        "source_refs": {
            "graph": "synthetic:" + graph_id + ":geometry",
            "external": "synthetic:" + graph_id + ":conductivity",
            "conditions": "synthetic:" + graph_id + ":conditions",
        },
    }


def build_records():
    sources = {
        "train-a": make_source("train-a", 1.0e-3),
        "validation-b": make_source("validation-b", 1.2e-3),
        "final-c": make_source("final-c", 0.8e-3),
    }
    feature_spec = {
        "schema": "fibernet.graph_feature_spec.v1",
        "length_unit": "m",
        "coordinate_frame": "specimen_xy",
        "dtype": "float32",
        # Columns are x, y, z, dimension. Length scales are in m.
        "node_offset": [0.0, 0.0, 0.0, 0.0],
        "node_scale": [1.0e-3, 1.0e-3, 1.0e-3, 2.0],
        "features": [
            {
                "name": "radius", "source": {"snapshot": "radius"},
                "unit": "m", "offset": 0.0, "scale": 1.0e-5,
                "required": True, "parity": "even",
            },
            {
                "name": "conductivity",
                "source": {"external": "conductivity_S_per_m"},
                "unit": "S/m", "offset": 0.0, "scale": 1.0e4,
                "required": True, "parity": "even",
            },
        ],
    }
    inputs = bind_graph_inputs(
        [item["batch"] for item in sources.values()],
        source_groups={
            "train-a": "mother-a",
            "validation-b": "mother-b",
            "final-c": "mother-c",
        },
        feature_spec=feature_spec,
        external={key: item["external"] for key, item in sources.items()},
        conditions={key: item["conditions"] for key, item in sources.items()},
        source_refs={key: item["source_refs"] for key, item in sources.items()},
    )
    split = bind_explicit_group_split(
        inputs, train_ids=["train-a"], validation_ids=["validation-b"],
        final_ids=["final-c"], split_seed=42,
    )
    degree = training_degree_histogram(
        inputs, split, training_sources={"train-a": sources["train-a"]},
    )
    admission = validate_pna_degree_admission(
        degree, scalers=["identity", "amplification", "attenuation"],
    )

    # A transport envelope must preserve the exact payload text and digest.
    envelope = {
        "kind": degree.kind,
        "payload_json": degree.payload_json,
        "digest": degree.digest,
    }
    restored_degree = FrozenRecord(**json.loads(json.dumps(envelope)))
    restored_degree.as_dict()  # Verify the payload against this envelope's digest.
    assert validate_pna_degree_admission(
        restored_degree, scalers=["identity", "amplification", "attenuation"],
    ) == admission
    return inputs, split, degree, admission


def main():
    inputs, split, degree, admission = build_records()
    summary = degree.as_dict()["summary"]
    assert summary["histogram"] == [1, 0, 2]
    assert summary["node_population"] == 3
    assert summary["incoming_messages"] == 4
    print(json.dumps({
        "input_binding": inputs.digest,
        "split": split.digest,
        "training_degree": degree.digest,
        "histogram": summary["histogram"],
        "node_population": summary["node_population"],
        "incoming_messages": summary["incoming_messages"],
        "pna_admission": admission.as_dict()["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
