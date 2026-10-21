"""Explicitly invoked, independent two-node electrothermal comparison.

Run: python -m benchmarks.electrothermal_two_node --output report.json

Importing this module performs no solve and imports no production solver.
The Decimal references use this fixture's scalar equations only. Comparison
criteria are specific to this fixture, not general solver accuracy claims.
"""

import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path


def _decimal_references(step_text):
    """Return independent discrete and continuous temperatures as Decimals."""
    with localcontext() as context:
        context.prec = 80
        dt = Decimal(step_text)
        horizon = Decimal("1")
        if not dt.is_finite() or dt <= 0:
            raise ValueError("The reference requires a positive exact 1 s grid.")
        steps = int(horizon / dt)
        if Decimal(steps) * dt != horizon:
            raise ValueError("The reference requires a positive exact 1 s grid.")
        initial = Decimal("300")
        alpha = Decimal("0.01")
        rate = Decimal("4")  # (4 V)^2 * 1 S * 0.5 / (2 J/K)
        one = Decimal("1")
        two = Decimal("2")
        four = Decimal("4")
        rise = Decimal("0")
        discrete = [initial]
        continuous = [initial]
        times = [Decimal("0")]
        for index in range(1, steps + 1):
            # Positive root of alpha*delta^2 + (1+alpha*rise)*delta
            # - rate*dt = 0, written without subtractive cancellation.
            coefficient = one + alpha * rise
            discriminant = coefficient * coefficient + four * alpha * rate * dt
            rise += two * rate * dt / (coefficient + discriminant.sqrt())
            time = Decimal(index) * dt
            exact_rise = (
                two * rate * time
                / (one + (one + two * alpha * rate * time).sqrt())
            )
            times.append(time)
            discrete.append(initial + rise)
            continuous.append(initial + exact_rise)
        return times, discrete, continuous


def run_benchmark():
    """Execute the three fixed grids and return a strict-JSON-ready report."""
    # Keep all production imports and numerical execution behind this call.
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.sim.electrothermal import (
        TemperatureConductanceLaw,
        solve_electrothermal,
    )

    graph = StructureGraph()
    left = graph.add_node([0.0, 0.0, 0.0])
    right = graph.add_node([1.0, 0.0, 0.0])
    edge = graph.add_edge(left, right)
    law = TemperatureConductanceLaw(
        G0_S={edge: 1.0},
        alpha_per_K={edge: 0.01},
        reference_temperature_K={edge: 300.0},
        endpoint_temperature_weights={edge: {left: 0.5, right: 0.5}},
        temperature_domain_K=(290.0, 350.0),
    )
    criteria = {
        "maximum_discrete_reference_temperature_error_K": 2e-10,
        "finest_grid_final_continuous_reference_temperature_error_K": 0.003,
        "successive_halving_final_error_ratio_lower": 0.45,
        "successive_halving_final_error_ratio_upper": 0.55,
    }
    runs = []
    final_errors = []
    for step_text in ("0.1", "0.05", "0.025"):
        times, discrete, continuous = _decimal_references(step_text)
        result = solve_electrothermal(
            graph,
            law,
            prescribed_voltage_V={left: 0.0, right: 4.0},
            capacity_J_per_K={left: 2.0, right: 2.0},
            conductance_W_per_K={edge: 0.0},
            initial_temperature_K={left: 300.0, right: 300.0},
            times_s=[float(time) for time in times],
            endpoint_heat_weights={edge: {left: 0.5, right: 0.5}},
            length_unit="m",
        )
        if not result.completed or len(result.frames) != len(times):
            raise RuntimeError("The benchmark requires a complete trajectory.")
        with localcontext() as context:
            context.prec = 80
            discrete_errors = [
                abs(Decimal.from_float(temperature) - reference)
                for frame, reference in zip(result.frames, discrete)
                for temperature in frame.temperature_K
            ]
            continuous_errors = [
                abs(Decimal.from_float(temperature) - reference)
                for frame, reference in zip(result.frames, continuous)
                for temperature in frame.temperature_K
            ]
            final_error = max(
                abs(Decimal.from_float(temperature) - continuous[-1])
                for temperature in result.frames[-1].temperature_K
            )
        final_errors.append(final_error)
        maximum_discrete_error = max(discrete_errors)
        runs.append({
            "dt_s": float(step_text),
            "interval_count": len(times) - 1,
            "completed": result.completed,
            "node_ids": list(result.node_ids),
            "times_s": [float(time) for time in times],
            "temperature_K": [
                list(frame.temperature_K) for frame in result.frames
            ],
            # Decimal strings retain reference precision in the JSON report.
            "backward_euler_reference_temperature_K": [
                str(value) for value in discrete
            ],
            "continuous_reference_temperature_K": [
                str(value) for value in continuous
            ],
            "maximum_discrete_reference_temperature_error_K": float(
                maximum_discrete_error
            ),
            "maximum_continuous_reference_temperature_error_K": float(
                max(continuous_errors)
            ),
            "final_continuous_reference_temperature_error_K": float(final_error),
            "discrete_reference_comparison_passed": (
                maximum_discrete_error <= Decimal("2e-10")
            ),
            "accepted_iterations_by_interval": [
                frame.accepted_iterations for frame in result.frames[1:]
            ],
            "input_sha256": result.input_sha256,
        })

    with localcontext() as context:
        context.prec = 80
        ratios = [
            finer / coarser if coarser != 0 else None
            for coarser, finer in zip(final_errors, final_errors[1:])
        ]
    ratio_comparisons = [
        ratio is not None and Decimal("0.45") <= ratio <= Decimal("0.55")
        for ratio in ratios
    ]
    finest_comparison = final_errors[-1] <= Decimal("0.003")
    return {
        "schema": "fibernet-electrothermal-two-node-comparison-1",
        "scope": (
            "Synthetic symmetric two-node fixture only; no general accuracy, "
            "material validity, or network convergence claim."
        ),
        "fixture": {
            "node_count": 2,
            "edge_count": 1,
            "prescribed_voltage_V": [0.0, 4.0],
            "G0_S": 1.0,
            "alpha_per_K": 0.01,
            "reference_temperature_K": 300.0,
            "temperature_domain_K": [290.0, 350.0],
            "capacity_each_J_per_K": 2.0,
            "thermal_edge_conductance_W_per_K": 0.0,
            "initial_temperature_each_K": 300.0,
            "endpoint_temperature_weights": [0.5, 0.5],
            "endpoint_heat_weights": [0.5, 0.5],
            "external_source_each_W": 0.0,
            "fixed_thermal_node_count": 0,
            "horizon_s": 1.0,
            "step_sizes_s": [0.1, 0.05, 0.025],
        },
        "reference": {
            "arithmetic": "decimal.Decimal",
            "precision_digits": 80,
            "discrete": "Independent positive quadratic backward-Euler root",
            "continuous": "Independent integrated scalar ODE closed form",
            "production_helpers_used": False,
            "reference_temperature_arrays_apply_to": "each node",
        },
        "fixture_specific_comparison_criteria": criteria,
        "runs": runs,
        "successive_halving_final_continuous_error_ratios": [
            float(ratio) if ratio is not None else None for ratio in ratios
        ],
        "successive_halving_ratio_comparisons_passed": ratio_comparisons,
        "finest_continuous_reference_comparison_passed": finest_comparison,
        "all_fixture_comparisons_passed": (
            all(run["discrete_reference_comparison_passed"] for run in runs)
            and finest_comparison
            and all(ratio_comparisons)
        ),
    }


def main(argv=None):
    """Write the JSON report; return nonzero if a fixture comparison fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON report to this file instead of standard output.",
    )
    args = parser.parse_args(argv)
    report = run_benchmark()
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")
    return 0 if report["all_fixture_comparisons_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
