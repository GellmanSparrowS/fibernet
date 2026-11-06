"""Storage-only preflight for the existing crosslinked CG example.

This entry point never executes a simulation, even when all budgets fit.
Supply the intended inert RuntimeSpec JSON and explicit existing ResourceLimits
JSON (all five fields), plus the detailed-export planning budget in bytes.
"""
import argparse
import json
from pathlib import Path

from examples.cg_crosslinked_network import build_case
from fibernet.integrations import ResourceLimits, RuntimeSpec, forecast_cg_fiber_only, prepare_cg


def preflight(prepared, runtime, *, limits, export_budget_bytes):
    """Return exact planning comparisons; budget fit never admits execution.

    Invalid inputs raise the existing contract errors. No provider validation,
    output directories, native tasks, or export files are created. Export and
    parent-memory comparisons are estimates, not enforcement or measured usage.
    """
    forecast = forecast_cg_fiber_only(prepared, runtime, limits,
        export_budget_bytes=export_budget_bytes, record_state=True)
    fields = (
        ('native_aggregate', 'native_aggregate_forecast_bytes', limits.output_bytes),
        ('trajectory_file', 'trajectory_bytes', limits.file_size_bytes),
        ('thermo_file', 'thermo_bytes', limits.file_size_bytes),
        ('detailed_export', 'detailed_export_forecast_bytes', export_budget_bytes),
        ('modeled_memory_envelope', 'planned_memory_bytes', limits.memory_bytes),
    )
    comparisons = {name: dict(forecast_bytes=forecast[key], budget_bytes=budget,
        fits=forecast[key] <= budget) for name, key, budget in fields}
    failed = [name for name, comparison in comparisons.items() if not comparison['fits']]
    return dict(schema='fibernet.example.cg-resource-preflight/1',
        status='refused' if failed else 'storage_fit_only',
        execution_admission=False, runtime_readiness='not_checked',
        whole_workflow_resource_acceptance='not_established',
        failed_comparisons=failed, comparisons=comparisons, forecast=forecast)


def _load_record(path):
    """Bound configuration reads and reject ambiguous duplicate JSON keys."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate configuration key: '+key)
            result[key] = value
        return result

    with Path(path).open('rb') as stream:
        data = stream.read(65537)
    if len(data) > 65536:
        raise ValueError('configuration JSON exceeds 64 KiB')
    record = json.loads(data, object_pairs_hook=unique_pairs)
    if type(record) is not dict:
        raise ValueError('configuration JSON must be an object')
    return record


def load_runtime(path):
    """Read inert configuration only; never inspect provider paths."""
    return RuntimeSpec(**_load_record(path))


def load_limits(path):
    """Read explicit existing native limits, without filling missing fields."""
    record = _load_record(path)
    if set(record) != set(ResourceLimits.__dataclass_fields__):
        raise ValueError('limits JSON must explicitly contain all ResourceLimits fields')
    return ResourceLimits(**record).checked()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-json', required=True, type=Path)
    parser.add_argument('--limits-json', required=True, type=Path)
    parser.add_argument('--export-budget-bytes', required=True, type=int)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--timestep-s', type=float, default=1e-7)
    args = parser.parse_args(argv)
    runtime = load_runtime(args.runtime_json)
    limits = load_limits(args.limits_json)
    _, graph, request = build_case(steps=args.steps, timestep_s=args.timestep_s)
    report = preflight(prepare_cg(graph, request), runtime, limits=limits,
                       export_budget_bytes=args.export_budget_bytes)
    print(json.dumps(report, indent=2))
    return 1 if report['failed_comparisons'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
