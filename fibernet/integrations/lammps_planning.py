"""Pure planning for existing small pair-free CG inputs; never runtime admission."""
import hashlib
import json

from . import _harmonic_profile as bounds
from ._lammps_forecast import payload_counts
from .lammps_backend import (LAMMPSContractError, PreparedRun, ResourceLimits,
                             RuntimeSpec, _json, validate_prepared)


SCHEMA = 'fibernet.lammps.pair-free-cg-forecast/1'
_PREPARED_BYTES = 1024*1024
# Structural counting limits are independent of a caller budget that may not fit.
# These are planner-input bounds, not changes to execution or ResourceLimits.
_CONFIGURATION_BYTES = 1024*1024
_PLANNING_TEXT_BYTES = 4*1024*1024


def _runtime_record(runtime, byte_limit):
    """Inspect configuration scalars only, without touching provider paths."""
    if type(runtime) is not RuntimeSpec:
        raise LAMMPSContractError('planning requires an inert RuntimeSpec record')
    record = {name: getattr(runtime, name) for name in RuntimeSpec.__dataclass_fields__}
    for name in ('mode', 'python_executable', 'package_dir', 'expected_banner'):
        if type(record[name]) is not str:
            raise LAMMPSContractError('runtime configuration requires text fields')
    if record['mode'] not in ('cli', 'cli_serial_static') or type(record['lammps_executable']) is not str:
        raise LAMMPSContractError('planning supports explicit CLI configuration only')
    if type(record['expected_version']) is not int or not 20000101 <= record['expected_version'] <= 21000101:
        raise LAMMPSContractError('runtime configuration version must be an integer date')
    for name in ('file_sha256', 'environment'):
        pairs = record[name]
        if type(pairs) not in (tuple, list) or len(pairs) > byte_limit:
            raise LAMMPSContractError('runtime configuration requires bounded pairs')
        for pair in pairs:
            if type(pair) not in (tuple, list) or len(pair) != 2 or any(type(v) is not str for v in pair):
                raise LAMMPSContractError('runtime configuration pairs must contain text')
    # Reject oversized structure before allocating duplicate-key sets.
    bounds.json_bytes(record, byte_limit)
    for name in ('file_sha256', 'environment'):
        pairs = record[name]
        if len({pair[0] for pair in pairs}) != len(pairs):
            raise LAMMPSContractError('duplicate runtime configuration key')
    return record


def forecast_cg_fiber_only(prepared, runtime, limits=ResourceLimits(), *,
                           export_budget_bytes, record_state=True):
    """Estimate existing pair-free CG storage without checking a provider.

    ``runtime`` supplies inert manifest/metadata configuration only. No path is
    opened, resolved or hashed, and no task is constructed. ``export_budget_bytes``
    is a caller planning budget, not a newly enforced export limit. CPU/time and
    actual RSS are unknown. A fitting forecast is neither execution permission nor
    runtime, physics or whole-workflow qualification. All existing caps/bytes stay
    unchanged. Raw parsed history remains retained; record_state=False avoids full
    converted-state materialization. Memory estimates remain conservative.
    """
    if type(prepared) is not PreparedRun or type(limits) is not ResourceLimits:
        raise LAMMPSContractError('planning requires PreparedRun and ResourceLimits')
    if type(record_state) is not bool:
        raise LAMMPSContractError('record_state must be boolean')
    if type(export_budget_bytes) is not int or not 1 <= export_budget_bytes <= 2**63-1:
        raise LAMMPSContractError('explicit positive export byte budget required')
    limits.checked()
    prepared_bytes = bounds.string_bytes((prepared.graph_json, prepared.contract_json,
                                         prepared.data_text, prepared.input_text), _PREPARED_BYTES)
    prepared = validate_prepared(prepared)
    c = json.loads(prepared.contract_json)
    if (c.get('model') != 'conservative-ball-cosine-strand-cg/1' or
            c.get('projectile_node_id') is not None or c.get('projectile_pairs') != [] or
            c.get('pair_model') != 'none' or prepared.profile is not None or
            'profile' in c or 'pair_encoding' in c or c.get('sample_every') != 1):
        raise LAMMPSContractError('forecast requires the existing unprofiled pair-free CG contract')
    record = _runtime_record(runtime, _CONFIGURATION_BYTES)
    limit_record = {name: getattr(limits, name) for name in ResourceLimits.__dataclass_fields__}
    manifest = dict(prepared.manifest(), runtime=record, limits=limit_record)
    bounds.json_bytes(manifest, _PLANNING_TEXT_BYTES)
    setup_bytes = bounds.string_bytes((prepared.data_text, prepared.input_text,
                                      prepared.graph_json, _json(manifest)), _PLANNING_TEXT_BYTES)
    metadata = bounds.export_metadata(prepared, runtime)
    metadata_base_bytes = bounds.json_bytes(metadata, _PLANNING_TEXT_BYTES, indent=2)
    from ._lammps_cg import FORCE_ROLES
    metadata.update(model_contract_sha256=hashlib.sha256(prepared.contract_json.encode()).hexdigest(),
        force_roles=dict(FORCE_ROLES), final_max_net_nodal_force_N=0.,
        edge_force_origin='harmonic bond contribution only: k*(length-equilibrium_length); tension-positive; not total nodal force',
        initial_bending_energy_J=c['initial_bending_energy_J'])
    # The final-force scalar is unknown. Add a full finite binary64 token reserve
    # beyond the placeholder, instead of claiming an exact future metadata size.
    metadata_forecast = bounds.json_bytes(metadata, _PLANNING_TEXT_BYTES, indent=2)+24
    n, e, frames = len(prepared.node_ids), len(prepared.edge_ids), prepared.steps+1
    table = c['type_pair_cutoff_payload_lower_bound_bytes']
    sizes = payload_counts(n, e, frames, frames, frames, row_bytes=239,
        prepared_bytes=prepared_bytes, setup_bytes=setup_bytes,
        metadata_reserve_bytes=metadata_forecast,
        raw_reserve_bytes=limits.output_bytes, export_reserve_bytes=export_budget_bytes,
        type_table_bytes=table, record_reference=record_state)
    comparisons = dict(
        native_aggregate=sizes['native_aggregate_forecast_bytes'] <= limits.output_bytes,
        trajectory_file=sizes['trajectory_bytes'] <= limits.file_size_bytes,
        thermo_file=sizes['thermo_bytes'] <= limits.file_size_bytes,
        detailed_export=sizes['detailed_export_forecast_bytes'] <= export_budget_bytes,
        modeled_memory_envelope=sizes['planned_memory_bytes'] <= limits.memory_bytes)
    return dict(schema=SCHEMA, source_identity=dict(prepared_sha256=prepared.identity,
            graph_sha256=hashlib.sha256(prepared.graph_json.encode()).hexdigest()),
        scope='forecast only; no provider, execution, performance or physics qualification',
        runtime_readiness='not_checked', execution_admission=False,
        observation_scope='every integration step retained; no continuous-time guarantee',
        model=c['model'], nodes=n, edges=e, angles=len(c['angles']), atom_types=n,
        pair_interactions=0, pair_coefficient_commands=n*(n+1)//2,
        pair_exclusion_commands=n*(n+1)//2,
        current_caps=dict(max_nodes=64, max_edges=128, max_angles=128, max_steps=1000),
        structural_planning_limits=dict(runtime_configuration_bytes=_CONFIGURATION_BYTES,
                                       counted_text_bytes=_PLANNING_TEXT_BYTES),
        integration_steps=prepared.steps, native_frames=frames, output_frames=frames,
        thermo_frames=frames, node_frames=n*frames, record_state=record_state,
        transient_state_scope='raw parsed history remains retained; record_state=False avoids full converted-state materialization; memory estimates remain conservative',
        prepared_bytes=prepared_bytes, metadata_base_bytes=metadata_base_bytes,
        metadata_forecast_bytes=metadata_forecast,
        native_type_pair_cutoff_payload_lower_bound_bytes=table,
        python_parsing_export_memory_estimate_bytes=sizes['planned_memory_bytes']-table,
        resource_semantics='memory_bytes is a native child address-space ceiling; modeled parent/copy memory is not enforced or measured RSS',
        actual_cpu_seconds=None, actual_peak_rss_bytes=None, predicted_cpu_seconds=None,
        limits=limit_record, export_budget_bytes=export_budget_bytes,
        budget_comparison=comparisons, fits_declared_budgets=all(comparisons.values()),
        **sizes)
