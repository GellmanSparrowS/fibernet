"""Retained input compatibility with synthetic output, not historical reruns."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path

import pytest

from fibernet.integrations.lammps_backend import PreparedRun, validate_prepared, LAMMPSContractError
from fibernet.integrations import prepare_cg
from examples.cg_bent_branched_network import build_case
from fibernet.integrations import lammps_saved as reader
from test_lammps_saved_read import (
    bundle, canonical, forbid_execution, read, reseal, tmp_path,
)


SOURCE = Path(__file__).resolve().parents[1]


def retained(name):
    frozen = json.loads((SOURCE/'tests/fixtures/lammps_cg_fiber_only_prepared.json').read_text())
    acceptance = json.loads((SOURCE/'tests/fixtures/lammps_cg_fiber_only_acceptance.json').read_text())
    case, fields = acceptance['cases'][name], frozen['cases'][name]
    p = PreparedRun(frozen['graph_json'], fields['contract_json'], frozen['data_text'],
                    fields['input_text'], tuple(acceptance['node_ids']),
                    tuple(acceptance['edge_ids']), case['timestep_s'], case['steps'], 1)
    assert p.identity == case['prepared_sha256']
    return p


@pytest.mark.parametrize('name', ['static', 'coarse', 'fine'])
def test_retained_input_bytes_follow_exact_current_compiler_compatibility(tmp_path, name):
    p = retained(name)
    root = tmp_path/'saved'
    bundle(root, p=p)  # This helper supplies synthetic output, never native bytes.
    try:
        validate_prepared(p)
    except LAMMPSContractError:
        # Python 3.9 norm arithmetic can produce a documented one-ULP
        # derived value difference. The reader must refuse, never normalize.
        with pytest.raises(reader.LAMMPSCompilerCompatibilityError):
            read(root)
    else:
        assert read(root).prepared == p


@pytest.mark.parametrize('direction', [-math.inf, math.inf])
def test_one_ulp_derived_scalar_drift_rejects_without_normalizing(tmp_path, direction):
    # Model the documented cross-Python derived-norm compatibility issue. This
    # is a deliberate one-ULP perturbation, not a claim to run Python 3.9 here.
    graph, request = build_case(steps=0)
    p = prepare_cg(graph, request)
    c = json.loads(p.contract_json)
    original = c['initial_bending_energy_J']
    c['initial_bending_energy_J'] = math.nextafter(original, direction)
    altered = replace(p, contract_json=canonical(c))
    altered = replace(altered, input_text=p.input_text.replace(
        p.completion_marker, altered.completion_marker))
    root = tmp_path/'saved'
    bundle(root, p=p)
    reseal(root, altered)
    (root/'log.lammps').write_text((root/'log.lammps').read_text().replace(
        p.completion_marker, altered.completion_marker))
    before = {n: (root/n).read_bytes() for n in reader._FILES}
    with pytest.raises(reader.LAMMPSCompilerCompatibilityError, match='no migration attempted'):
        read(root)
    assert {n: (root/n).read_bytes() for n in reader._FILES} == before


pytestmark = pytest.mark.skipif(os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW'),
                                reason='saved file reader requires POSIX no-follow descriptors')
