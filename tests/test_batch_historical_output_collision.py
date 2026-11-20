"""Resume reserves outputs of current sources without guessing JSON suffixes."""
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

import fibernet as fn
import fibernet.easy as easy


def save_graph(path):
    g=fn.StructureGraph(dimension=2)
    g.metadata['case']=path.name
    a=g.add_node([0.,0.]);b=g.add_node([1.,1.]);g.add_edge(a,b)
    g.save_json(path)
    return path


def snapshot(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def engine(monkeypatch):
    calls=[];failures=set()
    class Result:
        max_force=2.;max_stretch=1.5;mean_stretch=1.2;std_stretch=.1;time_seconds=.01
        def save(self,path,detailed):Path(path).write_text(json.dumps({'mode':'stretch','max_force':2.}))
    class Engine:
        def __init__(self):calls.append('init')
        def stretch_test(self,graph,**kwargs):
            name=graph.metadata['case'];calls.append(name)
            if name in failures:raise RuntimeError('controlled solver failure')
            return Result()
    monkeypatch.setattr(easy,'_taichi_engine_class',lambda:Engine)
    return calls,failures


def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def record_current_input(csv_path,source,success=True):
    existing=rows(csv_path);row={k:'' for k in existing[0]};row.update(filename=source.name,success=str(success))
    params=easy._batch_parameters('stretch','spring',1.5,1e5,.3,1000,200,'reject_internal_points')
    envelope=easy._batch_binding(hashlib.sha256(source.read_bytes()).hexdigest(),params,easy._batch_implementation_binding())
    row['run_binding']=easy._batch_canonical_json(envelope)
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(existing[0]));w.writeheader();w.writerows([*existing,row])


@pytest.mark.parametrize('failed',[False,True])
@pytest.mark.parametrize('pending',[False,True])
def test_resumed_current_outputs_checked_before_no_pending_return(tmp_path,engine,failed,pending):
    calls,failures=engine;save_graph(tmp_path/'a.json')
    if failed:failures.add('a.json')
    easy.batch_simulate_from_json(str(tmp_path),str(tmp_path))
    # A valid independent graph must still expose the path ambiguity; do not
    # detect historical outputs by parsing them or guessing their suffix.
    source=save_graph(tmp_path/'a_result.json')
    record_current_input(tmp_path/'simulation_results.csv',source)
    if pending:save_graph(tmp_path/'z.json')
    before=snapshot(tmp_path);calls.clear()
    with pytest.raises(ValueError,match='aliases input.*keep input and output directories separate'):
        easy.batch_simulate_from_json(str(tmp_path),str(tmp_path))
    assert calls==[] and snapshot(tmp_path)==before


def test_generated_result_is_not_recorded_as_a_new_failed_input(tmp_path,engine):
    calls,_=engine;save_graph(tmp_path/'a.json');easy.batch_simulate_from_json(str(tmp_path),str(tmp_path));before=snapshot(tmp_path);calls.clear()
    with pytest.raises(ValueError,match='aliases input'):
        easy.batch_simulate_from_json(str(tmp_path),str(tmp_path))
    assert calls==[] and snapshot(tmp_path)==before
    assert [r['filename'] for r in rows(tmp_path/'simulation_results.csv')]==['a.json']


@pytest.mark.parametrize('failed',[False,True])
def test_deleted_historical_input_does_not_reserve_a_current_name(tmp_path,engine,failed):
    calls,failures=engine;source=save_graph(tmp_path/'a.json')
    if failed:failures.add('a.json')
    easy.batch_simulate_from_json(str(tmp_path),str(tmp_path));source.unlink()
    save_graph(tmp_path/'a_result.json');calls.clear()
    easy.batch_simulate_from_json(str(tmp_path),str(tmp_path))
    assert calls==['init','a_result.json']
    assert [r['filename'] for r in rows(tmp_path/'simulation_results.csv')]==['a.json','a_result.json']
    assert (tmp_path/'a_result_result.json').exists()


def test_a_failed_current_case_without_path_overlap_remains_skipped(tmp_path,engine):
    calls,failures=engine;save_graph(tmp_path/'a.json');failures.add('a.json');easy.batch_simulate_from_json(str(tmp_path),str(tmp_path));before=snapshot(tmp_path);calls.clear();failures.clear()
    easy.batch_simulate_from_json(str(tmp_path),str(tmp_path))
    assert calls==[] and snapshot(tmp_path)==before


@pytest.mark.parametrize('alias',['symlink','hardlink'])
@pytest.mark.parametrize('pending',[False,True])
def test_historical_result_alias_to_current_input_is_rejected(tmp_path,engine,alias,pending):
    calls,_=engine;inputs=tmp_path/'in';outputs=tmp_path/'out';inputs.mkdir();save_graph(inputs/'a.json');easy.batch_simulate_from_json(str(inputs),str(outputs));source=save_graph(inputs/'b.json');dest=outputs/'a_result.json';dest.unlink()
    try:
        if alias=='symlink':dest.symlink_to(source)
        else:os.link(source,dest)
    except (OSError,NotImplementedError) as e:pytest.skip(str(e))
    if not pending:record_current_input(outputs/'simulation_results.csv',source)
    before=snapshot(tmp_path);calls.clear()
    with pytest.raises(ValueError,match='aliases input'):
        easy.batch_simulate_from_json(str(inputs),str(outputs))
    assert calls==[] and snapshot(tmp_path)==before


def test_separate_paths_accept_real_result_named_structure_and_skip_successes(tmp_path,engine):
    calls,_=engine;inputs=tmp_path/'in';outputs=tmp_path/'out';inputs.mkdir();save_graph(inputs/'a.json');save_graph(inputs/'a_result.json')
    easy.batch_simulate_from_json(str(inputs),str(outputs));assert calls==['init','a.json','a_result.json'];before=snapshot(tmp_path);calls.clear();easy.batch_simulate_from_json(str(inputs),str(outputs));assert calls==[] and snapshot(tmp_path)==before
    assert all(r['success']=='True' for r in rows(outputs/'simulation_results.csv'))


def test_lone_result_named_graph_is_not_filtered(tmp_path,engine):
    calls,_=engine;save_graph(tmp_path/'a_result.json');easy.batch_simulate_from_json(str(tmp_path),str(tmp_path))
    assert calls==['init','a_result.json'];assert rows(tmp_path/'simulation_results.csv')[0]['success']=='True'


def test_input_binding_mismatch_is_still_checked_first(tmp_path,engine):
    calls,_=engine;source=save_graph(tmp_path/'a.json');easy.batch_simulate_from_json(str(tmp_path),str(tmp_path));source.write_bytes(source.read_bytes()+b'\n');before=snapshot(tmp_path);calls.clear()
    with pytest.raises(ValueError,match='input bytes mismatch'):
        easy.batch_simulate_from_json(str(tmp_path),str(tmp_path))
    assert calls==[] and snapshot(tmp_path)==before
