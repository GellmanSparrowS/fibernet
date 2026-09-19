"""Standalone resumable RL worker. Run: python scripts/rl_worker.py request.json."""
import json
import sys
from pathlib import Path
from dataclasses import asdict, replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fslab.structure import StructureFactory, CUSTOM_CELLS
from fslab.search_algorithms import run_rl, atomic_json
from fslab.inverse import decode_line_params, objective_of, target_curve, TARGETS, fast_cfg
from fslab.engine2 import Engine2
from fslab.simcache import _save_cache


def main(request_path):
    path = Path(request_path).resolve()
    request = json.loads(path.read_text(encoding='utf-8'))
    folder = path.parent
    CUSTOM_CELLS.update(request.get('custom_cells', {}))
    factory = StructureFactory(**request['factory'])
    pts = max(1, factory.n_pts_per_side)
    amplitude = float(request.get('amplitude', .6))
    if not .05 <= amplitude <= 1.:
        raise ValueError('deformation amplitude must be 0.05..1.0')
    initial = np.r_[np.asarray(factory.spectrum() or [[0., 0.]] * pts).ravel() / amplitude,
                    factory.perturbation / .5]
    target = target_curve(request['target']) if request['target'] in TARGETS else None
    model = None
    if request.get('model_hash'):
        import pickle, hashlib
        data = (folder / 'model.pkl').read_bytes()
        if hashlib.sha256(data).hexdigest() != request['model_hash']:
            raise ValueError('model snapshot checksum mismatch')
        model = pickle.loads(data)  # Only snapshots produced by this application's parent process.
    records = []
    best = {'best_dist': None, 'best_spec': None, 'best_label': factory.unit}
    if (folder / 'result.json').exists():
        previous = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        records, best = previous['records'], {k: previous[k] for k in best}

    def objective(x):
        ld, pert = decode_line_params(x, pts, amplitude)
        current = replace(factory, line_displacements=ld, perturbation=pert)
        if model is None:
            run = Engine2(current.build(), fast_cfg(2.).engine_cfg()).run()
            value, _ = objective_of(run, request['target'], target)
        else:
            from fslab.model_inverse import predict_structure
            value, run = predict_structure(current.build(), model, request['target'])
        if best['best_dist'] is None or value < best['best_dist']:
            best.update(best_dist=value, best_spec=dict(unit=factory.unit, pert=pert, line_displacements=ld))
            if model is None:
                _save_cache(str(folder / 'best_run.npz'), run)
            else:
                from fslab.model_inverse import save_prediction
                save_prediction(folder / 'best_prediction.npz', run)
        record = dict(eval_id=(records[-1]['eval_id'] if records else 0)+1, stage=request['algorithm'], label=factory.unit,
                      params=np.asarray(x).tolist(), dist=value, best_dist=best['best_dist'])
        records.append(record)
        del records[:-2000]
        atomic_json(folder / 'result.json', dict(best, records=records, evaluations=records[-1]['eval_id'] if records else 0, stopped=False))
        snapshot = folder / ('candidate_%06d.npz' % record['eval_id'])
        if model is None:
            _save_cache(str(snapshot), run)
        else:
            from fslab.model_inverse import save_prediction
            save_prediction(snapshot, run)
        print(json.dumps({'record': record}), flush=True)
        return value

    run_rl(request['algorithm'], objective, initial, request['budget'], request['seed'],
           request.get('parameters'), folder, lambda: (folder / 'STOP').exists())
    atomic_json(folder / 'result.json', dict(best, records=records, evaluations=records[-1]['eval_id'] if records else 0, stopped=(folder / 'STOP').exists()))


if __name__ == '__main__':
    main(sys.argv[1])
