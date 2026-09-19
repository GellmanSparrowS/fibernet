"""Parameter limits, physical topology coverage, fresh and resumed jobs."""
import os
import sys
import tempfile
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from fslab.learning import Regressor, MODEL_SPECS
from fslab.mlmodel import gen_dataset
from fslab.search_algorithms import run_rl, SEARCH_SPECS, atomic_json
from fslab.rl_process import run_external
from fslab.structure import StructureFactory, all_unit_keys
from fslab.inverse import run_inverse
from fslab.surface_mapping import load_obj


def main():
    records = []
    root = Path('_tmp')
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix='regression23_') as directory:
        folder = Path(directory).resolve()
        X, Y = gen_dataset('square', n=12, path=str(folder/'physical.npz'))
        for key, spec in MODEL_SPECS.items():
            for parameter, descriptor in spec[2].items():
                for value in descriptor[1:3]:
                    model = Regressor(key, {parameter: value})
                    history = model.train(X, Y, epochs=20)
                    assert np.isfinite(model.predict(X)).all() and history['train']
                    records.append(dict(kind='supervised', algorithm=key, parameter=parameter, value=value))
        print('Supervised physical-label parameter endpoints PASS', flush=True)
        for key, spec in SEARCH_SPECS.items():
            if key == 'cem':
                continue
            for parameter, descriptor in spec[2].items():
                for value in descriptor[1:3]:
                    budget = max(40, int(value)+8) if parameter in ('n_steps', 'learning_starts') else 40
                    result = run_rl(key, lambda x: float(np.sum((x-.4)**2)), np.zeros(3),
                                    budget=budget, seed=4, parameters={parameter: value})
                    assert result['evaluations'] == budget and np.isfinite(result['best_value'])
                    records.append(dict(kind='rl_parameter_contract', algorithm=key, parameter=parameter, value=value, evaluations=budget))
            print(key, 'parameter endpoints PASS', flush=True)
        for unit in all_unit_keys():
            factory = StructureFactory(unit=unit, grid_x=1, grid_y=1, n_pts_per_side=2)
            def builder(key, pert, ld):
                return replace(factory, unit=key, perturbation=pert, line_displacements=ld).build()
            result = run_inverse(builder, 'J', budget=3, fixed_unit=unit, pts=2, initial_spec=factory)
            assert len(result['records']) == 3 and result['best_run'] is not None
            assert result['best_spec']['unit'] == unit
            records.append(dict(kind='physical_topology', unit=unit, evaluations=3))
        old = os.environ.get('FIBERSCOPE_TRAINING_DIR')
        os.environ['FIBERSCOPE_TRAINING_DIR'] = str(folder/'training')
        try:
            factory = StructureFactory(grid_x=1, grid_y=1, n_pts_per_side=1)
            updates = []
            def callback(rec, run):
                assert run is not None
                updates.append(rec.eval_id)
            first = run_external(factory, 'J', 'PPO', {}, 3, 23, callback)
            second = run_external(factory, 'J', 'PPO', {}, 3, 23, callback)
            assert first['checkpoint'] != second['checkpoint']
            resumed = run_external(factory, 'J', 'PPO', {}, 5, 23, callback, resume=True)
            assert resumed['checkpoint'] == second['checkpoint']
            assert updates == [1,2,3,1,2,3,4,5], updates
            assert resumed['best_dist'] <= second['best_dist']
            records.append(dict(kind='fresh_and_resume', candidate_updates=8, passed=True))
        finally:
            if old is None:
                os.environ.pop('FIBERSCOPE_TRAINING_DIR', None)
            else:
                os.environ['FIBERSCOPE_TRAINING_DIR'] = old
        invalid = folder/'invalid.obj'
        for face in ('f 0 1 2', 'f 1 1 2', 'f 1 2 4'):
            invalid.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\n'+face, encoding='utf-8')
            try:
                load_obj(invalid)
                raise AssertionError('invalid face accepted')
            except ValueError:
                pass
    atomic_json('docs/validation/parameters23.json', dict(passed=True, cases=records))
    print('2.3 parameter, actual topology, fresh/resume and import guards PASS', flush=True)


if __name__ == '__main__':
    main()
