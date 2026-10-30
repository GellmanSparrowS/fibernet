#!/usr/bin/env python3
"""
FiberNet Tutorial Pipeline — Standalone Runner

Runs the complete pipeline from the tutorial notebook outside Jupyter.
Supports checkpoint/resume, memory monitoring, and partial execution.

Usage:
    python run_pipeline.py                        # Run full pipeline
    python run_pipeline.py --skip-rl              # Skip RL section
    python run_pipeline.py --num-structures 100   # Quick test with 100 structures
    python run_pipeline.py --from-stage simulation  # Resume from simulation stage
"""

import os
import sys
import json
import time
import gc
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script mode
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Add fibernet to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fibernet as fn
from fibernet import pattern_2d, TaichiEngine, list_units, StructureGraph
from fibernet.viz.render import render_graph, _get_theme
from fibernet.sim.accelerated import _graph_to_arrays, _get_boundary_indices
from fibernet.analysis.graph_features import GraphFeatureExtractor


def get_memory_mb():
    """Get current process memory in MB."""
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except:
        pass
    return 0


def check_memory(limit_mb=4000):
    """Warn if memory exceeds limit."""
    mem = get_memory_mb()
    if mem > limit_mb:
        print(f'⚠ Memory warning: {mem:.0f} MB (limit: {limit_mb} MB)')
        gc.collect()
    return mem


class PipelineRunner:
    """Manages the complete FiberNet tutorial pipeline."""
    
    def __init__(self, data_dir, viz_dir, num_structures=2000, skip_rl=False):
        self.DATA_OUT = Path(data_dir)
        self.VIZ_OUT = Path(viz_dir)
        self.DATA_OUT.mkdir(parents=True, exist_ok=True)
        self.VIZ_OUT.mkdir(parents=True, exist_ok=True)
        
        self.NUM_STRUCTURES = num_structures
        self.SKIP_RL = skip_rl
        
        # Parameters
        self.BOX = (1.0, 1.0)
        self.GRID = (4, 4)
        self.UNIT = 'honeycomb'
        self.N_PTS_PER_SIDE = 3
        self.PERTURBATION = 0.40
        self.STIFFNESS = 1e5
        self.DAMPING = 0.3
        self.NUM_STEPS = 8000
        self.TARGET_STRETCH = 1.5
        
        # State
        self.all_structures = []
        self.all_metadata = []
        self.stage_checkpoints = {}
        
        print(f'Pipeline Runner initialized:')
        print(f'  Structures: {self.NUM_STRUCTURES}')
        print(f'  Data: {self.DATA_OUT}')
        print(f'  Viz:  {self.VIZ_OUT}')
        print(f'  Skip RL: {self.SKIP_RL}')
        print(f'  Memory: {get_memory_mb():.0f} MB')
    
    def _checkpoint(self):
        path = self.DATA_OUT / 'pipeline_checkpoint.json'
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError('expected an object')
            return data
        except (OSError, ValueError) as exc:
            raise ValueError(f'Invalid pipeline checkpoint: {path}') from exc

    @staticmethod
    def _write_json(path, data):
        # Publish only a fully serialized file. Keep existing evidence on failure.
        payload = json.dumps(data, indent=1, allow_nan=False)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(payload)
        temporary.replace(path)

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _save_checkpoint(self, stage, data):
        ckpt = self._checkpoint()
        ckpt[stage] = data
        self._write_json(self.DATA_OUT / 'pipeline_checkpoint.json', ckpt)
        self.stage_checkpoints[stage] = data

    def _check_stage_done(self, stage):
        return stage in self._checkpoint()

    def _generation_config(self):
        return dict(count=self.NUM_STRUCTURES, unit=self.UNIT, box=list(self.BOX),
                    grid=list(self.GRID), n_pts_per_side=self.N_PTS_PER_SIDE,
                    perturbation=self.PERTURBATION)

    def _simulation_config(self):
        return dict(stiffness=self.STIFFNESS, damping=self.DAMPING,
                    num_steps=self.NUM_STEPS, target_stretch=self.TARGET_STRETCH,
                    dt=1e-4, save_interval=1000, dashpot=10.0, drag=1.0,
                    boundary_pct=0.10)

    def _load_generation(self):
        record = self._checkpoint().get('generation')
        if (not isinstance(record, dict) or record.get('version') != 1
                or record.get('config') != self._generation_config()):
            raise ValueError('Missing, legacy, or incompatible generation checkpoint; '
                             'preserve this directory and use a new --data-dir for a fresh run')
        artifacts = record.get('artifacts')
        if not isinstance(artifacts, list) or len(artifacts) != self.NUM_STRUCTURES:
            raise ValueError('Generation artifact count mismatch')
        graphs, metadata = [], []
        for i, item in enumerate(artifacts):
            name = f'honeycomb_{i:03d}'
            path = self.DATA_OUT / 'structures' / f'{name}.json'
            if (not isinstance(item, dict) or item.get('name') != name
                    or not path.is_file() or self._sha256(path) != item.get('sha256')):
                raise ValueError(f'Missing or changed saved graph: {path}')
            try:
                graph = StructureGraph.load_json(path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'Invalid saved StructureGraph: {path}') from exc
            meta = item.get('metadata')
            if (not isinstance(meta, dict) or meta.get('name') != name
                    or meta.get('nodes') != graph.num_nodes
                    or meta.get('edges') != graph.num_edges):
                raise ValueError(f'Incompatible graph metadata: {path}')
            graphs.append(graph)
            metadata.append(dict(meta))
        # Replace state only after the whole set has been validated.
        self.all_structures, self.all_metadata = graphs, metadata
        self._graph_hashes = [item['sha256'] for item in artifacts]

    def _load_simulations(self, complete=False):
        path = self.DATA_OUT / 'simulations' / 'checkpoint.json'
        if not path.is_file():
            if complete or any(path.parent.glob('*_sim.json')):
                raise ValueError(f'Missing simulation checkpoint: {path}')
            return []
        try:
            records = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f'Invalid simulation checkpoint: {path}') from exc
        if (not isinstance(records, list) or len(records) > self.NUM_STRUCTURES
                or (complete and len(records) != self.NUM_STRUCTURES)):
            raise ValueError('Simulation checkpoint count mismatch')
        expected_names = {f'{self.all_metadata[i]["name"]}_sim.json'
                          for i in range(len(records))}
        if any(p.name not in expected_names for p in path.parent.glob('*_sim.json')):
            raise ValueError('Uncommitted simulation outputs exist; preserve them before restarting')
        if complete:
            marker = self._checkpoint().get('simulation')
            if not isinstance(marker, dict) or marker.get('count') != self.NUM_STRUCTURES:
                raise ValueError('Simulation completion count mismatch')
        for i, result in enumerate(records):
            if (not isinstance(result, dict) or result.get('index') != i
                    or result.get('name') != self.all_metadata[i]['name']
                    or result.get('graph_sha256') != self._graph_hashes[i]
                    or result.get('pipeline_config') != self._simulation_config()):
                raise ValueError(f'Legacy or incompatible simulation record: {i}')
            for key in ('max_force', 'max_stretch', 'energy'):
                value = result.get(key)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value)):
                    raise ValueError(f'Invalid simulation metric {key} at index {i}')
        for meta, result in zip(self.all_metadata, records):
            meta.update({key: result[key] for key in ('max_force', 'max_stretch', 'energy')})
        return records

    def _require_features(self):
        record = self._checkpoint().get('features')
        path = self.DATA_OUT / 'structures_features.csv'
        if (not isinstance(record, dict) or record.get('count') != self.NUM_STRUCTURES
                or not path.is_file() or record.get('sha256') != self._sha256(path)
                or record.get('simulation_sha256') != self._sha256(
                    self.DATA_OUT / 'simulations' / 'checkpoint.json')):
            raise ValueError('Missing, legacy, or changed features artifact')

    def stage_generation(self):
        """Generate once; all resume paths restore the saved StructureGraphs."""
        if self._check_stage_done('generation'):
            self._load_generation()
            print(f'✓ Loaded {len(self.all_structures)} saved structures')
            return
        struct_dir = self.DATA_OUT / 'structures'
        sim_dir = self.DATA_OUT / 'simulations'
        if (self._checkpoint() or (struct_dir.exists() and any(struct_dir.iterdir()))
                or (sim_dir.exists() and any(sim_dir.iterdir()))
                or (self.DATA_OUT / 'structures_features.csv').exists()):
            raise ValueError('Uncommitted or legacy artifacts exist; preserve this directory '
                             'and use a new --data-dir for a fresh run')
        if self.NUM_STRUCTURES <= 0:
            raise ValueError('num_structures must be positive')
        struct_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        artifacts = []
        for i in range(self.NUM_STRUCTURES):
            g = pattern_2d(unit=self.UNIT, box=self.BOX, grid=self.GRID, seed=100+i,
                           n_pts_per_side=self.N_PTS_PER_SIDE, perturbation=self.PERTURBATION)
            name = f'honeycomb_{i:03d}'
            path = struct_dir / f'{name}.json'
            temporary = path.with_suffix('.json.tmp')
            g.save_json(temporary)
            temporary.replace(path)
            meta = dict(name=name, seed=100+i, nodes=g.num_nodes, edges=g.num_edges,
                        perturbation=self.PERTURBATION)
            artifacts.append(dict(name=name, sha256=self._sha256(path), metadata=meta))
        self._save_checkpoint('generation', dict(version=1, config=self._generation_config(),
                                                artifacts=artifacts, time=time.time()-t0))
        self._load_generation()

    def stage_simulation(self):
        """Stage 2: Run simulations."""
        self._load_generation()
        complete = self._check_stage_done('simulation')
        ckpt = self._load_simulations(complete=complete)
        if complete:
            record = self._checkpoint()['simulation']
            if not isinstance(record, dict) or record.get('count') != self.NUM_STRUCTURES:
                raise ValueError('Simulation completion count mismatch')
            print(f'✓ Loaded {len(ckpt)} simulation results')
            return

        print(f'Simulating {self.NUM_STRUCTURES} structures...')
        engine = TaichiEngine() if len(ckpt) < self.NUM_STRUCTURES else None
        sim_dir = self.DATA_OUT / 'simulations'
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        def get_boundary_nodes(pos, pct=0.10):
            try:
                bnd = _get_boundary_indices(pos, pct=pct)
                return bnd['left'], bnd['right']
            except TypeError:
                x_sorted = np.argsort(pos[:, 0])
                n = len(pos)
                n_bnd = max(int(n * pct), 1)
                return list(x_sorted[:n_bnd]), list(x_sorted[-n_bnd:])
        
        def run_stretch(g):
            pos, elems, nids, _ = _graph_to_arrays(g)
            left_nodes, right_nodes = get_boundary_nodes(pos, pct=0.10)
            L_x = pos[:, 0].max() - pos[:, 0].min()
            target_disp = L_x * (self.TARGET_STRETCH - 1)
            disp_schedule = {}
            for ni in right_nodes:
                disp_schedule[ni] = [(0, np.zeros(3)), (self.NUM_STEPS, np.array([target_disp, 0, 0]))]
            return engine.dynamics(g, fixed_nodes=left_nodes,
                displacement_schedule=disp_schedule, stiffness=self.STIFFNESS,
                damping=self.DAMPING, dt=1e-4, num_steps=self.NUM_STEPS,
                save_interval=1000, dashpot=10.0, drag=1.0)
        
        # Checkpoint support
        ckpt_path = sim_dir / 'checkpoint.json'
        start_idx = len(ckpt)
        if start_idx:
            print(f'  Resuming from {start_idx}/{self.NUM_STRUCTURES}')

        t0 = time.time()
        for i in range(start_idx, self.NUM_STRUCTURES):
            g = self.all_structures[i]
            result = run_stretch(g)
            
            sim_dict = result.to_dict()
            sim_dict['graph_sha256'] = self._graph_hashes[i]
            sim_dict['pipeline_config'] = self._simulation_config()
            sim_dict['index'] = i
            sim_dict['name'] = self.all_metadata[i]['name']
            ckpt.append(sim_dict)
            
            self.all_metadata[i]['max_force'] = result.max_force
            self.all_metadata[i]['max_stretch'] = result.max_stretch
            self.all_metadata[i]['energy'] = result.energy
            
            result.save(str(sim_dir / f'{self.all_metadata[i]["name"]}_sim.json'), detailed=False)
            
            if (i + 1) % 50 == 0 or i == self.NUM_STRUCTURES - 1:
                self._write_json(ckpt_path, ckpt)
                elapsed = time.time() - t0
                rate = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
                remaining = (self.NUM_STRUCTURES - i - 1) / rate if rate > 0 else 0
                print(f'  {i+1}/{self.NUM_STRUCTURES} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, {get_memory_mb():.0f}MB)')
                gc.collect()
        
        self._write_json(ckpt_path, ckpt)
        
        elapsed = time.time() - t0
        forces_kN = np.array([m['max_force'] for m in self.all_metadata]) / 1000.0
        print(f'✓ Completed {len(ckpt)} simulations in {elapsed:.0f}s')
        print(f'  Force range: {forces_kN.min():.1f} - {forces_kN.max():.1f} kN')
        print(f'  Mean force: {forces_kN.mean():.1f} kN')
        self._save_checkpoint('simulation', {'count': len(ckpt), 'time': elapsed})
    
    def stage_features(self):
        """Stage 3: Extract features."""
        self._load_generation()
        if not self._check_stage_done('simulation'):
            raise ValueError('Feature extraction requires completed saved simulations')
        self._load_simulations(complete=True)
        if self._check_stage_done('features'):
            self._require_features()
            print('✓ Feature extraction already done (checkpoint)')
            return
        
        if not self.all_metadata or 'max_force' not in self.all_metadata[0]:
            print('✗ No simulation results, run simulation first')
            return
        
        if (self.DATA_OUT / 'structures_features.csv').exists():
            raise ValueError('Uncommitted features CSV exists; preserve it before restarting')
        print(f'Extracting features for {len(self.all_structures)} structures...')
        extractor = GraphFeatureExtractor()
        t0 = time.time()
        for i, (g, meta) in enumerate(zip(self.all_structures, self.all_metadata)):
            meta.update(extractor.extract(g))
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(self.all_structures)} ({time.time()-t0:.1f}s)')
        
        import pandas as pd
        df = pd.DataFrame(self.all_metadata)
        csv_path = self.DATA_OUT / 'structures_features.csv'
        df.to_csv(csv_path, index=False)
        
        elapsed = time.time() - t0
        print(f'✓ Extracted features in {elapsed:.1f}s, saved to {csv_path}')
        self._save_checkpoint('features', {'count': len(self.all_structures), 'time': elapsed,
                                           'sha256': self._sha256(csv_path),
                                           'simulation_sha256': self._sha256(
                                               self.DATA_OUT / 'simulations' / 'checkpoint.json')})
    
    def stage_ml(self):
        """Stage 4: ML binary classification."""
        if self._check_stage_done('ml'):
            print('✓ ML analysis already done (checkpoint)')
            return
        
        print('Running ML binary classification...')
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                     f1_score, roc_auc_score)
        import pandas as pd
        
        # Load features
        csv_path = self.DATA_OUT / 'structures_features.csv'
        if not csv_path.exists():
            print('✗ Features CSV not found, run feature extraction first')
            return
        
        df = pd.read_csv(csv_path)
        skip_cols = ['name', 'seed', 'nodes', 'edges', 'max_force', 'max_stretch', 'energy', 'perturbation']
        feature_cols = [c for c in df.columns if c not in skip_cols and df[c].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].values
        y_force = df['max_force'].values
        threshold = np.median(y_force)
        y_binary = (y_force >= threshold).astype(int)
        
        print(f'  Threshold: {threshold/1000:.1f} kN (median)')
        print(f'  Low: {np.sum(y_binary == 0)}, High: {np.sum(y_binary == 1)}')
        
        rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=8, 
                                     oob_score=True, class_weight='balanced')
        rf.fit(X, y_binary)
        
        y_pred = rf.predict(X)
        y_proba = rf.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y_binary, y_pred)
        f1 = f1_score(y_binary, y_pred)
        auc = roc_auc_score(y_binary, y_proba)
        
        print(f'  Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}')
        print(f'  OOB Score: {rf.oob_score_:.3f}')
        
        import joblib
        model_path = self.VIZ_OUT / 'rf_strength_classifier.joblib'
        joblib.dump(rf, model_path)
        print(f'✓ ML model saved to {model_path}')
        
        self._save_checkpoint('ml', {'accuracy': acc, 'f1': f1, 'auc': auc, 'threshold': threshold})
    
    def run(self, from_stage=None):
        """Run the complete pipeline."""
        stages = [
            ('generation', self.stage_generation),
            ('simulation', self.stage_simulation),
            ('features', self.stage_features),
            ('ml', self.stage_ml),
        ]
        
        if from_stage not in (None, 'generation', 'simulation', 'features', 'ml'):
            raise ValueError(f'Unknown pipeline stage: {from_stage}')
        if from_stage in ('simulation', 'features', 'ml'):
            self._load_generation()
        if from_stage in ('features', 'ml'):
            if not self._check_stage_done('simulation'):
                raise ValueError('Resume requires completed saved simulations')
            self._load_simulations(complete=True)
        if from_stage == 'ml':
            self._require_features()
        start = False if from_stage else True
        for name, func in stages:
            if from_stage and name == from_stage:
                start = True
            if start:
                print(f'\n{"="*60}')
                print(f'Stage: {name}')
                print(f'{"="*60}')
                func()
                check_memory()


def main():
    parser = argparse.ArgumentParser(description='FiberNet Tutorial Pipeline Runner')
    parser.add_argument('--data-dir', default='tutorial_data', help='Data output directory')
    parser.add_argument('--viz-dir', default='tutorial_viz', help='Visualization output directory')
    parser.add_argument('--num-structures', type=int, default=2000, help='Number of structures')
    parser.add_argument('--skip-rl', action='store_true', help='Skip RL section')
    parser.add_argument('--from-stage', choices=['generation', 'simulation', 'features', 'ml'],
                       help='Resume from specific stage')
    args = parser.parse_args()
    
    runner = PipelineRunner(
        data_dir=args.data_dir,
        viz_dir=args.viz_dir,
        num_structures=args.num_structures,
        skip_rl=args.skip_rl,
    )
    runner.run(from_stage=args.from_stage)


if __name__ == '__main__':
    main()
