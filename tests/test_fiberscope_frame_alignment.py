"""Execute exact source with solver/GUI imports excluded; synthetic data only."""
import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(folder):
    base=ROOT
    path=base/'FiberScope/fslab/simcache.py'
    tree=ast.parse(path.read_text())
    # The tested source is unchanged except imports of optional solver packages.
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and
        (n.level or (n.module or '').startswith('fibernet')))]
    mod=types.ModuleType('isolated_'+folder);sys.modules[mod.__name__]=mod
    mod.__file__=str(path)
    mod.Engine2Config=object
    mod.FIBERNET_ROOT=str(base)
    exec(compile(tree,str(path),'exec'),mod.__dict__)
    return mod

sc=load('patched')

def pairset(a):return set(map(tuple,np.asarray(a,dtype=int).reshape(-1,2)))

def fixture(F):
    labels=np.arange(F,dtype=float)
    contacts=[np.empty((0,2),dtype=int) for _ in range(F)]
    for frame,pair in [(1,(0,1)),(2,(1,2)),(F-1,(4,5))]:
        if frame<F:contacts[frame]=np.array([pair])
    return sc.StretchRun(np.broadcast_to(labels[:,None,None],(F,6,2)).copy(),
        np.array([[0,1]]),np.ones(1),labels[:,None].copy(),labels.copy(),labels.copy(),
        np.array([0]),np.array([5]),{k:labels.copy() for k in ('axial','bend','contact','kinetic','work')},
        contacts[-1].copy(),contacts,np.array([len(a) for a in contacts]),{'synthetic':True})

class FrameAlignment(unittest.TestCase):
    def test_native_engine_result_without_history_roundtrips(self):
        from fibernet.sim.reduced_beam import Engine2Result
        original = fixture(10)
        native = Engine2Result(**{field.name: getattr(original, field.name)
                                 for field in dataclasses.fields(Engine2Result)})
        self.assertFalse(hasattr(native, 'contact_history'))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'native.npz')
            sc._save_cache(path, native)
            restored = sc._load_cache(path)
            self.assertIsNotNone(restored)
            self.assertIsNone(restored.contact_history)
            np.testing.assert_array_equal(restored.frames_xy, native.frames_xy)
            np.testing.assert_array_equal(restored.contact_counts, native.contact_counts)
            for a, b in zip(restored.contact_frames, native.contact_frames):
                self.assertEqual(pairset(a), pairset(b))

    def test_threshold_and_all_frame_aligned_fields(self):
        for F in (239,240,241,481):
            with self.subTest(F=F):
                raw=fixture(F);out=sc._downsample(fixture(F))
                idx=np.unique(np.linspace(0,F-1,240).round().astype(int)) if F>240 else np.arange(F)
                self.assertEqual(len(out.contact_frames),len(idx))
                for field in ('frames_xy','edge_strain','strain_levels','force_curve','contact_counts'):
                    np.testing.assert_array_equal(getattr(out,field),getattr(raw,field)[idx])
                for k in raw.energies:np.testing.assert_array_equal(out.energies[k],raw.energies[k][idx])
                for j,i in enumerate(idx):self.assertEqual(pairset(out.contact_frames[j]),pairset(raw.contact_frames[i]))
                np.testing.assert_array_equal(out.contact_pairs_last,raw.contact_pairs_last)
                if F>240:
                    self.assertEqual(len(out.contact_history),len(idx))
                    for j,i in enumerate(idx):
                        expected=set().union(*(pairset(p) for p in raw.contact_frames[:i+1]))
                        self.assertEqual(pairset(out.contact_history[j]),expected)
                else:self.assertIsNone(out.contact_history)

    def test_skipped_event_is_history_not_current_or_future(self):
        out=sc._downsample(fixture(481))
        self.assertEqual(out.frames_xy[1,0,0],2)
        self.assertEqual(pairset(out.contact_frames[1]),{(1,2)})
        self.assertEqual(pairset(out.contact_history[1]),{(0,1),(1,2)})
        self.assertEqual(pairset(out.contact_history[0]),set())
        self.assertNotIn((4,5),pairset(out.contact_history[-2]))
        self.assertEqual(pairset(out.contact_frames[-1]),{(4,5)})

    def test_npz_roundtrip_and_short_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            for F in (10,481):
                with self.subTest(F=F):
                    original=sc._downsample(fixture(F));path=str(Path(directory)/f'{F}.npz')
                    sc._save_cache(path,original);restored=sc._load_cache(path)
                    self.assertIsNotNone(restored)
                    np.testing.assert_array_equal(restored.frames_xy,original.frames_xy)
                    for a,b in zip(restored.contact_frames,original.contact_frames):self.assertEqual(pairset(a),pairset(b))
                    if F>240:
                        for a,b in zip(restored.contact_history,original.contact_history):self.assertEqual(pairset(a),pairset(b))
                    else:self.assertIsNone(restored.contact_history)

    def test_actual_runner_transfers_history_and_cache_identity_changes(self):
        class Graph:
            def node_positions(self):return np.zeros((6,3))
        class Factory:
            def build(self):return Graph()
            def key(self):return 'synthetic'
        class Engine:
            def __init__(self,*a):pass
            def run(self,**kwargs):return fixture(481)
        cfg=sc.RunConfig();cfg.engine_cfg=lambda n:None
        sc.ensure_fibernet=lambda:None
        sc._validate_planar_geometry=lambda *a:None
        sc.Engine2=Engine
        sc._engine_input_digest=lambda graph:'fixed synthetic input'
        with tempfile.TemporaryDirectory() as directory:
            run=sc.run_stretch(Factory(),cfg,cache_dir=directory)
            self.assertEqual(len(run.contact_history),240)
            cached=sc.run_stretch(Factory(),cfg,cache_dir=directory)
            self.assertTrue(cached.cache_hit)
            self.assertEqual(pairset(cached.contact_history[1]),{(0,1),(1,2)})
        new_key=sc.cache_tag(Factory(),cfg,graph=Graph())
        version,digest=sc.ENGINE_VERSION,sc.ENGINE_SRC_HASH
        try:
            sc.ENGINE_VERSION,sc.ENGINE_SRC_HASH='e2v14','bf860374a873'
            old_key=sc.cache_tag(Factory(),cfg,graph=Graph())
        finally:
            sc.ENGINE_VERSION,sc.ENGINE_SRC_HASH=version,digest
        self.assertNotEqual(new_key,old_key)
        self.assertEqual(sc.engine_src_hash(),sc.ENGINE_SRC_HASH)

    def test_actual_canvas_contact_history_block(self):
        path=ROOT/'FiberScope/studio/network_canvas.py'
        tree=ast.parse(path.read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='set_data')
        # Execute only the real contact-history block, without Qt or rendering.
        start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='history')
        block=method.body[start:start+3]
        for F in (10,481):
            run=sc._downsample(fixture(F));owner=types.SimpleNamespace(_contact_cum=None)
            env={'run':run,'self':owner,'np':np}
            exec(compile(ast.Module(body=block,type_ignores=[]),str(path),'exec'),env)
            self.assertEqual(len(owner._contact_cum),run.n_frames)
            if F>240:
                self.assertEqual(owner._contact_cum[1]-pairset(run.contact_frames[1]),{(0,1)})
                self.assertNotIn((4,5),owner._contact_cum[-2])

if __name__=='__main__':unittest.main()
