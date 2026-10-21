"""Check public archived thermal data/media without a physical solve."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parent
media=root;m=json.loads((media/'transient_thermal_network.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/m['renderer'])==m['renderer_sha256']
for rel,info in m['sources'].items():assert sha(root/rel)==info['sha256'],rel
assert sha(media/'transient_thermal_network.gif')==m['gif_sha256']
assert sha(media/'transient_thermal_network_final.svg')==m['svg_sha256']
assert 'np.int' not in (media/'transient_thermal_network_final.svg').read_text()
g=Image.open(media/'transient_thermal_network.gif');assert g.n_frames==81 and g.size==(960,1358)
for i in range(g.n_frames):g.seek(i);g.load();assert g.info['duration']==100
assert (media/'transient_thermal_network.gif').stat().st_size<3*1024**2
assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<20*1024**2
s=root/'data';a=np.load(s/'thermal.npz',allow_pickle=False);T=a['temperature_K'];times=a['times_s'];p=a['positions'];ids=a['node_ids'];edges=a['endpoints_by_id']
assert len(set(ids))==4340 and len(set(a['edge_ids']))==5073
assert set(edges.ravel())<=set(ids) and p.shape==(4340,3)
assert T.shape==(81,4340) and np.isfinite(T).all() and np.array_equal(times,np.arange(81)*.25)
assert not a['interval_diagnostics_available'][0] and a['interval_diagnostics_available'][1:].all()
assert np.isnan(a['free_node_residual_J'][0]).all() and np.isnan(a['component_energy_residual_J'][0]).all()
assert (a['node_stored_heat_change_J'][0]==0).all()
summary=json.loads((s/'summary.json').read_text());ledger=summary['independent_energy']
stored=(T-T[0])@a['capacity_J_per_K'];inputE=times*a['source_W'].sum();reservoir=np.r_[0.,np.cumsum([x['reservoir_J'] for x in ledger])]
assert np.max(abs(inputE+reservoir-stored))<1e-14
assert np.max(abs(stored-np.r_[0.,np.cumsum([x['stored_J'] for x in ledger])]))<1e-14
analysis=json.loads((s/'campaign_analysis.json').read_text());assert analysis['permutation']['initial_relative_heat_rate_gate_passed'] is False
assert m['mechanical_deformation'] is False and m['time_interpolation'] is False
print('PASS: hashes, all 81 decoded frames, stable IDs, archived NaN semantics, cumulative energy, preserved failed gate, size bound.')
