# Recorded viscoelastic network dynamics

This animation visualizes a saved FiberNet native axial generalized-Maxwell trajectory: 105 nodes, 273 physical edges, 39 fixed supports, 4096 time intervals of 0.0001220703125 seconds over 0.5 seconds. There are 129 saved states; the GIF displays every second state, including both endpoints, at 100 milliseconds per image. No physics was rerun to create this media.

The geometry displacement is magnified **3000 times**. Gray lines show rest geometry. Colors show actual signed axial stress, with fixed global limits of approximately ±18.94 Pa. Squares identify fixed supports. Energies use actual unamplified states and are normalized by initial kinetic energy. Numerical backward-Euler loss is shown separately from material dissipation. Small displacement is not a visual bug: the maximum recorded nodal displacement is approximately 0.2985 micrometres over a 13-millimetre-wide network.

All parameters are synthetic. The model has equilibrium modulus 100 kPa, Maxwell branches 200 kPa/0.02 s and 100 kPa/0.1 s, 0.001 kg explicit nodal masses and 1e-8 m² reference areas. Initial velocities excite the free motion; constant external loads and numerical drag are zero. This is small-strain axial material dynamics, not bending, contact, adhesion, fracture or experimentally calibrated spider silk.

The compact JSON retains every saved position, velocity, stress and displayed material ledger, plus stable IDs, edge connectivity, masses and initial geometry. It is a render dataset, not a restart checkpoint. The NPZ duplicates the render arrays for convenient analysis. The original native result SHA-256 is recorded in the manifest; extraction does not replace its full branch-history record.

Reproduce with Python, NumPy and Pillow:

    python render_recovered_dynamics.py viscoelastic_network_recorded.json rendered

The renderer does not import FiberNet or any simulation engine. It verifies frame identity, kinetic energy against stored masses and velocities, and unchanged fixed nodes before producing the GIF and contact sheet. Existing temporal-convergence evidence belongs to the original benchmark; a rendered GIF is not a new convergence test.

Chinese caption: 真实保存的黏弹网络动力学：105 个节点、273 条物理边和 39 个固定节点。显示实际轴向应力、运动及能量转移；几何变形明确放大 3000 倍，物理耗散与数值耗散分别呈现。采用合成参数，本次仅重绘已有轨迹，未重跑物理求解。
