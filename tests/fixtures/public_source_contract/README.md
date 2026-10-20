# Public source regression baselines

These text fixtures reproduce six modules already published in the
[FiberNet public API commit](https://github.com/GellmanSparrowS/fibernet/commit/8ed9db07afd6aed55f4d326e66d05f138803701e).
They retain the repository MIT license and are test data, not imported modules.

Tests compare full readable syntax trees with these baselines. The engine,
energy helper and visualization facade retain exact UTF-8 byte checks; the
other three facades permit checkout line-ending changes and comments.
They also exercise event/state behavior, explicit LAMMPS routing guards,
observer orchestration and locale-independent source loading. A source change
requires reviewing its behavior and updating its applicable baseline together.
No native physics run is implied by a source-contract pass.

The root package facade baseline also includes the additive closed-route graph
mapping and multi-stroke manufacturing exports. Its complete readable source is
reviewed with those exports; the other five module baselines remain unchanged.

The root facade also includes the one-way Joule load and workflow exports;
its baseline is updated with the additive public API.

The root and easy facades also expose the additive quasistatic viscoelastic
history API. Other backends and the remaining baseline modules are unchanged.
