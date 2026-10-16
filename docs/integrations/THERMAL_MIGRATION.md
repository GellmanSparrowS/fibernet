# Thermal entry points: support and migration

The old `fibernet.api.simulate_thermal` signature is retained solely to raise a clear `NotImplementedError`. Its former steady-state/effective-conductivity behavior is unsupported. It does not import a missing backend or silently dispatch to another physical model. No root `fibernet.simulate_thermal` or root transient alias was added.

Thermal template creation and thermal configuration execution reject explicitly before writing a template, generating a network or creating output. Loading an existing input configuration is permitted so its simulation type can be identified; this is not a simulation. Non-thermal configuration branches are unchanged.

`examples/03_thermal_analysis.py` is an import-safe notice. Executing it exits with code2 and explains unsupported CTE/thermal deformation. Its original bytes are preserved at `examples/_archived/03_thermal_analysis.py`; the archived source is historical and not supported for execution.

The four larger legacy callers (`research_case_study.py`, `comprehensive_demo.py`, `full_workflow.py`, `integration_examples.py`) are unchanged. Their README entries explicitly identify unsupported historical thermal calls and do not promise complete workflows. Other portions of those scripts were not audited or removed in this small migration.

## Current independent example

From the project root:

```sh
python -m examples.transient_thermal_explicit
```

The two-node/parallel-edge example supplies C[J/K], g[W/K], T0[K], Q[W], fixed temperature and time[s] explicitly. The default0.02s is only a quick entry-point demonstration, not full relaxation or a new time-accuracy study. There is no geometry/material inference, effective conductivity, CTE or thermal deformation. It calls the existing frozen `fibernet.sim.transient_thermal` module directly; its physical contract is unchanged.

Importing the example does not run a solver or write output. `build_inputs()` provides inspectable inputs; `main()` performs the explicit solve only when invoked. Numerical failures print the last accepted time, or explicitly report no accepted frame if the initial derived fields failed, then return nonzero. t=0 interval residuals are unavailable; fixed-node residuals are not confused with reservoir heat input.

## Verification distinctions

Compatibility tests use test doubles to inspect the requested inputs and error handling; import guards plus a solver-call sentinel verify no implicit solve. OldAPI/config tests verify explicit early rejection. The legacy notice is exercised as a subprocess without importing a backend.

One real CLI invocation on the existing two-node input is recorded separately. It completed to0.02s in2.281s process wall time with approximately114MiB sampled RSS. No other real solver invocation or parameter sweep was added for this migration. A later wording-only clarification labels its short duration; that does not constitute a second real run.

The first test harness timed out under the repository default pytest-forked setup. Original timeout logs are retained. Restricting an import guard to example globals and preloading package dependencies before fork resolved the harness; tests without fork also passed. This is test-environment evidence, not a thermal-model failure. Full repository integration verification remains a separate gate.
