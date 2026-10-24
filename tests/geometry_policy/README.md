# Geometry-policy contract tests

These portable tests exercise geometry acceptance, caller dispatch, streaming
batch resume, dataset planning/checkpoints, RL rejection reporting and result
JSON transport. They read the checkout's production source, use its import-free
geometry helper and explicitly selected Python definitions, and supply inert
collaborators in place of optional providers and numerical array operations.

They do not run a native physics solver or train an RL policy. They do not prove
numerical equivalence, installed-package integration, actual generator output,
GPU support or curved-beam mechanics. A transported geometry record is ordinary
provenance, not authenticated evidence of a solver run.

Run the standalone standard-library suite from the repository root:

```sh
python -m unittest discover -s tests/geometry_policy -t tests -p 'test_*.py' -v
```

To select this directory with pytest while avoiding the repository's native
provider setup, disable plugin autoload and conftest loading:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest tests/geometry_policy -q -o addopts=''
```

The environment-variable prefix shown is POSIX shell syntax. On PowerShell,
set `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'` before the Python command.
