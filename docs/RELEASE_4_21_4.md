# FiberNet 4.21.4

This patch catches precision loss in prescribed DC voltages and adds
examples that connect simulation results to heating, grouped training
data, and manufacturing centerlines. The default dependencies are unchanged.

## Prescribe DC voltages without losing the difference

When a supplied integer, fraction, or wider NumPy float is not exactly
representable as binary64, `solve_linear_transport` raises a value error
instead of silently changing the prescribed voltage. Repeated constraints
are compared before conversion. Ordinary Python floats are already
binary64 and continue to work. For custom numeric types, convert explicitly
if rounding is intended.

If the absolute level obscures a small voltage difference, shift the
reference with the original values first. For example, for a 1 S rod,
the exact integer pair `(2**53 + 1, 2**53)` can be entered relative to
the second electrode as `(1, 0)`, giving 1 A and 1 W. Converting the
original pair to floats before subtracting cannot recover that difference.
The separate MCP adapter also rejects an inexact integer-encoded voltage
without closing the session.

## Run the examples from the matching source archive

```sh
python -m pip install "fibernet==4.21.4"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.21.4"
python -m tarfile -e fibernet-4.21.4.tar.gz
cd fibernet-4.21.4
python -m examples.four_terminal_joule_heating
```

The four-terminal example compares an insulated graph and a graph with one
node held at 300 K, using the same mixed-voltage electrical result. It
accounts for the 3.4991 J of supplied heat as storage, plus heat removed by
the fixed-temperature node when present. It solves lumped node
temperatures, not a continuous solid temperature field.

Install the optional scikit-learn splitter to keep material variants of
the same simulated network together during training-data preparation:

```sh
python -m pip install scikit-learn
python -m examples.heterogeneous_3d_grouped_dataset --output grouped_fem
```

This example creates five 3D beam results, checks cut forces and strain
energy against an axial reference, and separates the samples by the
parent structure. It does not train a model. For a different downstream
use, the [manufacturing guide](MULTISTROKE_MANUFACTURING.md) shows how to
transfer a deformed beam network to centerline strokes using stable node IDs.
