# FiberNet 4.43.0

- **Faster prescribed-motion Tet4 mechanics.** Six signed loads on the 1,056-tet welded example took 0.47 s instead of 1.47 s on one BLAS thread (median of five runs). Geometry and strain checks retain scalar recovery when needed.
- **Runnable resistance attribution.** Use the same two-terminal solid to separate shape and prescribed conductivity effects. The example reports four resistances and their interaction.

```sh
python -m pip install "fibernet==4.43.0"
python -m fibernet.examples.tet4_mechanoelectric_looped_network --resistance-attribution
```
