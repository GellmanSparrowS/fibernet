# FiberNet 4.42.1

Tet4 resistance attribution now matches nodes using exact 64-bit integers.
If you used 4.42.0 with node IDs above 2⁵³, rerun the affected resistance
comparisons. The pore-wall mapping is unchanged.

```sh
python -m pip install -U "fibernet==4.42.1"
```
