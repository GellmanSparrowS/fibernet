# PDB export for sampled networks

```python
from fibernet.api import export, load
from fibernet.gen import ZigZagGenerator

network = ZigZagGenerator(n_cols=3, n_rows=3).to_fiber_network()
export(network, 'network.pdb')
restored = load('network.pdb')
```

The default example exports 45 fibers as 14,895 sampled beads. PDB coordinates
occupy fixed-width columns: atom serials use five digits and residue sequence
numbers use four. FiberNet retains global residue numbering up to 9,999 beads;
larger exports number residues separately within the existing A–Z chain labels.
When a chain label is reused, its residue sequence continues from the previous
fiber using that label. Atom serials and CONECT links remain global. All coordinate records precede
the connectivity records, so external PDB readers can read the complete network.

Exports support at most 99,999 beads overall and 9,999 beads per chain label in
the larger-file mode. Capacity is checked before opening the destination.
Sampling, coordinates and chain assignment follow the existing exporter. PDB
coordinates use the existing nanometer-to-angstrom conversion and three decimal
places; import reconstructs the sampled fiber chains from CONECT records.

For small files, existing ATOM payloads are unchanged; connectivity records now
appear together after all coordinates.

Reference: [wwPDB coordinate records](https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM).
