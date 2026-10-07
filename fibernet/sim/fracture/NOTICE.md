# Fracture module attribution and scope

Copyright (c) 2026 FiberNet Contributors. Distributed under the repository MIT License.

The circular-beam functions, sparse assembly/solve functions, and stress-removal functions were extracted from the authors’ independent linear spatial reference and subsequent private lightweight fracture implementation. PROVENANCE.json records source hashes and exact extracted function names. Module import locations and the public driver/adapter changed; no LAMMPS implementation or binary is included. The source mathematical reference was checked against OpenSees; this module does not copy or bundle OpenSees code.

NumPy and SciPy supply array/sparse numerical routines; psutil and threadpoolctl provide cooperative resource checks and serial numerical-library limits. Their own licenses remain separate.

This experimental linear-static model is not a geometrically nonlinear/contact model or a calibrated material law. Event tolerance is a numerical grouping rule, not a physical fracture process zone or material uncertainty. Free-fragment absolute poses remain undefined.
