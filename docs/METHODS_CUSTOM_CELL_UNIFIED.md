# Shared custom-cell configuration and route compilation

The existing FiberScope application stores user cells as a JSON object keyed by unit name. Each value contains a list of two-dimensional node coordinates in a unit cell, undirected edge endpoint pairs, optional bilingual labels, and optional editor settings. The development branch exposes this format through the original `fibernet` package. Both the APP and library now use the same bounded validation and atomic persistence implementation; the APP retains its editor and built-in name collision policy.

## Validation and coordinate convention

The unit cell coordinates are dimensionless. Multiplying by a positive `scale` (default 10) yields planar graph coordinates; the graph's default fiber radius is 0.1 in those coordinates. A valid custom cell has 2–48 nodes, 1–256 undirected edges, finite coordinates with absolute value no more than 10^6, integer endpoints, no self-edge or duplicate undirected edge, and one connected component. The validation does not require the drawn coordinates to fall inside [0, 1], because the unit box is a period rather than a clipping boundary. The input may supply editor `settings`, but these have no effect on graph geometry until explicitly interpreted by a caller.

The registry reads at most 1 MB and 256 cells by default. It can skip invalid records when loading historical APP data into the GUI; strict loading is the public default. Saving a cell first loads an existing registry, preserving unrelated entries. Complete replacement is an explicit operation. Files are serialized as UTF-8 JSON to a temporary sibling and atomically replaced. This guards against a partial target file on interrupted writes; simultaneous multi-process editing is not locked and remains outside the present contract.

## Reproducible use

`python -m examples.custom_cell_workflow --output-dir demo_custom_cell` writes an APP-compatible `custom_units.json`, reloads the cell in a fresh registry instance, tiles it 2×2 through the public planar manufacturing compiler, and saves graph and route arrays in `custom_route.npz`. It checks that the compiled graph is connected, every vertex has even degree, and the route closes. These are topological and geometric checks; nozzle collision, joining, adhesion, material failure, and device behavior are not inferred. The APP and library graph positions and edge ordering are checked exactly by a cross-end regression test.

This module is an API and persistence contract, not a claim that arbitrary user cells are physically printable. For the route compiler and edge identity semantics, see [the shared manufacturing methods](METHODS_MANUFACTURING_UNIFIED.md).
