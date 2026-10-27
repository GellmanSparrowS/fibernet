# Maintaining the README API guide

The README presents 49 representative interfaces in seven workflows. Its nested disclosures work directly on GitHub; the SVGs summarize the workflow groups.

`code_registry.json` remains the module inventory. `navigation-overlay.json` contains only the selected interfaces, their public ownership names, bilingual usage descriptions and explicit data handoffs. A handoff describes a connection the caller makes, not an automatic conversion or a claim that different physical models are interchangeable.

## Update the guide

1. Check the selected declarations, public exports, aliases and input/output contracts against the source revision. Set `sourceCommit` in `docs/api/navigation-overlay.json` to that full commit. Source links remain pinned to this revision even when later documentation-only commits update the README.
2. Edit the overlay descriptions and relationships where needed. Keep stable node IDs and the existing registry keys; subpackage APIs must retain their qualified namespace rather than appear as root exports.
3. Generate and verify:

   ```bash
   python scripts/update_api_navigation.py --write
   python scripts/update_api_navigation.py --check
   python tests/test_api_navigation.py
   ```

The generator uses Python's standard-library AST reader, without importing FiberNet, running examples or initializing optional backends. It updates only the two marked README sections, the matching Chinese section in `README_CN.md`, and `docs/media/ui/api-relationship-map-{en,zh}.svg`. Existing galleries and prose outside those markers are byte-preserved. Generated insertions adopt each document’s LF or CRLF line endings; mixed line endings reject rather than silently normalize the document.

The tests check reproducibility, selected source declarations, exact namespace/export links, basic native markup and the dataflow descriptions. They do not execute the displayed scientific examples or qualify optional runtimes. When a source implementation changes, review its usage contract before updating the pinned revision; syntax and a matching name alone cannot establish semantic compatibility.

## Maintain the module inventory separately

Use the existing `python scripts/update_code_registry.py` from a complete repository checkout when refreshing the module inventory. It scans `fibernet`, `tests`, `scripts`, `examples`, `benchmarks` and `manuscript`, preserves curated purposes, refreshes symbols/hashes, and marks unseen paths removed. Do not run it on a partial source snapshot. Updating this selective guide does not require rewriting unrelated registry entries.
