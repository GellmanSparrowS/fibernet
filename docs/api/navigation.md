# Maintaining the README API guide

The README presents 49 representative interfaces in seven workflows. Its nested disclosures work directly on GitHub; the static SVGs summarize 50 documented, directed dataflow handoffs between and within those workflows.

The SVGs are static directed chord summaries, not clickable navigation or runtime call graphs. Native branches, links and a complete linear handoff list provide the navigation. Relationship links target the corresponding API entry in the same language, with separate source links. Entry anchors use stable node IDs (`api-en-<id>` / `api-zh-<id>`), not translated labels. If GitHub leaves a destination hidden, expand its workflow group; this guide does not promise automatic disclosure expansion or synchronized language/view switching. English remains visible; Chinese is a native same-page disclosure.

GitHub renders an authored `<a name="api-en-graph"></a>` as `name="user-content-api-en-graph"`. Keep `entry_anchor()` names unchanged, and use `entry_href()` for relationship destinations such as `#user-content-api-en-graph`. The linear table and per-entry handoffs follow this same rule in both languages. This targets GitHub's rendered README anchors without changing disclosure defaults; other Markdown renderers may handle anchor names differently.

`code_registry.json` remains the module inventory. `navigation-overlay.json` contains only the selected interfaces, their public ownership names, bilingual usage descriptions and explicit data handoffs. A handoff describes a connection the caller makes, not an automatic conversion or a claim that different physical models are interchangeable.

## Relationship projection and acceptance

All views consume the same `derive()` result. `relationships()` sorts dataflow records by source workflow order, source ID, target ID and English label; it aggregates ordered workflow pairs without changing the original records. The current overlay has 101 typed relations: 50 dataflows, 49 workflow memberships, and two record-ownership relations (`StructureGraph` owns `SNode` and `SEdge`). These are selected documentation relationships, not an exhaustive API registry, inheritance tree or observed call graph.

1. **Faithful mapping:** 36 cross-workflow handoffs become 15 directed chords; 14 within-workflow handoffs become six explicit self-loops. Each ordered pair has a textual count; every group displays its within-workflow count, including zero. Arrowheads show direction. The complete 50-row Markdown list retains each source, target, contract and both endpoint source references. The two ownership relations appear separately from dataflow; workflow memberships remain the native branch tree.
2. **Text-first access:** meaningful alt text describes the static summary, while the linear table and named links work without the SVG. Qualified API names, direction and relation type are explicit text, not color-only encodings. Do not promise SVG clicks or automatic expansion of folded anchor destinations. Sources identify declarations at the pinned revision; they are not runtime certification.
3. **Bounded maintenance:** reuse the two existing SVG paths, without a new registry, frontend, dependency, external font or runtime request. Each SVG is capped at 16 KiB. README growth is capped at 64 KiB against the accepted baseline (157,596 bytes for `README.md`; 77,775 bytes for `README_CN.md`). The generator remains standard-library-only and does not import providers. Reference-style source links keep the complete handoff list compact.

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

Contract checks also cover target-name choices for `CurveInverseDesigner.run`, explicit geometry-policy handoffs, bilingual wording, and language-specific relationship destinations. Keep constructor arguments separate from method arguments: inverse search accepts a registered `target_name`, not an arbitrary target-curve array. A custom `evaluator(graph, target_name)` must return `(objective_to_minimize, run)` with a finite scalar objective; lower is better, so negate a custom maximization score before returning it. Reduced-beam input must lie in one XY plane and have no nonempty edge `internal_points`; callers must first represent those points as graph nodes and straight edges. Spring/Taichi rejects those interior points by default; explicit `geometry_policy='endpoint_chord'` ignores them mechanically and does not provide curved-fiber mechanics.

## Maintain the module inventory separately

Use the existing `python scripts/update_code_registry.py` from a complete repository checkout when refreshing the module inventory. It scans `fibernet`, `tests`, `scripts`, `examples`, `benchmarks` and `manuscript`, preserves curated purposes, refreshes symbols/hashes, and marks unseen paths removed. Do not run it on a partial source snapshot. Updating this selective guide does not require rewriting unrelated registry entries.
