# Homepage media

## Tensile recruitment

The horizontal axis is stretch ratio, L/L₀. The overview retains the original 18 saved frames and timing with the corrected axis label.

[Model and trajectory](MEDIA_GALLERY.md#recorded-stretch-and-recruitment) · [Editable final frame](media/homepage_overview/tensile_recruitment_labeled_final.svg) · [Label renderer](../scripts/relabel_saved_tensile.py) · [Pixel-preservation receipt](media/homepage_overview/tensile_label_receipt.json)

## GINE conductance

The overview replays the 40 saved training epochs, then shows the final predictions for 16 cases from four held-out parent networks. Training and validation loss are in mS²; the evaluation axes are in mS.

[Training history](media/new/learning/data/gine/training.json) · [Final results](media/new/learning/data/gine/final_results.json) · [Study and data splits](media/new/learning/README.md) · [Renderer](../scripts/render_homepage_gine.py) · [Render receipt](media/homepage_overview/gine_receipt.json)

## FiberScope workspaces

Saved screenshots show learning, inverse design, surface mapping and fabrication. Whole-window views alternate with control close-ups. The three export buttons are separate crops from the same saved fabrication screen, arranged vertically for readability.

[Original screenshots](../FiberScope/docs/screenshots/README.md) · [Desktop guide](../FiberScope/README.md) · [Renderer](../scripts/render_homepage_saved_ui.py) · [Crop coordinates and source hashes](media/homepage_overview/fiberscope_receipt.json)

## Static interventions

These grids show saved final networks and complete reaction curves, normalized by the original network's final reaction.

![Kagome static intervention results at stretch ratio 1.40](media/homepage_overview/intervention_kagome_grid.svg)

![Ring static intervention results at stretch ratio 1.40](media/homepage_overview/intervention_ring_grid.svg)

The saved reduced axial-model cases remove 1.0042% of edge length for kagome and 2.2234% for ring while retaining an Euler route, as recorded in the [kagome](media/route_preserving_intervention_kagome.json) and [ring](media/route_preserving_intervention_ring.json) manifests. Color marks strain-threshold recruitment and grip-spanning components, not force flow. High-late-strain deletion lowers final reaction to 0.850 of the original for kagome and 0.814 for ring. Low-early-strain and fixed-random results coincide in both cases.

[Full study and adverse results](CONSTRAINED_CYCLE_INTERVENTION_2026-09-24.md) · [Cross-model study](INDEPENDENT_FEM_CYCLE_INTERVENTION_2026-09-24.md) · [Saved results](../benchmarks/results/constrained_cycle_intervention.json) · [SVG relayout source](../scripts/relayout_homepage_interventions.py) · [Path-preservation receipt](media/homepage_overview/interventions_receipt.json)

## Original gallery

All 19 original GIFs are directly embedded in each language of the repository homepage, grouped by workflow, alongside three overview derivatives. Their bytes and recorded hashes are unchanged. The [selection manifest](media/homepage_overview/selection.json) records the 22 displayed animations and original hashes; the [media gallery](MEDIA_GALLERY.md) retains source data, parameters, static alternatives and renderers.

The language headings use independent native disclosures, with English open and Chinese closed initially. They are not script-driven mutually exclusive tabs. GIFs loop without a README playback control; use the linked static alternatives when motion is distracting.

Camera orbits, geometric design sweeps, graph traversal and saved interface tours are labeled separately from recorded simulation and training histories. LAMMPS and Taichi show actual saved displacement amplified ×1,000; viscoelastic motion is amplified ×3,000. Fracture observations use undeformed geometry, and thermal evolution uses fixed geometry with one-way Joule heating. This media bundle does not contain a deformation-coupled electrical trajectory. The GINE history comes from an archived research trainer; RL records use prescribed actions, not a trained policy.
