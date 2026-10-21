# Interpreting the homepage examples

The gallery brings geometry, simulation and learning together. This guide identifies what each visual encodes and where to find the full model and data.

## Geometry, motion and manufacturing

The 3D topology panels and the left half of the hero use camera rotation around fixed geometries. The hero's right half replays recorded reduced-model deformation. The reference-fiber profile comparison is geometric parameterization rather than a loading simulation. Recruitment measures connectivity above an axial-strain threshold, rather than a full force-flow map. See the [media gallery](MEDIA_GALLERY.md) and [recruitment method](METHODS_TENSILE_RECRUITMENT.md).

The learning/inverse-design and surface/export tours use saved interface images. Continuous-route playback shows graph traversal; route and solid exports are geometric outputs rather than printer execution records or solid-mechanics solutions. Route-preserving interventions are controlled structural comparisons. See the [manufacturing method](METHODS_MANUFACTURING_UNIFIED.md) and [controlled interventions](CONSTRAINED_CYCLE_INTERVENTION_2026-09-24.md).

## Dynamics, fracture and transport

LAMMPS and Taichi animations use their respective saved synthetic benchmarks, with displacement amplified 1,000 times. The viscoelastic example amplifies displacement 3,000 times and separates material dissipation from backward-Euler numerical loss. These numerical demonstrations do not establish experimentally calibrated material behavior. The linked [backend records](media/new/backends/README.md) and [viscoelastic model](media/new/viscoelastic/README.md) provide units, trajectories and energy definitions.

The fracture visual shows quasistatic member removal on the undeformed reference geometry, rather than fragment motion. Its boundary-dominated failures and disconnected final load path remain part of the [scenario and event record](media/new/fracture/README.md). The torus electrical field is a fixed synthetic steady solution. The [thermal visual](media/new/thermal/README.md) shows one-way Joule heating on fixed geometry, separately from mechanical deformation.

## Learning and design

The [GINE animation](media/new/learning/README.md) comes from an archived research trainer. The solved-field panel stays fixed while learning curves evolve; final predictions are evaluated on parent-separated networks. The full record includes individual prediction errors, placement-contrast outcomes and baseline comparisons. Current public GINE data/model adapters have their own [API documentation](integrations/EDGE_GINE.md).

The RL animation records prescribed actions and native environment responses, rewards and termination. It is an environment-interaction example, not a trained policy. CEM target-curve search is a separate derivative-free optimization workflow. The runnable physical-learning demo starts with synthetic data; saved physical labels supply the response-learning task. See the [learning records](media/new/learning/README.md) and [target-curve methods](METHODS_CURVE_INVERSE_UNIFIED.md).

## Installation and API selection

PyPI 4.2.0 is the published package; the 4.3.0 development checkout contains newer APIs. Module-level solvers have explicit input and result contracts. Select supported entry points with the [API map](API_LAYERS.md), including the distinction between beam-frame FEM and the reduced spring/bending/contact model.
