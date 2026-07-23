"""
Machine Learning Tools for FiberNet — Comprehensive AI Pipeline.

Modules
-------
- features: Feature extraction from fiber networks
- utils: One-line ML workflows (train, CV, visualize)
- models: PyTorch model zoo (MLP, ResNet, Attention, Ensemble)
- training: Advanced training loops with checkpointing
- gnn: Graph Neural Networks (GCN, GAT, GraphSAGE)
- generative: VAE / CVAE for structure generation
- diffusion: DDPM / DDIM diffusion models for structure generation
- gan: GANs (standard, WGAN-GP, conditional) for structure generation
- pinn: Physics-Informed Neural Networks
- field_prediction: U-Net / field MLP for 2D/3D spatial field prediction
- neural_operator: Fourier Neural Operator (FNO) / DeepONet
- inverse_design: Property-to-structure inverse design
- optimization: Multi-objective optimization (Optuna, NSGA-II)
- active_learning: Efficient data acquisition strategies
- transfer_learning: Pre-train + fine-tune, few-shot, domain adaptation
- multiscale: Multi-scale feature extraction and hierarchical learning
- symbolic: Symbolic regression for interpretable formulas
- xai: Explainable AI (SHAP, permutation importance)
- dataset: Dataset generation and management

Quick Start
-----------
>>> from fibernet.ml import train_predictor, cross_validate
>>> model, metrics = train_predictor(X, y, model_type="rf")

>>> # Diffusion model generation
>>> from fibernet.ml.diffusion import FiberDiffusion, DiffusionTrainer
>>> diff = FiberDiffusion(n_features=20)
>>> DiffusionTrainer(diff).fit(X_train, epochs=200)
>>> X_gen = diff.sample(n=100)

>>> # GAN generation
>>> from fibernet.ml.gan import FiberWGAN, GANTrainer
>>> gan = FiberWGAN(n_features=20, latent_dim=32)
>>> GANTrainer(gan).fit(X_train, epochs=300)

>>> # GNN
>>> from fibernet.ml.gnn import FiberGNN, graph_from_structure
>>> gnn = FiberGNN(node_dim=5, hidden=64, n_outputs=1, n_layers=3)

>>> # Inverse design
>>> from fibernet.ml.inverse_design import TandemNetwork, InverseDesignTrainer
>>> tandem = TandemNetwork(n_features=20, n_properties=3)
>>> InverseDesignTrainer(tandem).fit(X, y, epochs=200)
"""

# Core utilities
from fibernet.ml.utils import (
    train_predictor,
    cross_validate,
    compare_models,
    predict_from_csv,
    plot_predictions,
    plot_feature_importance,
    plot_residuals,
    plot_learning_curve,
)


# Dataset
from fibernet.ml.dataset import FiberNetDataset

__all__ = [
    # Utils
    "train_predictor",
    "cross_validate",
    "compare_models",
    "predict_from_csv",
    "plot_predictions",
    "plot_feature_importance",
    "plot_residuals",
    "plot_learning_curve",
    # Features
    # FEM
    "BeamFrameFEM",
    "BeamFrameFEM_v6",
    # Dataset
    "FiberNetDataset",
]

# Lazy imports for heavy submodules
def __getattr__(name):
    """Lazy import for optional heavy submodules."""
    _submodules = {
        # Models
        "FiberMLP": "fibernet.ml.models",
        "FiberResNet": "fibernet.ml.models",
        "FiberAttentionNet": "fibernet.ml.models",
        "FiberUncertaintyNet": "fibernet.ml.models",
        "FiberMultiTaskNet": "fibernet.ml.models",
        "FiberEnsemble": "fibernet.ml.models",
        # Training
        "train_model": "fibernet.ml.training",
        "TrainingHistory": "fibernet.ml.training",
        "save_model_bundle": "fibernet.ml.training",
        "load_model_bundle": "fibernet.ml.training",
        # GNN
        "FiberGNN": "fibernet.ml.gnn",
        "graph_from_structure": "fibernet.ml.gnn",
        "train_gnn": "fibernet.ml.gnn",
        # Generative
        "FiberVAE": "fibernet.ml.generative",
        "FiberCVAE": "fibernet.ml.generative",
        "train_vae": "fibernet.ml.generative",
        # Diffusion
        "FiberDiffusion": "fibernet.ml.diffusion",
        "ConditionalFiberDiffusion": "fibernet.ml.diffusion",
        "DiffusionTrainer": "fibernet.ml.diffusion",
        "GraphDiffusionGenerator": "fibernet.ml.diffusion",
        # GAN
        "FiberGAN": "fibernet.ml.gan",
        "FiberWGAN": "fibernet.ml.gan",
        "FiberCGAN": "fibernet.ml.gan",
        "GANTrainer": "fibernet.ml.gan",
        # PINN
        "PhysicsLoss": "fibernet.ml.pinn",
        "PINNModel": "fibernet.ml.pinn",
        "train_pinn": "fibernet.ml.pinn",
        # Field prediction
        "FiberUNet": "fibernet.ml.field_prediction",
        "FiberFieldMLP": "fibernet.ml.field_prediction",
        "train_field_model": "fibernet.ml.field_prediction",
        # Neural operator
        "FiberFNO": "fibernet.ml.neural_operator",
        "FiberDeepONet": "fibernet.ml.neural_operator",
        "NeuralOperatorTrainer": "fibernet.ml.neural_operator",
        # Inverse design
        "InverseDesignNet": "fibernet.ml.inverse_design",
        "TandemNetwork": "fibernet.ml.inverse_design",
        "InverseDesignTrainer": "fibernet.ml.inverse_design",
        # Active learning
        "UncertaintySampling": "fibernet.ml.active_learning",
        "ActiveLearningLoop": "fibernet.ml.active_learning",
        # Transfer learning
        "FiberTransferNet": "fibernet.ml.transfer_learning",
        "FiberPrototypicalNet": "fibernet.ml.transfer_learning",
        "DomainAdapter": "fibernet.ml.transfer_learning",
        # Multi-scale
        "MultiScaleFeatureExtractor": "fibernet.ml.multiscale",
        "HierarchicalEncoder": "fibernet.ml.multiscale",
        "ScaleBridgeModel": "fibernet.ml.multiscale",
        # Symbolic
        "SymbolicRegressor": "fibernet.ml.symbolic",
        # XAI
        "explain_model": "fibernet.ml.xai",
        "permutation_importance": "fibernet.ml.xai",
        # Optimization
        "FeatureExtractor": "fibernet.ml.features",
        "GraphFeatureExtractor": "fibernet.ml.features",
        "MultiObjectiveOptimizer": "fibernet.ml.optimization",
        "ParetoAnalysis": "fibernet.ml.optimization",

        # GFlowNet
        "FiberGFlowNet": "fibernet.ml.gflownet",
        "GFlowNetTrainer": "fibernet.ml.gflownet",
        "StructureState": "fibernet.ml.gflownet",
        "connectivity_reward": "fibernet.ml.gflownet",
        "property_target_reward": "fibernet.ml.gflownet",
        # Differentiable physics
        "DifferentiableSpringNetwork": "fibernet.ml.differentiable_physics",
        "DifferentiableBeamNetwork": "fibernet.ml.differentiable_physics",
        "DifferentiableFEA": "fibernet.ml.differentiable_physics",
        "PhysicsOptimizer": "fibernet.ml.differentiable_physics",
        "DifferentiableMaterialModel": "fibernet.ml.differentiable_physics",
        # Physics-informed GNN
        "PhysicsInformedGNN": "fibernet.ml.pinn_gnn",
        "PhysicsInformedMessagePassing": "fibernet.ml.pinn_gnn",
        "PhysicsGNNTrainer": "fibernet.ml.pinn_gnn",
        "ForceBalanceLoss": "fibernet.ml.pinn_gnn",
        "ConstitutiveLoss": "fibernet.ml.pinn_gnn",
        "EnergyConservationLoss": "fibernet.ml.pinn_gnn",
        # Neural ODE
        "FiberNeuralODE": "fibernet.ml.neural_ode",
        "ODESolver": "fibernet.ml.neural_ode",
        "StressRelaxationODE": "fibernet.ml.neural_ode",
        "CreepODE": "fibernet.ml.neural_ode",
        "FatigueODE": "fibernet.ml.neural_ode",
        "NeuralODETrainer": "fibernet.ml.neural_ode",
        # Conservative NN
        "HamiltonianNN": "fibernet.ml.conservative_nn",
        "LagrangianNN": "fibernet.ml.conservative_nn",
        "EnergyConservingNN": "fibernet.ml.conservative_nn",
        "MomentumConservingNN": "fibernet.ml.conservative_nn",
        "DivergenceFreeNet": "fibernet.ml.conservative_nn",
        "ConservativeLoss": "fibernet.ml.conservative_nn",
        "ConservativeTrainer": "fibernet.ml.conservative_nn",
        # Beam Frame FEM
        "BeamFrameFEM": "fibernet.ml.beam_frame_fem_v6",
        "BeamFrameFEM_v6": "fibernet.ml.beam_frame_fem_v6",
    }

    if name in _submodules:
        import importlib
        mod = importlib.import_module(_submodules[name])
        return getattr(mod, name)

    raise AttributeError(f"module 'fibernet.ml' has no attribute '{name}'")
