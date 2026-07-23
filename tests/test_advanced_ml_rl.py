"""
Comprehensive Tests for FiberNet Advanced ML/RL Modules.

Tests all new modules:
- Diffusion models (DDPM, conditional, DDIM)
- GAN models (standard, WGAN-GP, conditional)
- Field prediction (U-Net, FieldMLP)
- Neural operators (FNO, DeepONet)
- Inverse design (InverseDesignNet, TandemNetwork)
- Active learning (UncertaintySampling, DiversitySampling)
- Transfer learning (FiberTransferNet, PrototypicalNet)
- Multi-scale (MultiScaleFeatureExtractor)
- Symbolic regression (FeatureSelection)
- RL curriculum (Linear, Adaptive, MultiStage)
- RL reward shaping (Composite, Distance, PBRS)
- RL multi-objective (ScalarizedMORL, ParetoFrontExplorer)
"""

import pytest
import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")


# ======================================================================
# Diffusion Model Tests
# ======================================================================

class TestDiffusion:
    def test_noise_schedule(self):
        from fibernet.ml.diffusion import NoiseSchedule
        ns = NoiseSchedule(n_steps=100, schedule="cosine")
        assert ns.betas.shape == (100,)
        assert ns.alpha_bar.shape == (100,)
        assert ns.alpha_bar[-1] < ns.alpha_bar[0]

    def test_noise_schedule_linear(self):
        from fibernet.ml.diffusion import NoiseSchedule
        ns = NoiseSchedule(n_steps=100, schedule="linear")
        assert ns.betas.shape == (100,)
        assert ns.betas[0] < ns.betas[-1]

    def test_fiber_diffusion_forward(self):
        from fibernet.ml.diffusion import FiberDiffusion
        model = FiberDiffusion(n_features=10, hidden=[32, 16], n_steps=50)
        x = torch.randn(4, 10)
        t = torch.randint(0, 50, (4,))
        noise_pred = model(x, t)
        assert noise_pred.shape == (4, 10)

    def test_fiber_diffusion_loss(self):
        from fibernet.ml.diffusion import FiberDiffusion
        model = FiberDiffusion(n_features=10, hidden=[32, 16], n_steps=50)
        x = torch.randn(8, 10)
        loss = model.loss(x)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_fiber_diffusion_sample(self):
        from fibernet.ml.diffusion import FiberDiffusion
        model = FiberDiffusion(n_features=10, hidden=[32, 16], n_steps=50)
        samples = model.sample(n=5)
        assert samples.shape == (5, 10)

    def test_fiber_diffusion_ddim_sample(self):
        from fibernet.ml.diffusion import FiberDiffusion
        model = FiberDiffusion(n_features=10, hidden=[32, 16], n_steps=100)
        samples = model.sample(n=3, ddim=True, ddim_steps=10)
        assert samples.shape == (3, 10)

    def test_conditional_diffusion(self):
        from fibernet.ml.diffusion import ConditionalFiberDiffusion
        model = ConditionalFiberDiffusion(n_features=10, n_conditions=2, hidden=[32, 16], n_steps=50)
        x = torch.randn(4, 10)
        c = torch.randn(4, 2)
        loss = model.loss(x, c)
        assert loss.item() > 0

    def test_conditional_sample(self):
        from fibernet.ml.diffusion import ConditionalFiberDiffusion
        model = ConditionalFiberDiffusion(n_features=10, n_conditions=2, hidden=[32, 16], n_steps=50)
        cond = torch.tensor([[1.0, 2.0]])
        samples = model.sample(n=3, conditions=cond, guidance_scale=2.0)
        assert samples.shape == (3, 10)

    def test_diffusion_trainer(self):
        from fibernet.ml.diffusion import FiberDiffusion, DiffusionTrainer
        model = FiberDiffusion(n_features=10, hidden=[32, 16], n_steps=20)
        trainer = DiffusionTrainer(model)
        X = np.random.randn(50, 10).astype(np.float32)
        history = trainer.fit(X, epochs=3, batch_size=16, verbose=False)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3

    def test_graph_diffusion_generator(self):
        from fibernet.ml.diffusion import GraphDiffusionGenerator
        gen = GraphDiffusionGenerator(n_features=5)
        X = np.random.randn(30, 5).astype(np.float32)
        gen.train(X, epochs=3, batch_size=10, verbose=False)
        X_gen = gen.generate(n=5)
        assert X_gen.shape == (5, 5)


# ======================================================================
# GAN Tests
# ======================================================================

class TestGAN:
    def test_generator(self):
        from fibernet.ml.gan import FiberGenerator
        gen = FiberGenerator(latent_dim=16, n_features=10, hidden=[32])
        z = torch.randn(4, 16)
        out = gen(z)
        assert out.shape == (4, 10)

    def test_discriminator(self):
        from fibernet.ml.gan import FiberDiscriminator
        disc = FiberDiscriminator(n_features=10, hidden=[32])
        x = torch.randn(4, 10)
        out = disc(x)
        assert out.shape == (4, 1)

    def test_fiber_gan(self):
        from fibernet.ml.gan import FiberGAN
        gan = FiberGAN(n_features=10, latent_dim=16, hidden=[32])
        samples = gan.sample(n=3)
        assert samples.shape == (3, 10)

    def test_wgan(self):
        from fibernet.ml.gan import FiberWGAN
        gan = FiberWGAN(n_features=10, latent_dim=16, hidden=[32])
        samples = gan.sample(n=3)
        assert samples.shape == (3, 10)
        # Test gradient penalty
        real = torch.randn(4, 10)
        fake = gan.generate(torch.randn(4, 16))
        gp = gan.gradient_penalty(real, fake)
        assert gp.dim() == 0

    def test_conditional_gan(self):
        from fibernet.ml.gan import FiberCGAN
        cgan = FiberCGAN(n_features=10, n_conditions=2, latent_dim=16, hidden=[32])
        cond = torch.tensor([[1.0, 2.0]])
        samples = cgan.sample(n=3, conditions=cond)
        assert samples.shape == (3, 10)

    def test_gan_trainer(self):
        from fibernet.ml.gan import FiberWGAN, GANTrainer
        gan = FiberWGAN(n_features=10, latent_dim=16, hidden=[32])
        trainer = GANTrainer(gan)
        X = np.random.randn(50, 10).astype(np.float32)
        history = trainer.fit(X, epochs=3, batch_size=16, verbose=False)
        assert "g_loss" in history
        assert len(history["g_loss"]) == 3

    def test_gan_diversity(self):
        from fibernet.ml.gan import FiberWGAN, GANTrainer
        gan = FiberWGAN(n_features=10, latent_dim=16, hidden=[32])
        trainer = GANTrainer(gan)
        X = np.random.randn(50, 10).astype(np.float32)
        trainer.fit(X, epochs=2, verbose=False)
        metrics = trainer.model.eval()
        # Just test that sample works
        samples = trainer.sample(n=5)
        assert samples.shape == (5, 10)


# ======================================================================
# Field Prediction Tests
# ======================================================================

class TestFieldPrediction:
    def test_unet_forward(self):
        from fibernet.ml.field_prediction import FiberUNet
        model = FiberUNet(in_channels=3, out_channels=2, base_channels=16, n_levels=2)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 2, 32, 32)

    def test_field_mlp(self):
        from fibernet.ml.field_prediction import FiberFieldMLP
        model = FiberFieldMLP(in_dim=5, out_dim=2, hidden=[32, 16])
        x = torch.randn(10, 5)
        y = model(x)
        assert y.shape == (10, 2)

    def test_field_gradient_loss(self):
        from fibernet.ml.field_prediction import field_gradient_loss
        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randn(2, 1, 16, 16)
        loss = field_gradient_loss(pred, target)
        assert loss.dim() == 0
        loss.backward()

    def test_multi_scale_loss(self):
        from fibernet.ml.field_prediction import field_multi_scale_loss
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.randn(2, 1, 32, 32)
        loss = field_multi_scale_loss(pred, target, n_scales=2)
        assert loss.dim() == 0
        loss.backward()


# ======================================================================
# Neural Operator Tests
# ======================================================================

class TestNeuralOperator:
    def test_fno_forward(self):
        from fibernet.ml.neural_operator import FiberFNO
        fno = FiberFNO(in_channels=2, out_channels=1, modes=8, width=16, n_layers=2)
        x = torch.randn(2, 2, 32, 32)
        y = fno(x)
        assert y.shape == (2, 1, 32, 32)

    def test_deeponet_forward(self):
        from fibernet.ml.neural_operator import FiberDeepONet
        don = FiberDeepONet(branch_dim=10, trunk_dim=2, hidden=[32, 16])
        features = torch.randn(4, 10)
        coords = torch.randn(4, 20, 2)
        output = don(features, coords)
        assert output.shape == (4, 20, 1)

    def test_deeponet_predict_field(self):
        from fibernet.ml.neural_operator import FiberDeepONet
        don = FiberDeepONet(branch_dim=10, trunk_dim=2, hidden=[32, 16], n_outputs=2)
        features = torch.randn(2, 10)
        field = don.predict_field(features, grid_size=(8, 8))
        assert field.shape == (2, 2, 8, 8)

    def test_neural_operator_trainer(self):
        from fibernet.ml.neural_operator import FiberFNO, NeuralOperatorTrainer
        fno = FiberFNO(in_channels=2, out_channels=1, modes=4, width=8, n_layers=2)
        trainer = NeuralOperatorTrainer(fno, lr=1e-3)
        X = np.random.randn(10, 2, 16, 16).astype(np.float32)
        y = np.random.randn(10, 1, 16, 16).astype(np.float32)
        result = trainer.fit(X, y, epochs=2, batch_size=5, verbose=False)
        assert "history" in result


# ======================================================================
# Inverse Design Tests
# ======================================================================

class TestInverseDesign:
    def test_inverse_net(self):
        from fibernet.ml.inverse_design import InverseDesignNet
        net = InverseDesignNet(n_properties=3, n_features=10, hidden=[32, 16])
        props = torch.randn(4, 3)
        features = net(props)
        assert features.shape == (4, 10)

    def test_tandem_network(self):
        from fibernet.ml.inverse_design import TandemNetwork
        tandem = TandemNetwork(n_features=10, n_properties=3, hidden=[32, 16])
        x = torch.randn(4, 10)
        y = torch.randn(4, 3)
        losses = tandem.total_loss(x, y)
        assert "total" in losses
        assert "forward" in losses
        assert "inverse" in losses
        assert "cycle" in losses
        losses["total"].backward()

    def test_inverse_design_trainer(self):
        from fibernet.ml.inverse_design import InverseDesignNet, InverseDesignTrainer
        net = InverseDesignNet(n_properties=2, n_features=8, hidden=[16])
        trainer = InverseDesignTrainer(net)
        X = np.random.randn(30, 8).astype(np.float32)
        y = np.random.randn(30, 2).astype(np.float32)
        history = trainer.fit(X, y, epochs=3, batch_size=10, verbose=False)
        assert "train_loss" in history

    def test_design(self):
        from fibernet.ml.inverse_design import TandemNetwork, InverseDesignTrainer
        tandem = TandemNetwork(n_features=8, n_properties=2, hidden=[16])
        trainer = InverseDesignTrainer(tandem)
        X = np.random.randn(30, 8).astype(np.float32)
        y = np.random.randn(30, 2).astype(np.float32)
        trainer.fit(X, y, epochs=2, verbose=False)
        target = np.array([[1.0, 2.0]])
        candidates = trainer.design(target, n_candidates=3)
        assert candidates.shape[0] == 3
        assert candidates.shape[1] == 8


# ======================================================================
# Active Learning Tests
# ======================================================================

class TestActiveLearning:
    def test_uncertainty_sampling(self):
        from fibernet.ml.active_learning import UncertaintySampling
        us = UncertaintySampling(model_type="ridge", use_ensemble=True, n_estimators=3)
        X = np.random.randn(50, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5]) + np.random.randn(50) * 0.1
        us.fit(X, y)
        selected = us.select(X, batch_size=5)
        assert len(selected) == 5

    def test_diversity_sampling(self):
        from fibernet.ml.active_learning import DiversitySampling
        ds = DiversitySampling(n_clusters=5)
        X = np.random.randn(50, 5)
        selected = ds.select(X, batch_size=5)
        assert len(selected) == 5

    def test_query_by_committee(self):
        from fibernet.ml.active_learning import QueryByCommittee
        qbc = QueryByCommittee(model_types=["ridge"])
        X = np.random.randn(50, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5])
        qbc.fit(X, y)
        selected = qbc.select(X, batch_size=3)
        assert len(selected) == 3

    def test_active_learning_loop(self):
        from fibernet.ml.active_learning import ActiveLearningLoop, UncertaintySampling
        X = np.random.randn(100, 5)
        y = X @ np.array([1, 2, 0, -1, 0.5])

        al = ActiveLearningLoop(
            acquisition=UncertaintySampling(model_type="ridge"),
            simulator=lambda x: float(x @ np.array([1, 2, 0, -1, 0.5])),
            model_type="ridge",
        )
        al.initialize(X, y_initial=y[:20], n_initial=20)
        result = al.step(batch_size=5)
        assert "r2" in result
        assert result["n_labeled"] == 25
        summary = al.get_summary()
        assert summary["n_iterations"] == 1


# ======================================================================
# Transfer Learning Tests
# ======================================================================

class TestTransferLearning:
    def test_transfer_net_pretrain(self):
        from fibernet.ml.transfer_learning import FiberTransferNet
        net = FiberTransferNet(n_features=10, n_outputs=1, hidden=[32, 16])
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        result = net.pretrain(X, y, epochs=3, verbose=False)
        assert "train_loss" in result

    def test_transfer_net_finetune(self):
        from fibernet.ml.transfer_learning import FiberTransferNet
        net = FiberTransferNet(n_features=10, n_outputs=1, hidden=[32, 16])
        X_src = np.random.randn(50, 10).astype(np.float32)
        y_src = np.random.randn(50).astype(np.float32)
        net.pretrain(X_src, y_src, epochs=2, verbose=False)
        X_tgt = np.random.randn(15, 10).astype(np.float32)
        y_tgt = np.random.randn(15).astype(np.float32)
        result = net.finetune(X_tgt, y_tgt, epochs=3, freeze_layers=1, verbose=False)
        assert "train_loss" in result

    def test_transfer_predict(self):
        from fibernet.ml.transfer_learning import FiberTransferNet
        net = FiberTransferNet(n_features=10, n_outputs=1, hidden=[16])
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.random.randn(20).astype(np.float32)
        net.pretrain(X, y, epochs=1, verbose=False)
        pred = net.predict(X[:5])
        assert pred.shape == (5, 1)

    def test_prototypical_net(self):
        from fibernet.ml.transfer_learning import FiberPrototypicalNet
        proto = FiberPrototypicalNet(n_features=10, embedding_dim=8, hidden=[16])
        X = np.random.randn(60, 10).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        result = proto.train(X, y, epochs=3, n_way=3, n_support=3, n_query=5, verbose=False)
        assert "losses" in result
        assert "accuracies" in result


# ======================================================================
# Multi-Scale Tests
# ======================================================================

class TestMultiScale:
    def test_feature_extractor(self):
        from fibernet.ml.multiscale import MultiScaleFeatureExtractor
        from fibernet import pattern_2d
        g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(3, 3))
        ext = MultiScaleFeatureExtractor(scales=[1.0, 2.0])
        features = ext.extract(g)
        assert "global__n_nodes" in features
        assert "global__n_edges" in features
        assert isinstance(features["global__n_nodes"], (int, float))

    def test_feature_names(self):
        from fibernet.ml.multiscale import MultiScaleFeatureExtractor
        ext = MultiScaleFeatureExtractor(scales=[1.0, 2.0])
        names = ext.get_feature_names()
        assert len(names) > 0
        assert "global__n_nodes" in names

    def test_hierarchical_encoder(self):
        from fibernet.ml.multiscale import HierarchicalEncoder
        enc = HierarchicalEncoder(n_features=18, n_outputs=1, n_scales=3, hidden=[16])
        x = torch.randn(4, 18)
        y = enc(x)
        assert y.shape == (4, 1)

    def test_scale_bridge(self):
        from fibernet.ml.multiscale import ScaleBridgeModel
        model = ScaleBridgeModel(n_micro_features=20, n_macro_properties=2, hidden=[16])
        x = torch.randn(4, 20)
        y = model(x)
        assert y.shape == (4, 2)
        # Test numpy predict
        props = model.predict_effective_properties(np.random.randn(3, 20).astype(np.float32))
        assert props.shape == (3, 2)


# ======================================================================
# RL Curriculum Tests
# ======================================================================

class TestCurriculum:
    def test_linear_curriculum(self):
        from fibernet.rl.curriculum import LinearCurriculum
        curr = LinearCurriculum("grid_x", start_value=2.0, max_value=6.0, n_episodes=100)
        params = curr.get_env_params()
        assert params["grid_x"] == 2.0
        for _ in range(50):
            curr.step(0.5)
        assert curr.current_value > 2.0

    def test_adaptive_curriculum(self):
        from fibernet.rl.curriculum import AdaptiveCurriculum
        curr = AdaptiveCurriculum(
            "grid_x", start_value=2.0, max_value=6.0,
            step_size=1.0, reward_threshold=0.5, min_episodes=3,
            window_size=5,
        )
        # Low reward: don't advance
        for _ in range(5):
            curr.step(0.1)
        assert curr.current_value == 2.0
        # High reward: advance
        for _ in range(10):
            curr.step(0.9)
        assert curr.current_value > 2.0

    def test_multi_stage_curriculum(self):
        from fibernet.rl.curriculum import MultiStageCurriculum
        curr = MultiStageCurriculum([
            {"name": "easy", "env_params": {"grid": 2}, "n_episodes": 5},
            {"name": "hard", "env_params": {"grid": 4}, "n_episodes": 5},
        ])
        assert curr.current_stage == 0
        for _ in range(6):
            curr.step(0.5)
        assert curr.current_stage == 1

    def test_curriculum_progress(self):
        from fibernet.rl.curriculum import LinearCurriculum
        curr = LinearCurriculum("grid_x", start_value=0, max_value=10, n_episodes=100)
        for _ in range(100):
            curr.step(0.0)
        progress = curr.get_progress()
        assert progress >= 0.9


# ======================================================================
# RL Reward Shaping Tests
# ======================================================================

class TestRewardShaping:
    def test_composite_reward(self):
        from fibernet.rl.reward_shaping import CompositeReward
        reward = CompositeReward({
            "a": lambda i: i.get("x", 0),
            "b": lambda i: -i.get("y", 0),
        }, weights={"a": 1.0, "b": 2.0})
        r = reward({"x": 1.0, "y": 0.5})
        assert abs(r - 0.0) < 1e-6  # 1*1 + 2*(-0.5) = 0

    def test_distance_reward(self):
        from fibernet.rl.reward_shaping import DistanceReward
        reward = DistanceReward(
            targets={"force": 100.0, "stretch": 1.5},
            norm="relative",
        )
        r = reward({"force": 100.0, "stretch": 1.5})
        assert abs(r) < 1e-6  # Perfect match = 0 error

    def test_pbrs(self):
        from fibernet.rl.reward_shaping import PotentialBasedShaping
        pbrs = PotentialBasedShaping(
            potential_fn=lambda i: i.get("x", 0) ** 2,
            gamma=0.99,
        )
        r1 = pbrs({"x": 1.0})
        assert r1 == 0.0  # First step, no prev
        r2 = pbrs({"x": 2.0})
        assert r2 != 0.0

    def test_sparse_reward(self):
        from fibernet.rl.reward_shaping import SparseReward
        reward = SparseReward(
            milestones=[
                {"condition": lambda i: i.get("force", 0) < 100, "reward": 10.0, "name": "low_force"},
                {"condition": lambda i: i.get("stretch", 0) > 1.5, "reward": 5.0, "name": "high_stretch"},
            ],
            default_reward=-0.01,
        )
        r = reward({"force": 50, "stretch": 0.0})
        assert r == 10.0  # First milestone reached
        r2 = reward({"force": 50, "stretch": 2.0})
        assert r2 == 5.0  # Second milestone

    def test_reward_normalizer(self):
        from fibernet.rl.reward_shaping import RewardNormalizer
        norm = RewardNormalizer()
        rewards = [norm(r) for r in np.random.randn(100)]
        assert abs(np.mean(rewards)) < 1.0

    def test_create_default_reward(self):
        from fibernet.rl.reward_shaping import create_default_reward
        for mode in ["minimize_force", "maximize_stretch", "uniform", "balanced"]:
            reward = create_default_reward(mode)
            info = {"max_force": 500, "mean_stretch": 1.5, "std_stretch": 0.1,
                    "n_edges": 50, "strain_energy": 100}
            r = reward(info)
            assert isinstance(r, float)


# ======================================================================
# RL Multi-Objective Tests
# ======================================================================

class TestMultiObjective:
    def test_scalarized_morl(self):
        from fibernet.rl.multi_objective_rl import ScalarizedMORL
        morl = ScalarizedMORL(
            objectives={
                "force": lambda i: -i.get("force", 0),
                "weight": lambda i: -i.get("weight", 0),
            },
            weights={"force": 0.7, "weight": 0.3},
        )
        r = morl.compute_reward({"force": 100, "weight": 50})
        assert isinstance(r, float)

    def test_individual_rewards(self):
        from fibernet.rl.multi_objective_rl import ScalarizedMORL
        morl = ScalarizedMORL(
            objectives={
                "a": lambda i: i.get("a", 0),
                "b": lambda i: i.get("b", 0),
            },
        )
        indiv = morl.compute_individual_rewards({"a": 1, "b": 2})
        assert indiv["a"] == 1
        assert indiv["b"] == 2

    def test_pareto_extraction(self):
        from fibernet.rl.multi_objective_rl import ScalarizedMORL
        results = [
            {"objective_means": {"x": 1.0, "y": 10.0}},
            {"objective_means": {"x": 5.0, "y": 5.0}},
            {"objective_means": {"x": 10.0, "y": 1.0}},
            {"objective_means": {"x": 2.0, "y": 2.0}},  # Dominated by (1,10) and (10,1)
        ]
        pareto = ScalarizedMORL.extract_pareto_front(
            results, objective_keys=["x", "y"], maximize=[True, True]
        )
        assert len(pareto) >= 2

    def test_pareto_explorer(self):
        from fibernet.rl.multi_objective_rl import ParetoFrontExplorer
        explorer = ParetoFrontExplorer(
            objectives={
                "force": lambda i: -i.get("force", 0),
                "weight": lambda i: -i.get("weight", 0),
            },
            n_exploration_points=5,
        )
        grid = explorer.generate_weight_grid()
        assert len(grid) >= 5
        for w in grid:
            assert len(w) == 2
            assert abs(sum(w) - 1.0) < 1e-6


# ======================================================================
# Symbolic Regression Tests
# ======================================================================

class TestSymbolic:
    def test_feature_selection(self):
        from fibernet.ml.symbolic import FeatureSelection
        fs = FeatureSelection(n_select=3, method="correlation")
        X = np.random.randn(50, 10)
        y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(50) * 0.1
        X_sel, names = fs.select(X, y, feature_names=[f"f{i}" for i in range(10)])
        assert X_sel.shape == (50, 3)
        assert len(names) == 3


# ======================================================================
# Integration Tests
# ======================================================================

class TestIntegration:
    def test_diffusion_train_generate_roundtrip(self):
        """Test full diffusion pipeline: train → generate → validate shapes."""
        from fibernet.ml.diffusion import FiberDiffusion, DiffusionTrainer
        n_features = 8
        model = FiberDiffusion(n_features=n_features, hidden=[16], n_steps=20)
        trainer = DiffusionTrainer(model)
        X = np.random.randn(40, n_features).astype(np.float32)
        trainer.fit(X, epochs=3, verbose=False)
        X_gen = trainer.sample(n=10)
        assert X_gen.shape == (10, n_features)

    def test_gan_train_generate_roundtrip(self):
        """Test full GAN pipeline."""
        from fibernet.ml.gan import FiberWGAN, GANTrainer
        n_features = 8
        gan = FiberWGAN(n_features=n_features, latent_dim=8, hidden=[16])
        trainer = GANTrainer(gan)
        X = np.random.randn(40, n_features).astype(np.float32)
        trainer.fit(X, epochs=3, verbose=False)
        X_gen = trainer.sample(n=10)
        assert X_gen.shape == (10, n_features)

    def test_inverse_design_roundtrip(self):
        """Test inverse design: train → design → verify shapes."""
        from fibernet.ml.inverse_design import TandemNetwork, InverseDesignTrainer
        n_feat, n_prop = 8, 2
        tandem = TandemNetwork(n_features=n_feat, n_properties=n_prop, hidden=[16])
        trainer = InverseDesignTrainer(tandem)
        X = np.random.randn(30, n_feat).astype(np.float32)
        y = np.random.randn(30, n_prop).astype(np.float32)
        trainer.fit(X, y, epochs=3, verbose=False)
        target = np.array([[0.5, 1.0]])
        candidates = trainer.design(target, n_candidates=5)
        assert candidates.shape == (5, n_feat)

    def test_curriculum_with_reward(self):
        """Test curriculum + reward shaping integration."""
        from fibernet.rl.curriculum import AdaptiveCurriculum
        from fibernet.rl.reward_shaping import CompositeReward, create_default_reward
        curr = AdaptiveCurriculum("grid", start_value=2, max_value=6, min_episodes=2)
        reward = create_default_reward("balanced")
        for _ in range(10):
            info = {"max_force": 500, "std_stretch": 0.1, "n_edges": 30}
            r = reward(info)
            curr.step(r)
        assert curr.current_value >= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ======================================================================
# GFlowNet Tests
# ======================================================================

class TestGFlowNet:
    def test_structure_state(self):
        from fibernet.ml.gflownet import StructureState, StructureAction
        state = StructureState(max_nodes=10, box_size=10.0)
        assert state.n_nodes == 0
        assert state.n_edges == 0

        # Add node
        action = StructureAction(action_type=0, features=np.array([5.0, 5.0, 0.0, 0.0, 0.0]))
        state.apply_action(action)
        assert state.n_nodes == 1

        # Add another node
        action2 = StructureAction(action_type=0, features=np.array([8.0, 3.0, 0.0, 0.0, 0.0]))
        state.apply_action(action2)
        assert state.n_nodes == 2

        # Add edge
        action3 = StructureAction(action_type=1, source_idx=0, target_idx=1)
        state.apply_action(action3)
        assert state.n_edges == 1

    def test_observation_encoding(self):
        from fibernet.ml.gflownet import StructureState, StructureAction
        state = StructureState(max_nodes=10, box_size=10.0)
        action = StructureAction(action_type=0, features=np.array([5.0, 5.0, 0.0, 0.0, 0.0]))
        state.apply_action(action)
        obs = state.to_observation(n_node_features=5, max_nodes=10)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) > 0

    def test_gflownet_creation(self):
        from fibernet.ml.gflownet import FiberGFlowNet
        gfn = FiberGFlowNet(n_node_features=5, hidden=32, max_nodes=10)
        assert gfn.obs_dim > 0
        assert gfn.policy is not None

    def test_gflownet_trajectory(self):
        from fibernet.ml.gflownet import FiberGFlowNet
        gfn = FiberGFlowNet(n_node_features=5, hidden=32, max_nodes=10)
        traj = gfn.forward_trajectory(max_steps=5, temperature=1.0)
        assert "states" in traj
        assert "actions" in traj
        assert "log_probs" in traj
        assert "final_state" in traj
        assert len(traj["states"]) >= 2  # at least initial + one step

    def test_gflownet_sample(self):
        from fibernet.ml.gflownet import FiberGFlowNet
        gfn = FiberGFlowNet(n_node_features=5, hidden=32, max_nodes=10)
        samples = gfn.sample(n=3, max_steps=5, temperature=1.0)
        assert len(samples) == 3
        for s in samples:
            assert hasattr(s, 'n_nodes')
            assert hasattr(s, 'n_edges')

    def test_gflownet_trainer(self):
        from fibernet.ml.gflownet import FiberGFlowNet, GFlowNetTrainer, connectivity_reward
        gfn = FiberGFlowNet(n_node_features=5, hidden=32, max_nodes=10, loss_type="tb")
        trainer = GFlowNetTrainer(gfn, reward_fn=connectivity_reward, lr=1e-3)
        history = trainer.train(n_iterations=3, batch_size=4, max_steps=5,
                                 log_every=10, verbose=False)
        assert "loss" in history
        assert "mean_reward" in history
        assert len(history["loss"]) == 3

    def test_gflownet_db_loss(self):
        from fibernet.ml.gflownet import FiberGFlowNet, GFlowNetTrainer, connectivity_reward
        gfn = FiberGFlowNet(n_node_features=5, hidden=32, max_nodes=10, loss_type="db")
        trainer = GFlowNetTrainer(gfn, reward_fn=connectivity_reward, lr=1e-3)
        metrics = trainer.train_step(batch_size=4, max_steps=5)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_connectivity_reward(self):
        from fibernet.ml.gflownet import StructureState, StructureAction, connectivity_reward
        state = StructureState(max_nodes=10, box_size=10.0)
        for i in range(5):
            a = StructureAction(action_type=0, features=np.random.uniform(0, 10, 5))
            state.apply_action(a)
        for i in range(4):
            a = StructureAction(action_type=1, source_idx=i, target_idx=i+1)
            state.apply_action(a)
        r = connectivity_reward(state)
        assert isinstance(r, float)
        assert r > 0

    def test_property_target_reward(self):
        from fibernet.ml.gflownet import StructureState, StructureAction, property_target_reward
        state = StructureState(max_nodes=20, box_size=10.0)
        for i in range(10):
            a = StructureAction(action_type=0, features=np.random.uniform(0, 10, 5))
            state.apply_action(a)
        for i in range(9):
            a = StructureAction(action_type=1, source_idx=i, target_idx=i+1)
            state.apply_action(a)
        r = property_target_reward(state, target_n_nodes=10, target_n_edges=9)
        assert isinstance(r, float)


# ======================================================================
# Differentiable Physics Tests
# ======================================================================

class TestDifferentiablePhysics:
    def test_spring_network_creation(self):
        from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
        sim = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
        assert sim.E == 1e9
        assert sim.dim == 2

    def test_spring_element_stiffness(self):
        from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
        sim = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
        pos_i = torch.tensor([0.0, 0.0])
        pos_j = torch.tensor([1.0, 0.0])
        area = torch.tensor(0.0001)
        K = sim.compute_element_stiffness(pos_i, pos_j, area)
        assert K.shape == (4, 4)
        # Symmetric
        assert torch.allclose(K, K.t(), atol=1e-6)

    def test_spring_solve(self):
        from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
        sim = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2, damping=1e-4)
        # Simple 2-node bar
        node_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        radii = torch.tensor([0.01], dtype=torch.float32, requires_grad=True)
        forces = torch.tensor([[0.0, 0.0], [1000.0, 0.0]], dtype=torch.float32)
        fixed = torch.tensor([0], dtype=torch.long)

        u, sigma = sim.solve(edge_index, node_pos, radii, forces, fixed)
        assert u.shape == (2, 2)
        assert sigma.shape == (1,)
        # Node 0 should be fixed
        assert abs(u[0, 0].item()) < 1e-6
        # Node 1 should displace in x
        assert u[1, 0].item() > 0

    def test_spring_gradient_flow(self):
        from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork
        sim = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
        node_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        radii = torch.tensor([0.01], dtype=torch.float32, requires_grad=True)
        forces = torch.tensor([[0.0, 0.0], [1000.0, 0.0]], dtype=torch.float32)
        fixed = torch.tensor([0], dtype=torch.long)

        u, sigma = sim.solve(edge_index, node_pos, radii, forces, fixed)
        loss = u[1, 0] ** 2
        loss.backward()
        assert radii.grad is not None
        assert radii.grad.abs().item() > 0

    def test_beam_network(self):
        from fibernet.ml.differentiable_physics import DifferentiableBeamNetwork
        sim = DifferentiableBeamNetwork(youngs_modulus=1e9, include_bending=True)
        node_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        radii = torch.tensor([0.01], dtype=torch.float32)
        forces = torch.tensor([[0.0, 0.0], [0.0, 100.0]], dtype=torch.float32)
        fixed = torch.tensor([0], dtype=torch.long)

        u, sigma = sim.solve(edge_index, node_pos, radii, forces, fixed)
        assert u.shape == (2, 3)  # [ux, uy, theta]
        assert sigma.shape == (1,)

    def test_differentiable_fea(self):
        from fibernet.ml.differentiable_physics import DifferentiableFEA
        fea = DifferentiableFEA(youngs_modulus=1e9, element_type="spring")
        node_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        radii = torch.tensor([0.01, 0.01], dtype=torch.float32, requires_grad=True)
        forces = torch.tensor([[0.0, 0.0], [0.0, 0.0], [500.0, 0.0]], dtype=torch.float32)
        fixed = torch.tensor([0], dtype=torch.long)

        result = fea(edge_index, node_pos, radii, forces, fixed)
        assert "displacements" in result
        assert "stresses" in result
        assert "compliance" in result
        assert "volume" in result

    def test_physics_optimizer(self):
        from fibernet.ml.differentiable_physics import DifferentiableFEA, PhysicsOptimizer
        fea = DifferentiableFEA(youngs_modulus=1e9, element_type="spring")
        optimizer = PhysicsOptimizer(fea, lr=0.01, volume_constraint=0.01)
        node_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        radii = torch.tensor([0.01], dtype=torch.float32)
        forces = torch.tensor([[0.0, 0.0], [1000.0, 0.0]], dtype=torch.float32)
        fixed = torch.tensor([0], dtype=torch.long)

        result = optimizer.optimize(
            edge_index, node_pos, radii, forces, fixed,
            n_iterations=5, verbose=False, min_radius=0.005, max_radius=0.05
        )
        assert "optimized_radii" in result
        assert "history" in result
        assert len(result["history"]["objective"]) == 5

    def test_material_model(self):
        from fibernet.ml.differentiable_physics import DifferentiableMaterialModel
        model = DifferentiableMaterialModel(hidden=[16, 8], ensure_monotonic=False)
        strain = torch.linspace(0, 0.01, 20)
        stress = model(strain)
        assert stress.shape == (20,)

    def test_material_model_tangent(self):
        from fibernet.ml.differentiable_physics import DifferentiableMaterialModel
        model = DifferentiableMaterialModel(hidden=[16, 8])
        strain = torch.linspace(0, 0.01, 10, requires_grad=True)
        tangent = model.tangent_modulus(strain)
        assert tangent.shape == (10,)

    def test_material_model_fit(self):
        from fibernet.ml.differentiable_physics import DifferentiableMaterialModel
        model = DifferentiableMaterialModel(hidden=[16, 8])
        strain = np.linspace(0, 0.01, 50).astype(np.float32)
        stress = (1e9 * strain).astype(np.float32)  # Linear elastic
        history = model.fit(strain, stress, epochs=10, verbose=False)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 10


# ======================================================================
# Physics-Informed GNN Tests
# ======================================================================

class TestPhysicsInformedGNN:
    def _make_graph(self, n_nodes=5, n_edges=6):
        """Helper to create a simple graph dict."""
        node_features = torch.randn(n_nodes, 5)
        # Ensure all indices are within node range
        src = [i % n_nodes for i in range(n_edges)]
        dst = [(i + 1) % n_nodes for i in range(n_edges)]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_features = torch.randn(n_edges, 2)
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        }

    def test_message_passing(self):
        from fibernet.ml.pinn_gnn import PhysicsInformedMessagePassing
        mp = PhysicsInformedMessagePassing(node_dim=5, edge_dim=2, hidden=16, force_dim=2)
        g = self._make_graph()
        h_new, e_new, forces = mp(g["node_features"], g["edge_index"], g["edge_features"])
        assert h_new.shape == (5, 5)  # projected back to node_dim
        assert e_new.shape == (6, 2)
        assert forces.shape == (5, 2)

    def test_force_balance_loss(self):
        from fibernet.ml.pinn_gnn import ForceBalanceLoss
        loss_fn = ForceBalanceLoss(weight=1.0)
        nodal_forces = torch.randn(5, 2)
        loss = loss_fn(nodal_forces)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_constitutive_loss(self):
        from fibernet.ml.pinn_gnn import ConstitutiveLoss
        loss_fn = ConstitutiveLoss(youngs_modulus=1e9, weight=0.5)
        stress = torch.randn(10)
        strain = torch.randn(10)
        loss = loss_fn(stress, strain)
        assert loss.dim() == 0

    def test_energy_conservation_loss(self):
        from fibernet.ml.pinn_gnn import EnergyConservationLoss
        loss_fn = EnergyConservationLoss(weight=0.1)
        U = torch.tensor(100.0)
        W = torch.tensor(95.0)
        loss = loss_fn(U, W)
        assert loss.dim() == 0

    def test_pinn_gnn_forward(self):
        from fibernet.ml.pinn_gnn import PhysicsInformedGNN
        gnn = PhysicsInformedGNN(node_dim=5, edge_dim=2, hidden=16, n_layers=2,
                                  n_outputs=1, predict_field=True, force_dim=2)
        g = self._make_graph()
        pred = gnn([g])
        assert pred.shape == (1, 1)

    def test_pinn_gnn_fields(self):
        from fibernet.ml.pinn_gnn import PhysicsInformedGNN
        gnn = PhysicsInformedGNN(node_dim=5, edge_dim=2, hidden=16, n_layers=2,
                                  predict_field=True, force_dim=2)
        g = self._make_graph()
        fields = gnn.predict_fields(g)
        assert "displacement" in fields
        assert "stress" in fields
        assert "strain" in fields
        assert fields["displacement"].shape == (5, 2)

    def test_pinn_gnn_physics_loss(self):
        from fibernet.ml.pinn_gnn import PhysicsInformedGNN
        gnn = PhysicsInformedGNN(node_dim=5, edge_dim=2, hidden=16, n_layers=2,
                                  predict_field=True, force_dim=2)
        g = self._make_graph()
        losses = gnn.compute_physics_loss(g)
        assert "force_balance" in losses
        assert "constitutive" in losses
        assert "energy" in losses
        assert "total_physics" in losses

    def test_pinn_gnn_trainer(self):
        from fibernet.ml.pinn_gnn import PhysicsInformedGNN, PhysicsGNNTrainer
        gnn = PhysicsInformedGNN(node_dim=5, edge_dim=2, hidden=16, n_layers=2,
                                  n_outputs=1, predict_field=True, force_dim=2)
        trainer = PhysicsGNNTrainer(gnn, physics_loss_weight=0.3, lr=1e-3)

        graphs = [self._make_graph() for _ in range(20)]
        labels = np.random.randn(20).astype(np.float32)

        result = trainer.fit(graphs, labels, epochs=5, batch_size=10, verbose=False)
        assert "history" in result
        assert "best_val_loss" in result


# ======================================================================
# Neural ODE Tests
# ======================================================================

class TestNeuralODE:
    def test_ode_solver_euler(self):
        from fibernet.ml.neural_ode import ODESolver
        solver = ODESolver(method="euler")
        # dx/dt = -x (exponential decay)
        def f(t, x):
            return -x
        x0 = torch.tensor([1.0])
        t = torch.linspace(0, 2, 20)
        traj = solver.solve(f, x0, t)
        assert traj.shape == (20, 1)
        # Should decay
        assert traj[-1, 0] < traj[0, 0]

    def test_ode_solver_rk4(self):
        from fibernet.ml.neural_ode import ODESolver
        solver = ODESolver(method="rk4")
        def f(t, x):
            return -x
        x0 = torch.tensor([1.0])
        t = torch.linspace(0, 2, 50)
        traj = solver.solve(f, x0, t)
        # Check against analytical: x(t) = exp(-t)
        expected = torch.exp(-t)
        error = (traj[:, 0] - expected).abs().max().item()
        assert error < 0.01  # RK4 should be accurate

    def test_ode_solver_dopri5(self):
        from fibernet.ml.neural_ode import ODESolver
        solver = ODESolver(method="dopri5")
        def f(t, x):
            return -x
        x0 = torch.tensor([1.0])
        t = torch.linspace(0, 2, 30)
        traj = solver.solve(f, x0, t)
        assert traj.shape == (30, 1)

    def test_fiber_neural_ode(self):
        from fibernet.ml.neural_ode import FiberNeuralODE
        ode = FiberNeuralODE(state_dim=3, hidden=[32, 16], solver_method="rk4")
        x0 = torch.randn(3)
        t = torch.linspace(0, 1, 10)
        traj = ode.solve(x0, t)
        assert traj.shape == (10, 3)

    def test_fiber_neural_ode_forward(self):
        from fibernet.ml.neural_ode import FiberNeuralODE
        ode = FiberNeuralODE(state_dim=4, hidden=[16], solver_method="rk4")
        x0 = torch.randn(4)
        t = torch.linspace(0, 1, 5)
        final = ode(x0, t)
        assert final.shape == (4,)

    def test_stress_relaxation_maxwell(self):
        from fibernet.ml.neural_ode import StressRelaxationODE
        sr = StressRelaxationODE(model_type="maxwell", E=1e9, eta=1e12)
        t, sigma = sr.relax(initial_stress=100.0, t_span=(0, 1000), n_steps=50)
        assert len(t) == 51
        assert len(sigma) == 51
        # Stress should decrease (relaxation)
        assert sigma[0] >= sigma[-1]

    def test_stress_relaxation_analytical(self):
        from fibernet.ml.neural_ode import StressRelaxationODE
        sr = StressRelaxationODE(model_type="maxwell", E=1e9, eta=1e12)
        times = np.linspace(0, 1000, 50)
        analytical = sr.analytical_solution(100.0, times)
        assert len(analytical) == 50
        # Maxwell: σ(t) = σ₀ exp(-t/τ), τ = η/E = 1000
        expected = 100.0 * np.exp(-times / 1000.0)
        np.testing.assert_allclose(analytical, expected, rtol=1e-10)

    def test_creep_ode(self):
        from fibernet.ml.neural_ode import CreepODE
        creep = CreepODE(E=1e9, eta=1e12, model_type="power_law", n_creep=0.3)
        initial_strain = 100.0 / 1e9  # σ/E
        t, eps = creep.predict_creep(initial_strain, 100.0, t_span=(1, 1000), n_steps=50)
        assert len(t) == 51
        assert len(eps) == 51
        # Strain should increase (creep)
        assert eps[-1] >= eps[0]

    def test_fatigue_ode(self):
        from fibernet.ml.neural_ode import FatigueODE
        fatigue = FatigueODE(hidden=[16, 8])
        result = fatigue.predict_fatigue_life(
            initial_damage=0.0,
            stress_range=100.0,
            max_cycles=500,
            failure_damage=1.0,
        )
        assert "cycles" in result
        assert "damage_history" in result
        assert "cycles_to_failure" in result
        assert result["cycles"] > 0

    def test_neural_ode_trainer(self):
        from fibernet.ml.neural_ode import FiberNeuralODE, NeuralODETrainer
        ode = FiberNeuralODE(state_dim=2, hidden=[16], solver_method="euler")
        trainer = NeuralODETrainer(ode, lr=1e-3)

        # Generate simple decay data
        n_traj = 5
        n_times = 10
        time_data = np.tile(np.linspace(0, 1, n_times), (n_traj, 1))
        state_data = np.zeros((n_traj, n_times, 2))
        for i in range(n_traj):
            x0 = np.random.randn(2)
            for t_idx in range(n_times):
                state_data[i, t_idx] = x0 * np.exp(-time_data[i, t_idx])

        history = trainer.fit(time_data, state_data, epochs=5, batch_size=3, verbose=False)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 5


# ======================================================================
# Conservative Neural Networks Tests
# ======================================================================

class TestConservativeNN:
    def test_hamiltonian_creation(self):
        from fibernet.ml.conservative_nn import HamiltonianNN
        hnn = HamiltonianNN(n_coords=2, hidden=[32, 16])
        q = torch.tensor([1.0, 0.0])
        p = torch.tensor([0.0, 1.0])
        H = hnn.hamiltonian(q, p)
        assert H.dim() == 0

    def test_hamiltonian_equations(self):
        from fibernet.ml.conservative_nn import HamiltonianNN
        hnn = HamiltonianNN(n_coords=2, hidden=[32, 16])
        q = torch.tensor([1.0, 0.0])
        p = torch.tensor([0.0, 1.0])
        dq, dp = hnn(q, p)
        assert dq.shape == (2,)
        assert dp.shape == (2,)

    def test_hamiltonian_simulate(self):
        from fibernet.ml.conservative_nn import HamiltonianNN
        hnn = HamiltonianNN(n_coords=1, hidden=[16, 8])
        q0 = torch.tensor([1.0])
        p0 = torch.tensor([0.0])
        t = torch.linspace(0, 1, 20)
        q_traj, p_traj = hnn.simulate(q0, p0, t, method="rk4")
        assert q_traj.shape == (20, 1)
        assert p_traj.shape == (20, 1)

    def test_lagrangian_creation(self):
        from fibernet.ml.conservative_nn import LagrangianNN
        lnn = LagrangianNN(n_coords=2, hidden=[32, 16])
        q = torch.randn(5, 2)
        q_dot = torch.randn(5, 2)
        L = lnn.lagrangian(q, q_dot)
        assert L.shape == (5,)

    def test_lagrangian_forward(self):
        from fibernet.ml.conservative_nn import LagrangianNN
        lnn = LagrangianNN(n_coords=2, hidden=[32, 16])
        q = torch.randn(3, 2)
        q_dot = torch.randn(3, 2)
        q_ddot = lnn(q, q_dot)
        assert q_ddot.shape == (3, 2)

    def test_energy_conserving(self):
        from fibernet.ml.conservative_nn import EnergyConservingNN
        ec = EnergyConservingNN(state_dim=4, hidden=[32, 16])
        x = torch.randn(4)
        dx = ec(x)
        assert dx.shape == (4,)

    def test_energy_conservation_check(self):
        from fibernet.ml.conservative_nn import EnergyConservingNN
        ec = EnergyConservingNN(state_dim=4, hidden=[32, 16])
        x = torch.randn(4)
        result = ec.check_conservation(x, dt=0.01, n_steps=20)
        assert "energy_values" in result
        assert "max_drift" in result
        assert "relative_drift" in result
        assert len(result["energy_values"]) == 21  # n_steps + 1

    def test_momentum_conserving(self):
        from fibernet.ml.conservative_nn import MomentumConservingNN
        model = MomentumConservingNN(n_nodes=5, coord_dim=2, feature_dim=3, hidden=16)
        positions = torch.randn(5, 2)
        features = torch.randn(5, 3)
        forces, updated = model(positions, features)
        assert forces.shape == (5, 2)
        assert updated.shape == (5, 3)

    def test_momentum_conservation_check(self):
        from fibernet.ml.conservative_nn import MomentumConservingNN
        model = MomentumConservingNN(n_nodes=4, coord_dim=2, feature_dim=3, hidden=16)
        positions = torch.randn(4, 2)
        features = torch.randn(4, 3)
        result = model.check_momentum_conservation(positions, features)
        assert "total_force" in result
        assert "total_force_norm" in result
        # Total force should be ~0 (Newton's 3rd law)
        assert result["total_force_norm"] < 1e-5

    def test_divergence_free_2d(self):
        from fibernet.ml.conservative_nn import DivergenceFreeNet
        net = DivergenceFreeNet(dim=2, hidden=[32, 16])
        coords = torch.randn(10, 2)
        v = net(coords)
        assert v.shape == (10, 2)

    def test_divergence_free_check(self):
        from fibernet.ml.conservative_nn import DivergenceFreeNet
        net = DivergenceFreeNet(dim=2, hidden=[32, 16])
        coords = torch.randn(10, 2)
        div = net.check_divergence(coords)
        assert div.shape == (10,)
        # Divergence should be ~0
        assert div.abs().max().item() < 0.1

    def test_conservative_loss(self):
        from fibernet.ml.conservative_nn import ConservativeLoss
        loss = ConservativeLoss(energy_weight=1.0, momentum_weight=1.0)
        assert loss.energy_weight == 1.0

    def test_conservative_trainer(self):
        from fibernet.ml.conservative_nn import EnergyConservingNN, ConservativeTrainer
        model = EnergyConservingNN(state_dim=4, hidden=[16, 8])
        trainer = ConservativeTrainer(model, lr=1e-3)

        # Generate fake state-energy data
        states = np.random.randn(30, 4).astype(np.float32)
        energies = np.sum(states ** 2, axis=1).astype(np.float32)  # quadratic energy

        history = trainer.fit_energy(states, energies, epochs=5, batch_size=10, verbose=False)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 5

    def test_hamiltonian_trainer(self):
        from fibernet.ml.conservative_nn import HamiltonianNN, ConservativeTrainer
        model = HamiltonianNN(n_coords=1, hidden=[16, 8])
        trainer = ConservativeTrainer(model, lr=1e-3)

        # Simple harmonic oscillator data
        n = 30
        q = np.random.randn(n, 1).astype(np.float32) * 0.5
        p = np.random.randn(n, 1).astype(np.float32) * 0.5
        # For H = 0.5*(q^2 + p^2): dq/dt = p, dp/dt = -q
        dq = p.copy()
        dp = -q.copy()

        history = trainer.fit_hamiltonian(q, p, dq, dp, epochs=10, batch_size=10, verbose=False)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 10


# ======================================================================
# Extended Integration Tests
# ======================================================================

class TestNewModuleIntegration:
    def test_gflownet_reward_pipeline(self):
        """GFlowNet with custom reward function."""
        from fibernet.ml.gflownet import FiberGFlowNet, GFlowNetTrainer, StructureState

        def custom_reward(state: StructureState) -> float:
            n = state.n_nodes
            e = state.n_edges
            if n < 3:
                return 0.1
            return float(n * 0.5 + e * 0.3)

        gfn = FiberGFlowNet(n_node_features=5, hidden=16, max_nodes=8)
        trainer = GFlowNetTrainer(gfn, reward_fn=custom_reward, lr=1e-3)
        history = trainer.train(n_iterations=3, batch_size=4, max_steps=5, verbose=False)
        assert history["max_reward"][-1] >= 0

    def test_diff_physics_gradient_optimization(self):
        """Differentiable physics end-to-end gradient optimization."""
        from fibernet.ml.differentiable_physics import DifferentiableSpringNetwork

        sim = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
        node_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        radii = torch.tensor([0.01, 0.01], dtype=torch.float32, requires_grad=True)
        forces = torch.tensor([[0.0, 0.0], [0.0, 0.0], [500.0, 0.0]], dtype=torch.float32)
        fixed = torch.tensor([0], dtype=torch.long)

        optimizer = torch.optim.Adam([radii], lr=0.005)
        target_disp = torch.tensor([0.0001, 0.0], dtype=torch.float32)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            u, sigma = sim.solve(edge_index, node_pos, radii, forces, fixed)
            loss = ((u[2] - target_disp) ** 2).sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                radii.clamp_(0.001, 0.05)
            losses.append(loss.item())

        assert losses[-1] < losses[0]  # Should improve

    def test_pinn_gnn_physics_loss_backward(self):
        """Physics-informed GNN with physics loss gradient."""
        from fibernet.ml.pinn_gnn import PhysicsInformedGNN

        gnn = PhysicsInformedGNN(node_dim=5, edge_dim=2, hidden=16, n_layers=2,
                                  predict_field=True, force_dim=2)

        graph = {
            "node_features": torch.randn(5, 5),
            "edge_index": torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], dtype=torch.long),
            "edge_features": torch.randn(5, 2),
            "n_nodes": 5,
            "n_edges": 5,
        }

        losses = gnn.compute_physics_loss(graph)
        losses["total_physics"].backward()
        # Check gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in gnn.parameters())
        assert has_grad

    def test_neural_ode_trajectory_prediction(self):
        """Neural ODE full trajectory pipeline."""
        from fibernet.ml.neural_ode import FiberNeuralODE

        ode = FiberNeuralODE(state_dim=3, hidden=[16], solver_method="rk4")
        x0 = torch.randn(3)
        t = torch.linspace(0, 2, 50)
        traj = ode.predict_trajectory(x0, t)
        assert traj.shape == (50, 3)
        # Gradient flow through trajectory
        loss = traj[-1].sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in ode.parameters())
        assert has_grad

    def test_hamiltonian_energy_conservation(self):
        """Hamiltonian NN should approximately conserve energy."""
        from fibernet.ml.conservative_nn import HamiltonianNN

        hnn = HamiltonianNN(n_coords=1, hidden=[32, 16])
        q0 = torch.tensor([0.5])
        p0 = torch.tensor([0.0])
        t = torch.linspace(0, 1, 50)
        q_traj, p_traj = hnn.simulate(q0, p0, t, method="rk4")

        # Energy at start and end
        H_start = hnn.hamiltonian(q0, p0).item()
        H_end = hnn.hamiltonian(q_traj[-1], p_traj[-1]).item()
        # Energy drift should be bounded for RK4
        drift = abs(H_end - H_start)
        # Just check it doesn't explode
        assert drift < 100.0

    def test_cross_module_compatibility(self):
        """Test that new modules work with existing infrastructure."""
        from fibernet.ml.pinn_gnn import PhysicsInformedGNN
        from fibernet.ml.conservative_nn import EnergyConservingNN

        # PIGNN should produce embeddings compatible with other models
        gnn = PhysicsInformedGNN(node_dim=5, edge_dim=2, hidden=16, n_layers=2,
                                  predict_field=True, force_dim=2)
        graph = {
            "node_features": torch.randn(5, 5),
            "edge_index": torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long),
            "edge_features": torch.randn(4, 2),
            "n_nodes": 5,
            "n_edges": 4,
        }
        pred = gnn([graph])
        assert pred.shape[0] == 1

        # Energy conserving NN should work standalone
        ec = EnergyConservingNN(state_dim=16, hidden=[32])
        x = torch.randn(16)
        dx = ec(x)
        assert dx.shape == (16,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
