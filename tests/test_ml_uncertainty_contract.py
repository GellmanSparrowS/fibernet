"""State-safe CPU MC-dropout inference contracts."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fibernet.ml import models


@pytest.fixture(autouse=True)
def tiny_cpu_seed():
    rng_state = torch.get_rng_state()
    torch.manual_seed(1827)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng_state)


def make_model(mode="eval", dtype=torch.float32):
    model = models.FiberUncertaintyNet(3, 2, hidden=[4, 4], dropout=0.4).to(dtype=dtype)
    if mode == "eval":
        model.eval()
    elif mode == "mixed":
        model.train()
        model.backbone[1].eval()
        model.backbone[3].eval()
        model.mu_head.eval()
    return model


def modes(model):
    return {name: module.training for name, module in model.named_modules()}


def buffers(model):
    return {name: value.clone() for name, value in model.named_buffers()}


def assert_buffers_equal(model, expected):
    for name, value in model.named_buffers():
        assert torch.equal(value, expected[name]), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_row_returns_finite_mean_std_matching_input(dtype):
    model = make_model(dtype=dtype)
    x = torch.randn(1, 3, dtype=dtype)
    mean, std = model.predict_with_uncertainty(x, n_mc=3)
    assert mean.shape == std.shape == (1, 2)
    assert mean.dtype == std.dtype == x.dtype
    assert mean.device == std.device == x.device
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert (std >= 0).all()


def test_batchnorm_buffers_are_unchanged():
    model = make_model()
    before = buffers(model)
    model.predict_with_uncertainty(torch.randn(3, 3), n_mc=3)
    assert_buffers_equal(model, before)


@pytest.mark.parametrize("mode", ["train", "eval", "mixed"])
def test_exact_module_modes_are_restored_after_success(mode):
    model = make_model(mode)
    before = modes(model)
    model.predict_with_uncertainty(torch.randn(3, 3), n_mc=3)
    assert modes(model) == before


@pytest.mark.parametrize("mode", ["train", "eval", "mixed"])
def test_exact_module_modes_are_restored_after_forward_exception(mode):
    model = make_model(mode)
    before_modes, before_buffers = modes(model), buffers(model)
    original = model.forward
    calls = 0
    def fail_on_second_call(x):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-pass failure")
        return original(x)
    model.forward = fail_on_second_call
    with pytest.raises(RuntimeError, match="injected second-pass failure"):
        model.predict_with_uncertainty(torch.randn(3, 3), n_mc=3)
    assert calls == 2
    assert modes(model) == before_modes
    assert_buffers_equal(model, before_buffers)


def test_only_dropout_is_training_during_every_mc_pass():
    model = make_model("mixed")
    original = model.forward
    seen = []
    def observed(x):
        seen.append([(isinstance(m, torch.nn.Dropout), m.training) for m in model.modules()])
        return original(x)
    model.forward = observed
    model.predict_with_uncertainty(torch.randn(3, 3), n_mc=3)
    assert len(seen) == 3
    assert all(training == dropout for state in seen for dropout, training in state)


def test_prediction_disables_autograd_and_restores_outer_grad_context():
    model = make_model()
    x = torch.randn(3, 3, requires_grad=True)
    enabled, saved = [], []
    original = model.forward
    def observed(x):
        enabled.append(torch.is_grad_enabled())
        return original(x)
    model.forward = observed
    def pack(value):
        saved.append(value)
        return value
    with torch.enable_grad():
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            mean, std = model.predict_with_uncertainty(x, n_mc=3)
        assert torch.is_grad_enabled()
    assert enabled == [False, False, False]
    assert saved == []
    assert not mean.requires_grad and not std.requires_grad
    assert mean.grad_fn is None and std.grad_fn is None
    assert x.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("n_mc,error", [
    (1, ValueError), (0, ValueError), (-1, ValueError),
    (True, TypeError), (False, TypeError),
    (2.0, TypeError), ("2", TypeError), (None, TypeError),
    ([2], TypeError), (np.bool_(True), TypeError), (torch.tensor(2), TypeError),
])
def test_invalid_pass_counts_fail_before_state_or_forward_changes(n_mc, error):
    model = make_model("mixed")
    before_modes, before_buffers = modes(model), buffers(model)
    called = []
    mode_calls = []
    original = model.forward
    original_train = model.train
    def observed(x):
        called.append(True)
        return original(x)
    model.forward = observed
    def observed_train(mode=True):
        mode_calls.append(mode)
        return original_train(mode)
    model.train = observed_train
    with pytest.raises(error, match="n_mc.*integer.*2"):
        model.predict_with_uncertainty(torch.randn(3, 3), n_mc=n_mc)
    assert called == []
    assert mode_calls == []
    assert modes(model) == before_modes
    assert_buffers_equal(model, before_buffers)


@pytest.mark.parametrize("n_mc", [2, np.int32(2), np.int64(2)])
def test_positive_integral_counts_are_accepted(n_mc):
    mean, std = make_model().predict_with_uncertainty(torch.randn(3, 3), n_mc=n_mc)
    assert mean.shape == std.shape == (3, 2)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()


def test_established_sample_variance_formula_is_preserved_for_multiple_draws():
    model = make_model()
    outputs = torch.tensor([[[1.0, 3.0, 0.0, 1.0]],
                            [[5.0, 7.0, 2.0, 3.0]],
                            [[9.0, 11.0, 4.0, 5.0]]])
    draws = iter(outputs)
    model.forward = lambda x: next(draws)
    mean, std = model.predict_with_uncertainty(torch.zeros(1, 3), n_mc=3)
    mus, logvars = outputs[..., :2], outputs[..., 2:]
    torch.testing.assert_close(mean, mus.mean(0))
    torch.testing.assert_close(std.square(), mus.var(0, unbiased=True) + logvars.exp().mean(0))


def test_mc_dropout_is_stochastic_with_batchnorm_frozen():
    model = make_model()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.fill_(0.3)
                module.bias.fill_(0.2)
    draws = []
    original = model.forward
    def observed(x):
        result = original(x)
        draws.append(result.detach().clone())
        return result
    model.forward = observed
    model.predict_with_uncertainty(torch.ones(3, 3), n_mc=4)
    assert any(not torch.equal(draws[0], value) for value in draws[1:])


def test_seeded_predictions_match_manual_dropout_only_oracle():
    model = make_model("mixed", dtype=torch.float64)
    oracle = copy.deepcopy(model)
    before_modes = modes(model)
    x = torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float64)
    oracle.eval()
    for module in oracle.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
    torch.manual_seed(31415)
    with torch.no_grad():
        outputs = torch.stack([oracle(x) for _ in range(5)])
        means, logvars = outputs[..., :2], outputs[..., 2:]
        expected_mean = means.mean(0)
        expected_std = (means.var(0, unbiased=True) + logvars.exp().mean(0)).sqrt()
    torch.manual_seed(31415)
    mean, std = model.predict_with_uncertainty(x, n_mc=5)
    torch.testing.assert_close(mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(std, expected_std, rtol=0, atol=0)
    assert modes(model) == before_modes


def test_parameters_buffers_and_existing_gradients_are_unchanged():
    model = make_model("mixed")
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    states = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = {name: value.grad.clone() for name, value in model.named_parameters()}
    x = torch.randn(3, 3, requires_grad=True)
    x_before = x.detach().clone()
    model.predict_with_uncertainty(x, n_mc=3)
    for name, value in model.state_dict().items():
        assert torch.equal(value, states[name]), name
    for name, value in model.named_parameters():
        assert torch.equal(value.grad, gradients[name]), name
    assert torch.equal(x, x_before)
    assert x.grad is None
