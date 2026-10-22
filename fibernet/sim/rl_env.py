"""Compatibility boundary for the unavailable effective-property RL environment.

The spring-dynamics SimResult does not provide a homogenized effective Young's
modulus E* or Poisson's ratio nu*. FiberNetworkEnv therefore rejects construction
before graph generation or provider work. Installing Gymnasium does not supply
this missing physical evaluator. No substitute modulus or reward is inferred.

For actual policy training, fibernet.rl.train_rl accepts an appropriate
caller-managed Gymnasium/SB3 environment. It does not create a physical evaluator.
"""

from __future__ import annotations

from typing import Optional, Tuple


_UNSUPPORTED_EFFECTIVE_PROPERTIES = (
    "FiberNetworkEnv is unavailable: effective Young's modulus E* and Poisson's "
    "ratio nu* are not provided by the spring-dynamics SimResult. "
    "A validated effective-property evaluator is required; changing target_E, "
    "target_nu, default_E or applied_strain cannot enable this route. "
    "For policy training, use fibernet.rl.train_rl with an appropriate "
    "caller-managed Gymnasium/SB3 environment."
)


class FiberNetworkEnv:
    """Unavailable legacy E*/nu* reward environment; retained for compatibility.

    The constructor signature is retained so existing callers receive an explicit
    NotImplementedError instead of failed simulations disguised as reward -10.
    All configurations are unsupported. No solver is initialized, and no episode
    is created. The arguments do not define a homogenization or material model.
    """

    def __init__(
        self,
        target_E: float = 1e6,
        target_nu: float = 0.0,
        box: Tuple[float, float] = (10.0, 10.0),
        max_grid: int = 8,
        radius_range: Tuple[float, float] = (0.02, 0.5),
        available_units: Optional[list] = None,
        multi_step: bool = False,
        default_E: float = 1e9,
        applied_strain: float = 0.01,
    ):
        raise NotImplementedError(_UNSUPPORTED_EFFECTIVE_PROPERTIES)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reject use of the unavailable effective-property evaluator."""
        raise NotImplementedError(_UNSUPPORTED_EFFECTIVE_PROPERTIES)

    def step(self, action: dict):
        """Reject use; an unsupported model is not a negative-reward episode."""
        raise NotImplementedError(_UNSUPPORTED_EFFECTIVE_PROPERTIES)

    def render(self):
        """No structure exists because the environment cannot be constructed."""
        return None

    def close(self):
        """No resources are acquired by this compatibility boundary."""
        pass
