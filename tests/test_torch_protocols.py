"""Tests for alphaforge.pipeline.torch_protocols module."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from alphaforge.pipeline.torch_protocols import (
    DifferentiableSignalModule,
    NumpyDifferentiableWrapper,
    TorchParametric,
)


class _SimpleModule(DifferentiableSignalModule):
    name = "simple"

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return x * self.scale


class TestTorchParametricProtocol:
    def test_torch_parametric_protocol(self) -> None:
        m = _SimpleModule()
        assert isinstance(m, TorchParametric)


class TestDifferentiableSignalModuleGetSet:
    def test_get_set(self) -> None:
        m = _SimpleModule()
        params = m.get_params()
        assert "scale" in params
        assert abs(params["scale"] - 2.0) < 1e-6
        m.set_params({"scale": 5.0})
        assert abs(m.scale.item() - 5.0) < 1e-6


class TestDifferentiableSignalModuleGradients:
    def test_gradients(self) -> None:
        m = _SimpleModule()
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = m(x)
        loss = y.sum()
        loss.backward()
        assert m.scale.grad is not None
        assert x.grad is not None


class TestNumpyWrapperForward:
    def test_numpy_wrapper_forward(self) -> None:
        def fwd(x_np, params):
            return x_np * params.get("scale", 1.0)

        wrapper = NumpyDifferentiableWrapper(
            forward_fn=fwd,
            param_names=["scale"],
            init_params={"scale": 3.0},
            name="test_wrapper",
        )
        x = torch.tensor([1.0, 2.0, 3.0])
        result = wrapper(x)
        np.testing.assert_allclose(result.detach().numpy(), [3.0, 6.0, 9.0])


class TestNumpyWrapperAnalyticalGrad:
    def test_numpy_wrapper_analytical_grad(self) -> None:
        def fwd(x_np, params):
            return x_np * params["scale"]

        def bwd(x_np, grad_np, params):
            return {"input": grad_np * params["scale"]}

        wrapper = NumpyDifferentiableWrapper(
            forward_fn=fwd,
            backward_fn=bwd,
            param_names=["scale"],
            init_params={"scale": 2.0},
        )
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = wrapper(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0])


class TestNumpyWrapperFiniteDiff:
    def test_numpy_wrapper_finite_diff(self) -> None:
        def fwd(x_np, params):
            return x_np ** 2 * params.get("scale", 1.0)

        wrapper = NumpyDifferentiableWrapper(
            forward_fn=fwd,
            param_names=["scale"],
            init_params={"scale": 1.0},
        )
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = wrapper(x)
        loss = y.sum()
        loss.backward()
        # Finite diff should give approximate gradients
        assert x.grad is not None
        # d/dx(x^2) = 2x at x=[2,3] → [4,6]
        np.testing.assert_allclose(x.grad.numpy(), [4.0, 6.0], atol=0.1)
