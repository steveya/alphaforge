"""Torch-compatible pipeline component protocols.

These extend the base Parametric protocol for components that can
participate in gradient-based optimization.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@runtime_checkable
class TorchParametric(Protocol):
    """A component whose parameters are torch tensors."""

    def parameters(self) -> list: ...  # list[nn.Parameter]
    def named_parameters(self) -> list: ...  # list[tuple[str, nn.Parameter]]
    def get_params(self) -> dict[str, float]: ...
    def set_params(self, params: dict[str, float]) -> None: ...


if HAS_TORCH:

    class DifferentiableSignalModule(nn.Module):
        """Base class for differentiable signal pipeline components."""

        name: str = "base"

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        def get_params(self) -> dict[str, float]:
            result = {}
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if p.numel() == 1:
                    result[name] = float(p.item())
                else:
                    # Multi-element params: store as first element (summary)
                    result[name] = float(p[0].item())
            return result

        def set_params(self, params: dict[str, float]) -> None:
            with torch.no_grad():
                for name, value in params.items():
                    parts = name.split(".")
                    obj = self
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    if hasattr(obj, parts[-1]):
                        param = getattr(obj, parts[-1])
                        if isinstance(param, nn.Parameter):
                            param.fill_(value)

    class _NumpyAutograd(torch.autograd.Function):
        """Custom autograd bridging numpy -> torch."""

        @staticmethod
        def forward(ctx, x, forward_fn, backward_fn, params_dict, fd_eps):
            x_np = x.detach().cpu().numpy()
            params_float = {k: float(v.item()) for k, v in params_dict.items()}
            result_np = forward_fn(x_np, params_float)
            ctx.save_for_backward(x)
            ctx.forward_fn = forward_fn
            ctx.backward_fn = backward_fn
            ctx.params_dict = params_dict
            ctx.params_float = params_float
            ctx.fd_eps = fd_eps
            return torch.tensor(result_np, dtype=x.dtype, device=x.device)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            x_np = x.detach().cpu().numpy()
            grad_np = grad_output.detach().cpu().numpy()

            if ctx.backward_fn is not None:
                grads = ctx.backward_fn(x_np, grad_np, ctx.params_float)
                grad_x = torch.tensor(
                    grads.get("input", np.zeros_like(x_np)),
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                # Finite differences for input gradient
                eps = ctx.fd_eps
                grad_x = torch.zeros_like(x)
                flat = x_np.ravel()
                for i in range(min(len(flat), 100)):  # cap for perf
                    flat_p = flat.copy()
                    flat_p[i] += eps
                    x_p = flat_p.reshape(x_np.shape)
                    out_p = ctx.forward_fn(x_p, ctx.params_float)
                    flat_m = flat.copy()
                    flat_m[i] -= eps
                    x_m = flat_m.reshape(x_np.shape)
                    out_m = ctx.forward_fn(x_m, ctx.params_float)
                    deriv = (out_p - out_m) / (2 * eps)
                    grad_x.view(-1)[i] = (
                        torch.tensor(deriv, dtype=x.dtype).ravel() * grad_output.ravel()
                    ).sum()

            return grad_x, None, None, None, None

    class NumpyDifferentiableWrapper(DifferentiableSignalModule):
        """Wrap a numpy function as a differentiable torch module."""

        def __init__(
            self,
            forward_fn,
            backward_fn=None,
            param_names: list[str] | None = None,
            init_params: dict[str, float] | None = None,
            fd_eps: float = 1e-5,
            name: str = "numpy_wrapper",
        ):
            super().__init__()
            self.name = name
            self.forward_fn = forward_fn
            self.backward_fn = backward_fn
            self.fd_eps = fd_eps
            self.param_names_list = list(param_names or [])

            for pname in self.param_names_list:
                val = (init_params or {}).get(pname, 0.0)
                setattr(self, pname, nn.Parameter(torch.tensor(float(val))))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            params = {
                pname: getattr(self, pname) for pname in self.param_names_list
            }
            return _NumpyAutograd.apply(
                x,
                self.forward_fn,
                self.backward_fn,
                params,
                self.fd_eps,
            )
