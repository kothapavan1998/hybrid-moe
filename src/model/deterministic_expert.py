"""
Deterministic Expert Module — core architectural component of Hybrid MoE.

Each deterministic expert replaces a neural FFN expert with a three-stage pipeline:
  1. Parameter Extraction MLP (learned): hidden_state → function arguments
  2. Deterministic Computation (exact): function arguments → result
  3. Result Projection (learned): result → hidden_dim embedding

Same input/output interface as a standard FFN expert: (batch, hidden_dim) → (batch, hidden_dim).
"""
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Dict


class DeterministicExpert(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_input_params: int,
        n_output_params: int,
        compute_fn: Callable[[torch.Tensor], torch.Tensor],
        intermediate_dim: int = 512,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        expert_name: str = "unnamed",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_input_params = n_input_params
        self.n_output_params = n_output_params
        self.compute_fn = compute_fn
        self.expert_name = expert_name
        self.input_names = input_names or [f"param_{i}" for i in range(n_input_params)]
        self.output_names = output_names or [f"output_{i}" for i in range(n_output_params)]

        # Stage 1: Extract function params from hidden state
        self.param_extractor = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.SiLU(),
            nn.Linear(intermediate_dim // 2, n_input_params),
        )

        # Stage 3: Project computation result back to hidden_dim
        self.result_projector = nn.Sequential(
            nn.Linear(n_output_params, intermediate_dim // 2),
            nn.SiLU(),
            nn.Linear(intermediate_dim // 2, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )

        # Start small to not disrupt the residual stream initially
        self.output_scale = nn.Parameter(torch.tensor(0.01))

        # Debug cache
        self._last_extracted_params = None
        self._last_result = None

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """(batch, hidden_dim) → (batch, hidden_dim)"""
        # Stage 1: Extract
        extracted = self.param_extractor(hidden_state)

        # Stage 2: Compute (deterministic, but differentiable via PyTorch autograd)
        result = self.compute_fn(extracted)

        # Stage 3: Project back
        projected = self.result_projector(result)

        # Scale and return
        output = projected * self.output_scale

        # Cache for debugging
        self._last_extracted_params = extracted.detach()
        self._last_result = result.detach()
        return output

    def get_last_computation(self) -> Optional[Dict]:
        if self._last_extracted_params is None:
            return None
        return {
            "expert": self.expert_name,
            "params": {n: self._last_extracted_params[:, i].tolist()
                      for i, n in enumerate(self.input_names)},
            "result": {n: self._last_result[:, i].tolist()
                      for i, n in enumerate(self.output_names)},
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"DeterministicExpert(name={self.expert_name}, "
                f"inputs={self.input_names}, params={self.param_count():,})")


class DeterministicExpertBank(nn.Module):
    """Collection of deterministic experts, indexed by expert_id."""

    def __init__(self, experts_registry: dict, hidden_dim: int = 2880,
                 intermediate_dim: int = 512):
        super().__init__()
        self.expert_modules = nn.ModuleDict()
        self.id_to_name = {}

        for name, cfg in experts_registry.items():
            expert = DeterministicExpert(
                hidden_dim=hidden_dim,
                n_input_params=cfg["n_input_params"],
                n_output_params=cfg["n_output_params"],
                compute_fn=cfg["function"],
                intermediate_dim=intermediate_dim,
                input_names=cfg.get("input_names"),
                output_names=cfg.get("output_names"),
                expert_name=name,
            )
            self.expert_modules[name] = expert
            self.id_to_name[cfg["expert_id"]] = name

    def forward_expert(self, expert_id: int, hidden_state: torch.Tensor) -> torch.Tensor:
        name = self.id_to_name[expert_id]
        return self.expert_modules[name](hidden_state)

    def is_computation_expert(self, expert_id: int) -> bool:
        return expert_id in self.id_to_name

    def get_all_computations(self) -> Dict:
        return {n: e.get_last_computation() for n, e in self.expert_modules.items()
                if e.get_last_computation() is not None}

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        lines = ["DeterministicExpertBank("]
        for n, e in self.expert_modules.items():
            lines.append(f"  {n}: {e.param_count():,} params")
        lines.append(f"  total: {self.total_params():,} params)")
        return "\n".join(lines)


def test_deterministic_expert():
    from .cre_experts import EXPERT_REGISTRY, compute_dscr

    print("Testing DeterministicExpert...")

    # Single expert
    expert = DeterministicExpert(
        hidden_dim=2880, n_input_params=2, n_output_params=1,
        compute_fn=compute_dscr, expert_name="dscr_test",
    )
    h = torch.randn(4, 2880)
    out = expert(h)
    assert out.shape == (4, 2880), f"Wrong shape: {out.shape}"
    print(f"  ✓ Single expert: (4, 2880) → {out.shape}, {expert.param_count():,} params")

    # Gradient flow
    h = torch.randn(2, 2880, requires_grad=True)
    out = expert(h)
    out.sum().backward()
    assert h.grad is not None
    print(f"  ✓ Gradient flows through deterministic computation")

    # Expert bank
    bank = DeterministicExpertBank(EXPERT_REGISTRY, hidden_dim=2880)
    h = torch.randn(8, 2880)
    out = bank.forward_expert(32, h)  # DSCR
    assert out.shape == (8, 2880)
    print(f"  ✓ Bank: expert 32 (DSCR), {bank.total_params():,} total params")
    print(f"\n{bank}")

    assert bank.is_computation_expert(32) == True
    assert bank.is_computation_expert(5) == False
    print("\n  ✓ All DeterministicExpert tests passed!")


if __name__ == "__main__":
    test_deterministic_expert()
