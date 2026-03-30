"""
Extended Router — extends gpt-oss-20b's 32-expert router to 36 (32 frozen + 4 trainable).
gpt-oss-20b uses: linear projection (2880→32), per-expert bias, top-4 with sigmoid routing.
Supports arbitrary n_new_experts (4 active in experiments, 6 defined in cre_experts.py).
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class ExtendedRouter(nn.Module):
    def __init__(self, original_weight: torch.Tensor, original_bias: Optional[torch.Tensor],
                 n_new_experts: int = 6, hidden_dim: int = 2880, top_k: int = 4):
        super().__init__()
        self.n_original = original_weight.shape[0]
        self.n_new = n_new_experts
        self.n_total = self.n_original + n_new_experts
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        # Freeze originals
        self.register_buffer("orig_weight", original_weight.detach().clone())
        self.register_buffer("orig_bias",
            original_bias.detach().clone() if original_bias is not None
            else torch.zeros(self.n_original))

        # Trainable new columns — small init so they don't dominate early
        self.new_weight = nn.Parameter(torch.randn(n_new_experts, hidden_dim) * 0.01)
        # Negative bias so computation experts aren't selected by default
        self.new_bias = nn.Parameter(torch.full((n_new_experts,), -2.0))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args: hidden_states (*, hidden_dim)
        Returns: weights (*, top_k), indices (*, top_k), logits (*, n_total)
        """
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            b, s, d = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, d)

        all_w = torch.cat([self.orig_weight, self.new_weight], dim=0)
        all_b = torch.cat([self.orig_bias, self.new_bias], dim=0)
        logits = hidden_states @ all_w.T + all_b

        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
        # gpt-oss uses sigmoid + normalize (not softmax)
        topk_weights = torch.sigmoid(topk_vals)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        if len(orig_shape) == 3:
            topk_weights = topk_weights.reshape(b, s, self.top_k)
            topk_idx = topk_idx.reshape(b, s, self.top_k)
            logits = logits.reshape(b, s, self.n_total)

        return topk_weights, topk_idx, logits

    def routing_stats(self, indices: torch.Tensor) -> dict:
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.n_total)
        total = flat.numel()
        neural = counts[:self.n_original].sum().item()
        comp = counts[self.n_original:].sum().item()
        return {
            "total": total, "neural": neural, "computation": comp,
            "computation_ratio": comp / max(total, 1),
            "per_computation_expert": {
                f"expert_{self.n_original + i}": counts[self.n_original + i].item()
                for i in range(self.n_new)
            },
        }

    def trainable_params(self) -> int:
        return self.new_weight.numel() + self.new_bias.numel()

    @classmethod
    def from_original(cls, linear: nn.Linear, n_new: int = 6, top_k: int = 4):
        """Create from an existing router nn.Linear layer."""
        return cls(linear.weight.data, 
                   linear.bias.data if linear.bias is not None else None,
                   n_new, linear.weight.shape[1], top_k)


def test_extended_router():
    print("Testing ExtendedRouter...")
    hidden_dim = 2880
    n_new = 4

    orig_w = torch.randn(32, hidden_dim)
    orig_b = torch.randn(32)
    router = ExtendedRouter(orig_w, orig_b, n_new_experts=n_new, hidden_dim=hidden_dim)

    h = torch.randn(2, 10, hidden_dim)
    weights, indices, logits = router(h)
    assert weights.shape == (2, 10, 4), f"weights: {weights.shape}"
    assert indices.shape == (2, 10, 4), f"indices: {indices.shape}"
    assert logits.shape == (2, 10, 36), f"logits: {logits.shape}"
    print(f"  ✓ Forward: (2,10,2880) → weights{weights.shape} indices{indices.shape}")

    assert torch.allclose(weights.sum(-1), torch.ones(2, 10), atol=0.01)
    print(f"  ✓ Weights sum to ~1.0")

    assert indices.min() >= 0 and indices.max() < 36
    print(f"  ✓ Indices in [0, 35]")

    stats = router.routing_stats(indices)
    print(f"  ✓ Routing: {stats['computation']} computation selections ({stats['computation_ratio']:.2%})")

    h = torch.randn(4, hidden_dim)
    w, idx, _ = router(h.unsqueeze(0))
    w.sum().backward()
    assert router.new_weight.grad is not None
    print(f"  ✓ Gradients flow to new params only ({router.trainable_params():,} trainable)")

    linear = nn.Linear(hidden_dim, 32)
    r2 = ExtendedRouter.from_original(linear, n_new=n_new)
    print(f"  ✓ from_original: {r2.n_total} total experts")
    print("\n  All ExtendedRouter tests passed!")


if __name__ == "__main__":
    test_extended_router()
