"""
Hybrid MoE Layer — patches a gpt-oss-20b MoE layer with deterministic computation experts.

Wraps the original MoE forward pass:
1. Runs the extended router (36 experts instead of 32)
2. For tokens routed to experts 0-31: runs original neural FFN experts (frozen)
3. For tokens routed to experts 32-35: runs deterministic computation experts
4. Combines outputs using router weights

gpt-oss-20b architecture (discovered via explore_model.py):
  - Router: GptOssTopKRouter — linear projection (2880 → 32) + per-expert bias, sigmoid top-4
  - Experts: GptOssExperts — packed weight tensors:
      gate_up_proj: (32, 2880, 5760)   [SwiGLU gate + up projection]
      down_proj:    (32, 2880, 2880)   [down projection]
  - Layer path: model.model.layers[i].mlp
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from .extended_router import ExtendedRouter
from .deterministic_expert import DeterministicExpertBank
from .cre_experts import EXPERT_REGISTRY, NUM_COMPUTATION_EXPERTS


ACTIVE_EXPERTS = {k: v for k, v in EXPERT_REGISTRY.items()
                  if k in ("dscr", "ltv", "cap_rate", "debt_yield")}
NUM_ACTIVE = len(ACTIVE_EXPERTS)


class HybridMoELayer(nn.Module):
    """
    Replaces a gpt-oss-20b MoE layer with extended routing + computation experts.

    During forward:
    - Extended router selects top-4 from 36 experts (32 neural + 4 deterministic)
    - Tokens routed to experts 0-31 go through original packed SwiGLU experts
    - Tokens routed to experts 32-35 go through deterministic computation experts
    - Outputs are weighted and summed
    """

    def __init__(
        self,
        original_moe_layer,
        hidden_dim: int = 2880,
        n_original_experts: int = 32,
        top_k: int = 4,
        expert_intermediate_dim: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_original = n_original_experts
        self.top_k = top_k

        self.original_moe = original_moe_layer

        original_router = self._find_router(original_moe_layer)
        self.extended_router = ExtendedRouter.from_original(
            original_router, n_new=NUM_ACTIVE, top_k=top_k)

        self.comp_experts = DeterministicExpertBank(
            ACTIVE_EXPERTS, hidden_dim=hidden_dim,
            intermediate_dim=expert_intermediate_dim)

        for p in self.original_moe.parameters():
            p.requires_grad = False

        self._gate_up_proj, self._down_proj = self._find_expert_weights(original_moe_layer)
        self._routing_log = None

    def _find_router(self, moe_layer) -> nn.Linear:
        """Find the router linear layer. gpt-oss-20b uses GptOssTopKRouter."""
        for attr in ['router', 'gate', 'w_gate', 'gate_proj', 'routing']:
            if hasattr(moe_layer, attr):
                candidate = getattr(moe_layer, attr)
                if isinstance(candidate, nn.Linear):
                    return candidate
                if hasattr(candidate, 'weight') and hasattr(candidate.weight, 'shape'):
                    if candidate.weight.shape[0] == self.n_original:
                        return candidate

        for name, mod in moe_layer.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.shape[0] == self.n_original:
                return mod

        raise ValueError(
            "Could not auto-detect router. "
            "Run scripts/explore_model.py to find the correct attribute name."
        )

    def _find_expert_weights(self, moe_layer):
        """
        Find packed expert weight tensors in gpt-oss-20b.
        Expected shapes: gate_up_proj (32, 2880, 5760), down_proj (32, 2880, 2880).
        """
        gate_up = None
        down = None

        for name, param in moe_layer.named_parameters():
            if 'gate_up' in name and param.dim() == 3:
                gate_up = param
            elif 'down_proj' in name and param.dim() == 3:
                down = param

        if gate_up is not None and down is not None:
            return gate_up, down

        for name, param in moe_layer.named_parameters():
            if param.dim() == 3 and param.shape[0] == self.n_original:
                if param.shape[2] == self.hidden_dim * 2:
                    gate_up = param
                elif param.shape[1] == self.hidden_dim and param.shape[2] == self.hidden_dim:
                    down = param

        return gate_up, down

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, dim)

        weights, indices, logits = self.extended_router(flat_hidden.unsqueeze(0))
        weights = weights.squeeze(0)
        indices = indices.squeeze(0)

        output = torch.zeros_like(flat_hidden)

        for k in range(self.top_k):
            expert_ids = indices[:, k]
            expert_weights = weights[:, k].unsqueeze(-1)

            is_neural = expert_ids < self.n_original
            is_comp = ~is_neural

            if is_neural.any():
                neural_hidden = flat_hidden[is_neural]
                neural_ids = expert_ids[is_neural]
                neural_out = self._run_neural_experts(neural_hidden, neural_ids)
                output[is_neural] += expert_weights[is_neural] * neural_out

            if is_comp.any():
                comp_hidden = flat_hidden[is_comp]
                comp_ids = expert_ids[is_comp]
                comp_out = self._run_comp_experts(comp_hidden, comp_ids)
                output[is_comp] += expert_weights[is_comp] * comp_out

        self._routing_log = self.extended_router.routing_stats(indices)
        return output.reshape(batch, seq_len, dim)

    def _run_neural_experts(self, hidden: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """
        Run tokens through original gpt-oss-20b SwiGLU experts using packed weights.

        gpt-oss-20b expert computation:
          gate_up = hidden @ gate_up_proj[expert_id]   # (tokens, 5760)
          gate, up = gate_up.chunk(2, dim=-1)          # each (tokens, 2880)
          intermediate = F.silu(gate) * up              # SwiGLU
          output = intermediate @ down_proj[expert_id].T
        """
        output = torch.zeros_like(hidden)

        if self._gate_up_proj is None or self._down_proj is None:
            return hidden

        unique_ids = expert_ids.unique()
        for eid in unique_ids:
            mask = expert_ids == eid
            tokens = hidden[mask]

            gate_up = tokens @ self._gate_up_proj[eid]
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = F.silu(gate) * up
            expert_out = intermediate @ self._down_proj[eid].T

            output[mask] = expert_out

        return output

    def _run_comp_experts(self, hidden: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """Run tokens through deterministic computation experts."""
        output = torch.zeros_like(hidden)
        unique_ids = expert_ids.unique()

        for eid in unique_ids:
            mask = expert_ids == eid
            tokens = hidden[mask]
            expert_out = self.comp_experts.forward_expert(eid.item(), tokens)
            output[mask] = expert_out

        return output

    def get_routing_log(self) -> Optional[Dict]:
        return self._routing_log

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def patch_model(
    model,
    layers_to_patch: Optional[List[int]] = None,
    hidden_dim: int = 2880,
    top_k: int = 4,
    expert_intermediate_dim: int = 512,
):
    """
    Patch a gpt-oss-20b model by replacing MoE layers with HybridMoELayers.

    gpt-oss-20b layer path: model.model.layers[i].mlp
    The MoE block lives at the .mlp attribute of each transformer layer.

    Args:
        model: The loaded gpt-oss-20b model
        layers_to_patch: Layer indices to patch (default: [8, 12, 16, 20])

    Returns:
        Tuple of (patched model, list of HybridMoELayer references)
    """
    if layers_to_patch is None:
        layers_to_patch = [8, 12, 16, 20]

    hybrid_layers = []

    layers_container = None
    for path in ['model.model.layers', 'model.layers', 'transformer.h']:
        obj = model
        try:
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            layers_container = obj
            print(f"  Found layers at: {path}")
            break
        except AttributeError:
            continue

    if layers_container is None:
        raise ValueError(
            "Could not find transformer layers. "
            "Run scripts/explore_model.py to find the correct path."
        )

    for layer_idx in layers_to_patch:
        layer = layers_container[layer_idx]

        moe_attr = None
        for attr in ['mlp', 'moe', 'block_sparse_moe', 'feed_forward']:
            if hasattr(layer, attr):
                moe_attr = attr
                break

        if moe_attr is None:
            print(f"  Warning: could not find MoE module in layer {layer_idx}, skipping")
            continue

        original_moe = getattr(layer, moe_attr)
        hybrid = HybridMoELayer(
            original_moe_layer=original_moe,
            hidden_dim=hidden_dim,
            top_k=top_k,
            expert_intermediate_dim=expert_intermediate_dim,
        )

        setattr(layer, moe_attr, hybrid)
        hybrid_layers.append(hybrid)
        print(f"  Patched layer {layer_idx} ({moe_attr})")

    total_new_params = sum(h.trainable_params() for h in hybrid_layers)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n  Patched {len(hybrid_layers)} layers")
    print(f"  New trainable params: {total_new_params:,}")
    print(f"  Frozen params: {frozen_params:,}")

    return model, hybrid_layers
