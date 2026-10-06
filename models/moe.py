import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SwiGLU

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.n_embd = config.n_embd

        # Router
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)

        # Expert MLPs
        self.experts = nn.ModuleList([
            SwiGLU(
                config.n_embd,
                config.expert_intermediate_size,
                config.n_embd,
                config.bias
            ) for _ in range(config.n_experts)
        ])

        # Shared expert
        if config.use_shared_expert:
            self.shared_expert = SwiGLU(
                config.n_embd,
                config.shared_expert_intermediate_size,
                config.n_embd,
                config.bias
            )
        else:
            self.shared_expert = None

        # Auxiliary-loss-free load balancing (DeepSeek-V3 style): a bias that
        # steers expert selection without entering the loss.
        self.register_buffer('expert_bias', torch.zeros(config.n_experts))
        self.bias_update_rate = getattr(config, 'bias_update_rate', 0.001)
        # The bias must stay small relative to the affinities it adjusts; an
        # unbounded bias stops being a correction and becomes the router.
        self.max_expert_bias = getattr(config, 'max_expert_bias', 1.0)
        # Usage within this fraction of an even share counts as balanced.
        self.balance_deadband = getattr(config, 'balance_deadband', 0.02)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        # Routing phase.
        #
        # The balancing bias steers WHICH experts are chosen; it must not reach
        # the gate values, because those carry the router's gradient. Softmaxing
        # the biased logits (as this did before) feeds the bias back into the
        # router's own learning signal, so the router learns to counteract the
        # correction being applied to it. The two then escalate: measured on
        # checkpoint sft_003, the bias had run to +113.5 for an expert still
        # receiving 0% of traffic, while the two experts taking 100% of it held
        # the lowest biases in the block. Selection and gating are therefore
        # computed from different tensors, as in DeepSeek-V3.
        affinity = self.router(x_flat)                     # gradient path
        selection_logits = affinity + self.expert_bias     # selection only

        _, top_k_indices = torch.topk(selection_logits, self.top_k, dim=-1)
        top_k_affinity = affinity.gather(-1, top_k_indices)
        routing_weights = torch.zeros_like(affinity)
        routing_weights.scatter_(-1, top_k_indices,
                                 F.softmax(top_k_affinity, dim=-1))

        # Expert computation
        output = torch.zeros_like(x_flat)
        expert_usage = torch.zeros(self.n_experts, device=x.device)

        # Process through selected experts
        for expert_idx in range(self.n_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            expert_usage[expert_idx] = expert_mask.sum().float()

            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)

                # Weight by routing probability
                weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)
                output[expert_mask] += expert_output * weights

        # Add shared expert output
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x_flat)
            output += shared_output

        # Auxiliary-loss-free load balancing.
        #
        # A fixed-size step in a fixed direction never settles: an expert at the
        # mean keeps being nudged, and over a long run the bias simply walks.
        # After 247,850 steps the saved biases spanned 148 in a single block,
        # which is not a nudge. The step is now proportional to how far the
        # expert actually is from even usage, with a deadband so balanced
        # experts are left alone, and the bias is bounded so it can never
        # dominate the affinities it is meant to adjust.
        if self.training:
            with torch.no_grad():
                total = expert_usage.sum()
                if total > 0:
                    share = expert_usage / total
                    target = 1.0 / self.n_experts
                    error = share - target
                    # Inside the deadband the expert is balanced; leave it.
                    adjust = torch.where(error.abs() > self.balance_deadband,
                                         -error / target, torch.zeros_like(error))
                    self.expert_bias += self.bias_update_rate * adjust
                    self.expert_bias.clamp_(-self.max_expert_bias,
                                            self.max_expert_bias)

        return output.view(batch_size, seq_len, hidden_size)