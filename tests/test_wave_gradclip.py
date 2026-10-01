"""Why gradient clipping is second-order under Adam (optimisation plan item 1, §8).

Item 1 (adaptive gradient clip) was implemented, ablated, and found to be a measured
NEGATIVE: across c in {1, 8, 20, adaptive kappa=2} the val bpb moved only ~0.009
(the tightest clip marginally best). The adaptive machinery was therefore reverted;
the default clip is unchanged (global L2 norm to grad_clip=1.0).

This single test is kept as the lasting record of *why*: it numerically proves the
Section 1.2 claim that scaling every gradient by a CONSTANT factor leaves AdamW's
parameter trajectory unchanged (away from the eps floor). That cancellation is the
reason a clip which fires every step cannot help/hurt Adam through its average
magnitude -- so relaxing it is not a convergence lever.
"""

import torch


def test_gradclip_constant_factor_is_adam_noop():
    """Scaling every gradient by a constant s does not change AdamW's trajectory:
    m_hat -> s*m_hat, sqrt(v_hat) -> s*sqrt(v_hat), and s cancels in m_hat/sqrt(v_hat)
    once eps is negligible. This is why a per-step clip is ~second-order for Adam."""
    torch.manual_seed(0)
    target = torch.randn(16, dtype=torch.float64)
    theta_a = torch.nn.Parameter(torch.zeros(16, dtype=torch.float64))
    theta_b = torch.nn.Parameter(torch.zeros(16, dtype=torch.float64))
    optA = torch.optim.AdamW([theta_a], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    optB = torch.optim.AdamW([theta_b], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    s = 0.3
    for _ in range(100):
        optA.zero_grad(set_to_none=True)
        optB.zero_grad(set_to_none=True)
        theta_a.grad = (theta_a.detach() - target)          # grad of 0.5||theta - t||^2
        theta_b.grad = s * (theta_b.detach() - target)      # identical grad, scaled by const s
        optA.step()
        optB.step()
    err = (theta_a - theta_b).abs().max().item()
    assert err < 1e-6, f"constant-factor clip changed the Adam trajectory by {err}"
