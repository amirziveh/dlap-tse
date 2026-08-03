#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_sdf_models.py — unit tests for the CPZ-faithful common SDF core
====================================================================
Verifies (against hand calculations and the official CPZ structure):
  - common_sdf:            M_t = 1 - (1/N_t) * sum_i omega_t(i) R^e_{t,i} * mean(N_t)
  - pricing_errors_common: alpha_i = mean_t(M_t R^e_{t,i}) with a COMMON M
  - weighted_pricing_loss: official form: mean_i (count_i / max_count) * alpha_i^2
  - SDFNet/MomentsNet:     shapes, tanh bound, nonlinearity in x
  - sign_normalize:        per-window sign guard for loadings
  - training convergence:  common SDF prices a synthetic factor structure in-sample
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sdf_models import (SDFNet, MomentsNet, common_sdf, pricing_errors_common,
                        weighted_pricing_loss, sign_normalize,
                        sdf_portfolio_return, critic_moment_alphas)

torch.manual_seed(0)


def test_common_sdf_matches_hand_calc():
    T, N = 5, 4
    omega = torch.rand(T, N)
    R = torch.rand(T, N) - 0.5
    mask = torch.ones(T, N, dtype=torch.bool)
    M = common_sdf(omega, R, mask)
    n_t = mask.sum(dim=1).float()
    hand = 1.0 - (omega * R).sum(dim=1) / n_t
    assert torch.allclose(M, hand, atol=1e-6), M


def test_common_sdf_masks_invalid():
    T, N = 5, 4
    omega = torch.rand(T, N)
    R = torch.rand(T, N) - 0.5
    mask = torch.ones(T, N, dtype=torch.bool)
    mask[2, 1] = False
    M = common_sdf(omega, R, mask)
    wr = (omega * R) * mask.float()
    n_t = mask.sum(dim=1).float()
    hand = 1.0 - wr.sum(dim=1) / n_t
    assert torch.allclose(M, hand, atol=1e-6)


def test_common_sdf_is_common_across_stocks():
    """The SDF value at time t must NOT depend on the stock index."""
    T, N = 6, 5
    omega = torch.rand(T, N)
    R = torch.rand(T, N) - 0.5
    mask = torch.ones(T, N, dtype=torch.bool)
    M = common_sdf(omega, R, mask)
    assert M.shape == (T,)
    assert M.ndim == 1


def test_pricing_errors_common():
    T, N = 6, 3
    M = torch.tensor([1.0, 1.1, 1.2, 0.9, 1.0, 1.05])
    R = torch.rand(T, N) - 0.5
    mask = torch.ones(T, N, dtype=torch.bool)
    alpha = pricing_errors_common(M, R, mask)
    for i in range(N):
        hand = (M * R[:, i]).mean()
        assert abs(alpha[i].item() - hand.item()) < 1e-6


def test_pricing_errors_requires_min_obs():
    T, N = 10, 2
    M = torch.rand(T)
    R = torch.rand(T, N)
    mask = torch.ones(T, N, dtype=torch.bool)
    mask[5:, 0] = False  # stock 0 has 5 obs only -> NaN
    alpha = pricing_errors_common(M, R, mask)
    assert torch.isnan(alpha[0])
    assert not torch.isnan(alpha[1])


def test_weighted_loss_official_form():
    # counts: stock0=4, stock1=6, stock2=6 -> w = 2/3, 1, 1 ; mean over valid
    alpha = torch.tensor([0.1, 0.2, 0.3])
    mask = torch.zeros(6, 3, dtype=torch.bool)
    mask[:, 0] = torch.tensor([1, 1, 1, 1, 0, 0]).bool()
    mask[:, 1] = True
    mask[:, 2] = True
    loss = weighted_pricing_loss(alpha, mask)
    w = torch.tensor([4 / 6, 1.0, 1.0])
    hand = (w * alpha ** 2).mean()
    assert abs(loss.item() - hand.item()) < 1e-6


def test_sdfnet_shapes_and_nonlinearity():
    T, N, S, F = 5, 4, 4, 11
    net = SDFNet(state_dim=S, n_features=F)
    z = torch.randn(T, S)
    x = torch.randn(T, N, F)
    w = net(z, x)
    assert w.shape == (T, N)
    w2 = net(z, x * 0.5)
    assert not torch.allclose(w, w2)  # nonlinear in characteristics


def test_momentsnet_shape_and_tanh():
    T, N, S, F, K = 5, 4, 4, 11, 8
    net = MomentsNet(state_dim=S, n_features=F, n_moments=K)
    z = torch.randn(T, S)
    x = torch.randn(T, N, F)
    m = net(z, x)
    assert m.shape == (T, N, K)
    assert m.abs().max().item() <= 1.0 + 1e-6


def test_sign_normalize():
    omega = torch.randn(6, 5)
    rp = -omega.sum(dim=1).abs() - 0.1  # negative portfolio mean -> must flip
    out = sign_normalize(omega, rp)
    assert out.sum().item() >= 0


def test_critic_moment_alphas():
    T, N, K = 6, 5, 8
    M = torch.rand(T)
    m = torch.rand(T, N, K) * 2 - 1
    R = torch.rand(T, N) - 0.5
    mask = torch.ones(T, N, dtype=torch.bool)
    ak = critic_moment_alphas(M, m, R, mask)
    assert ak.shape == (K, N)
    # hand computation for one (k, i)
    hand = (m[:, 2, 3] * M * R[:, 2]).mean()
    assert abs(ak[3, 2].item() - hand.item()) < 1e-6
    # NaN where insufficient obs
    mask2 = mask.clone()
    mask2[3:, 0] = False  # 3 valid obs < 6 -> NaN
    ak2 = critic_moment_alphas(M, m, R, mask2)
    assert torch.isnan(ak2[:, 0]).all()
    assert not torch.isnan(ak2[:, 1]).any()


def test_sdf_portfolio_return():
    T, N = 4, 3
    omega = torch.tensor([[1.0, -2.0, 0.5], [0.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    R = torch.rand(T, N) - 0.5
    mask = torch.ones(T, N, dtype=torch.bool)
    rp = sdf_portfolio_return(omega, R, mask)
    for t in range(T):
        hand = (omega[t] * R[t]).sum() / omega[t].abs().sum()
        assert abs(rp[t].item() - hand.item()) < 1e-6


def test_training_converges_synthetic():
    """Synthetic one-factor structure: R_i = b_i * f + eps, f = mu + sig*eta.
    The common SDF M = 1 - (1/N) sum_j omega_j R_j with omega_j ~ b_j should
    drive in-sample pricing errors toward zero."""
    torch.manual_seed(1)
    T, N = 60, 30
    b = 0.5 + torch.rand(N)                       # betas in (0.5, 1.5)
    f = 0.05 + 0.10 * torch.randn(T)              # factor with mu=0.05
    R = b.unsqueeze(0) * f.unsqueeze(1) + 0.01 * torch.randn(T, N)
    z = torch.randn(T, 4)
    x = b.unsqueeze(0).unsqueeze(-1).expand(T, N, 1)   # F=1 char = beta
    mask = torch.ones(T, N, dtype=torch.bool)
    net = SDFNet(state_dim=4, n_features=1, hidden=(16, 16), dropout_p=0.0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(3000):
        opt.zero_grad()
        w = net(z, x)
        M = common_sdf(w, R, mask)
        alpha = pricing_errors_common(M, R, mask)
        loss = weighted_pricing_loss(alpha, mask)
        loss.backward()
        opt.step()
    with torch.no_grad():
        M = common_sdf(net(z, x), R, mask)
        alpha = pricing_errors_common(M, R, mask)
    mean_abs_alpha = alpha.abs().mean().item()
    assert mean_abs_alpha < 0.02, f"in-sample mean|alpha| = {mean_abs_alpha:.4f}"


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {e}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
