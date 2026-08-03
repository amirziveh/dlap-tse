#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdf_models.py — DLAP-TSE Phase 3: deep SDF networks (CPZ 2024 architecture)
=============================================================================
Faithful to the official config (code/official_cpz/config/config.json):
  - Z_net: 1-layer LSTM over the macro panel -> z_t in R^4 (learned macro states)
  - M_net: MLP z_t -> SDF weights w(z_t) in R^F (hidden [64,64], dropout keep 0.95)
  - SDF:   M_{i,t} = 1 - w(z_t)' x_{i,t}   (linear in characteristics, network weights)
  - Loss:  mean_i ( E_t[M R^e] )^2   (squared pricing errors; weighted = VW option)
"""
import torch
import torch.nn as nn


class ZNet(nn.Module):
    """LSTM state extractor: macro panel -> low-dim states z_t (dim 4)."""

    def __init__(self, macro_dim: int, state_dim: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=macro_dim, hidden_size=state_dim,
                            num_layers=1, batch_first=True)

    def forward(self, macro_seq: torch.Tensor):
        """macro_seq: (T, macro_dim) -> z: (T, state_dim) (causal, per-t state)"""
        out, _ = self.lstm(macro_seq.unsqueeze(0))  # (1, T, state_dim)
        return out.squeeze(0)


class ConstZNet(nn.Module):
    """Learned constant state — NO macro conditioning (E4: states off).
    z_t = z for all t; macro input ignored."""

    def __init__(self, state_dim: int = 4):
        super().__init__()
        self.z = nn.Parameter(torch.zeros(state_dim))

    def forward(self, macro_seq: torch.Tensor):
        T = macro_seq.shape[0]
        return self.z.unsqueeze(0).expand(T, -1)


class CriticNet(nn.Module):
    """Adversarial critic (E5): z_t -> portfolio weights over characteristics
    w_c(z_t) in [-1, 1]^F. Chooses the portfolio the SDF prices worst."""

    def __init__(self, state_dim: int, n_features: int,
                 hidden: tuple = (64, 64), dropout_p: float = 0.05):
        super().__init__()
        layers = []
        d_in = state_dim
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout_p)]
            d_in = h
        layers += [nn.Linear(d_in, n_features), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        """z: (T, state_dim) -> wc: (T, F)"""
        return self.net(z)


def critic_portfolio_return(wc: torch.Tensor, x: torch.Tensor, R: torch.Tensor,
                            mask: torch.Tensor):
    """Adversarial portfolio excess return: r_c,t = mean_i (wc_t' x_i,t R_i,t).
    wc: (T,F) x: (T,N,F) R: (T,N) mask: (T,N) -> (T,)"""
    pr = torch.einsum("tf,tnf->tn", wc, x) * R
    pr = torch.where(mask, pr, torch.zeros_like(pr))
    cnt = mask.sum(dim=1).clamp(min=1)
    return pr.sum(dim=1) / cnt


def critic_alpha(M: torch.Tensor, wc: torch.Tensor, x: torch.Tensor,
                 R: torch.Tensor, mask: torch.Tensor):
    """Pricing error of the critic portfolio:
    alpha_c = mean_{t,i} M_i,t * (wc_t' x_i,t) * R_i,t  over valid obs.
    M: (T,N) wc: (T,F) x: (T,N,F) R: (T,N) mask: (T,N) -> scalar"""
    pr = torch.einsum("tf,tnf->tn", wc, x) * R
    pr = torch.where(mask, pr, torch.zeros_like(pr))
    n = mask.sum().clamp(min=1)
    return (M * pr).sum() / n


class MNet(nn.Module):
    """Weight network: z_t -> w(z_t) in R^F. SDF = 1 - w'x."""

    def __init__(self, state_dim: int, n_features: int,
                 hidden: tuple = (64, 64), dropout_p: float = 0.05):
        super().__init__()
        layers = []
        d_in = state_dim
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout_p)]
            d_in = h
        layers.append(nn.Linear(d_in, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        """z: (T, state_dim) -> w: (T, n_features)"""
        return self.net(z)


def sdf_values(w: torch.Tensor, x: torch.Tensor):
    """M_{i,t} = 1 - w_t' x_{i,t}
    w: (T, F)  x: (T, N, F) -> M: (T, N)"""
    return 1.0 - torch.einsum("tf,tnf->tn", w, x)


def pricing_errors(M: torch.Tensor, R: torch.Tensor, mask: torch.Tensor):
    """alpha_i = mean_t(M_{i,t} R^e_{i,t}) over available months (>=6).
    M: (T,N) R: (T,N) mask: (T,N) bool (finite & valid). Returns (N,) with
    NaN where insufficient data."""
    T, N = R.shape
    mr = M * R
    mr = torch.where(mask, mr, torch.zeros_like(mr))
    cnt = mask.sum(dim=0).clamp(min=1)
    alpha = mr.sum(dim=0) / cnt
    alpha = torch.where(cnt >= 6, alpha, torch.full_like(alpha, float("nan")))
    return alpha


# =========================================================================
# CPZ-faithful COMMON SDF (true replication of Chen, Pelger & Zhu 2024)
# -------------------------------------------------------------------------
# Official structure (code/torch_port/torch_version/models/models.py:65-87,
# losses.py):  SDF weights omega_t(i) = dense_net(concat(z_t, x_{i,t}))
#  ->  M_t = 1 + sum_i omega_t(i) * R^e_{t,i} / N_t * mean(N_t)
#  ->  alpha_i = (1/T_i) sum_t M_t * R^e_{t,i}   (ONE common M per time t)
#  ->  loss = mean_i (count_i / max_count) * alpha_i^2   (weighted, official)
# =========================================================================

class SDFNet(nn.Module):
    """CPZ weight network: (z_t, x_{i,t}) -> omega_t(i) in R (one scalar per stock).

    Faithful to the official DenseNetwork: input = concat(characteristics, state),
    hidden [64, 64] ReLU + dropout (keep 0.95), output_size = 1, NO output
    activation. The SDF itself is built in common_sdf().
    """

    def __init__(self, state_dim: int, n_features: int,
                 hidden: tuple = (64, 64), dropout_p: float = 0.05):
        super().__init__()
        layers = []
        d_in = state_dim + n_features
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout_p)]
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        """z: (T,S)  x: (T,N,F) -> omega: (T,N)"""
        zz = z.unsqueeze(1).expand(-1, x.shape[1], -1)      # (T,N,S)
        inp = torch.cat([x, zz], dim=-1)                    # (T,N,S+F)
        return self.net(inp).squeeze(-1)


class MomentsNet(nn.Module):
    """CPZ critic/moment network: (z_t, x_{i,t}) -> m_t(i) in R^K (K=8, tanh).

    The critic searches for K characteristic-conditional portfolios ("moments")
    on which the SDF prices worst; the SDF minimizes the same conditional
    pricing errors (alternating Adam, loss_factor 1.0). Faithful to the
    official MomentsModel (dense over concat(x, z), tanh output).
    """

    def __init__(self, state_dim: int, n_features: int, n_moments: int = 8,
                 hidden: tuple = (64, 64), dropout_p: float = 0.05):
        super().__init__()
        layers = []
        d_in = state_dim + n_features
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout_p)]
            d_in = h
        layers += [nn.Linear(d_in, n_moments), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        """z: (T,S)  x: (T,N,F) -> m: (T,N,K)"""
        zz = z.unsqueeze(1).expand(-1, x.shape[1], -1)
        inp = torch.cat([x, zz], dim=-1)
        return self.net(inp)


def common_sdf(omega: torch.Tensor, R: torch.Tensor, mask: torch.Tensor):
    """Common SDF (published CPZ equation):  M_t = 1 - (1/N_t) * sum_i omega_t(i) * R^e_{t,i}.

    omega: (T,N)  R: (T,N)  mask: (T,N) bool -> M: (T,)
    NOTE: no mean(N_t) rescale. The official code's '/N_t * mean(N_t)' cancels
    to ~1 when the cross-section is nearly constant (US data); with a variable
    cross-section it would blow up the SDF scale. The paper's equation is
    M_{t+1} = 1 - sum_i omega_t(i) R^e_{t+1,i} with omega absorbing 1/N_t.
    """
    wr = torch.where(mask, omega * R, torch.zeros_like(omega))
    n_t = mask.sum(dim=1).clamp(min=1).float()
    return 1.0 - wr.sum(dim=1) / n_t


def pricing_errors_common(M: torch.Tensor, R: torch.Tensor, mask: torch.Tensor):
    """alpha_i = (1/T_i) sum_t M_t * R^e_{t,i}  with a COMMON SDF M: (T,).

    Returns (N,) with NaN where fewer than 6 valid months (MIN_OBS_ALPHA)."""
    mr = torch.where(mask, M.unsqueeze(1) * R, torch.zeros_like(R))
    cnt = mask.sum(dim=0).clamp(min=1)
    alpha = mr.sum(dim=0) / cnt
    return torch.where(cnt >= 6, alpha, torch.full_like(alpha, float("nan")))


def weighted_pricing_loss(alpha: torch.Tensor, mask: torch.Tensor):
    """Official weighted loss: mean over valid stocks of (count_i/max_count)*alpha_i^2."""
    cnt = mask.sum(dim=0).float()
    valid = ~torch.isnan(alpha)
    a = alpha[valid]
    if a.numel() == 0:
        return torch.tensor(float("nan"), requires_grad=True)
    w = cnt[valid] / cnt[valid].max().clamp(min=1.0)
    return (w * a ** 2).mean()


def sdf_portfolio_return(omega: torch.Tensor, R: torch.Tensor, mask: torch.Tensor):
    """CPZ SDF-implied portfolio:  r_p,t = sum_i omega_t(i) R^e_{t,i} / sum_i |omega_t(i)|.

    Unit gross leverage by construction (comparable with leverage-normalized
    benchmark portfolios). omega: (T,N) R: (T,N) mask: (T,N) -> (T,)."""
    wr = torch.where(mask, omega * R, torch.zeros_like(omega))
    aw = torch.where(mask, omega.abs(), torch.zeros_like(omega))
    den = aw.sum(dim=1).clamp(min=1e-12)
    return wr.sum(dim=1) / den


def critic_moment_alphas(M: torch.Tensor, m: torch.Tensor, R: torch.Tensor,
                         mask: torch.Tensor):
    """Conditional pricing errors of the K critic portfolios:
    alpha_{k,i} = mean_t(m_{k,t,i} * M_t * R^e_{t,i}) -> (K, N) (NaN if <6 obs)."""
    mr = torch.where(mask, M.unsqueeze(1) * R, torch.zeros_like(R))  # (T,N)
    pr = torch.einsum("tnk,tn->tnk", m, mr)                          # (T,N,K)
    pr = torch.where(mask.unsqueeze(-1), pr, torch.zeros_like(pr))   # (T,N,K)
    cnt = mask.sum(dim=0).clamp(min=1)
    alpha_k = pr.permute(2, 0, 1).sum(dim=1) / cnt                   # (K,N)
    return torch.where(cnt >= 6, alpha_k, torch.full_like(alpha_k, float("nan")))


def sign_normalize(omega: torch.Tensor, rp: torch.Tensor):
    """Per-window sign guard for loadings interpretation: flip omega so that the
    SDF portfolio mean return is non-negative. Does NOT change the SDF itself
    (the squared-pricing-error loss pins the sign); only fixes the arbitrary
    sign of a converged local optimum for cross-sectional loadings."""
    if rp.mean() < 0:
        return -omega
    return omega
