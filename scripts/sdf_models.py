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
