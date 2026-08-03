#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_e2.py — DLAP-TSE Phase 3/4: E2–E5 deep SDF training & evaluation
=======================================================================
Rolling-window protocol identical to E1 (train 48 / valid 12 / test 12 (60-month lookback),
common period 2008-07..2026-06) so results are directly comparable.

Experiments:
  E2  --charset sy   --states lstm            (11 SY signals, macro states)
  E3  --charset all  --states lstm            (20 chars, macro states)
  E4a --charset all  --states const           (20 chars, NO macro conditioning)
  E4b --charset sy   --states const           (11 chars, NO macro conditioning)
  E5a --charset sy   --states lstm --critic   (adversarial critic ON)
  E5b --charset all  --states lstm --critic

Per window:
  1. Normalize macro with train mean/std; impute missing (ffill, then 0)
  2. Train Z_net (LSTM 6->4, or learned constant) + M_net (z -> w) jointly on
     squared-pricing-error loss  mean_i (E_t[M R^e])^2
  3. E5: CriticNet z -> portfolio weights (tanh) maximizes (E[M r_c])^2;
     SDF minimizes it too (alternating Adam, loss_factor 1.0)
  4. Early stopping on validation loss (patience 25, max 400 epochs)
  5. Test: M = 1 - w(z)'x, SDF portfolio return r_p,t = sum_i M_i R^e_i / sum_i M_i

Outputs: results/{e2,e3,e4a,e4b,e5a,e5b}_results.csv + _pooled_series.csv
"""
import argparse
import csv
import os
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_core import load_npz, load_rf, load_factors_ff5, load_factors_q, \
    sharpe_ann, rolling_windows
from sdf_models import ZNet, ConstZNet, MNet, CriticNet, sdf_values, \
    pricing_errors, critic_alpha

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

torch.manual_seed(42)

SY_INDICES = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
STATE_DIM = 4
LR = 1e-3
MAX_EPOCHS = 400
PATIENCE = 25
MIN_OBS_ALPHA = 6
LOSS_FACTOR = 1.0  # critic term weight in the SDF loss (official loss_factor)


def load_data():
    arr, dates, variables, tickers = load_npz()
    rf = load_rf()
    ff5 = load_factors_ff5()
    q = load_factors_q()
    common = [d for d in dates if (int(d[:4]), int(d[5:7])) in ff5 and
              (int(d[:4]), int(d[5:7])) in q]
    pos = {d: i for i, d in enumerate(dates)}
    idx = [pos[d] for d in common]
    R = arr[idx, :, 0].astype(float)
    R[R < -50] = np.nan
    rf_v = np.array([rf.get(d, np.nan) for d in common])
    R_exc = R - rf_v[:, None]
    X = arr[idx, :, 1:].astype(float)
    X[X < -50] = np.nan
    M = np.load(DATA / "Macro_all.npz", allow_pickle=True)
    macro = M["data"].astype(float)[idx, :]
    for s in range(macro.shape[1]):
        col = macro[:, s]
        prev = None
        for t in range(len(col)):
            if math.isfinite(col[t]):
                prev = col[t]
            else:
                col[t] = prev if prev is not None else 0.0
    return R_exc, X, macro, common


def make_tensors(R, X):
    mask = np.isfinite(R) & np.isfinite(X).all(axis=2)
    return (torch.from_numpy(np.nan_to_num(R, nan=0.0)).float(),
            torch.from_numpy(np.nan_to_num(X, nan=0.0)).float(),
            torch.from_numpy(mask))


def train_window(R_tr, X_tr, macro_tr, R_va, X_va, macro_va, n_features,
                 states="lstm", critic=False):
    R_tr_t, X_tr_t, mask_tr = make_tensors(R_tr, X_tr)
    R_va_t, X_va_t, mask_va = make_tensors(R_va, X_va)
    mu = macro_tr.mean(axis=0)
    sd = macro_tr.std(axis=0) + 1e-12
    mac_tr = torch.from_numpy((macro_tr - mu) / sd).float()
    mac_va = torch.from_numpy((macro_va - mu) / sd).float()

    znet = ZNet(macro_dim=macro_tr.shape[1], state_dim=STATE_DIM) if states == "lstm" \
        else ConstZNet(STATE_DIM)
    mnet = MNet(state_dim=STATE_DIM, n_features=n_features)
    opt_s = torch.optim.Adam(list(znet.parameters()) + list(mnet.parameters()), lr=LR)
    cnet = None
    opt_c = None
    if critic:
        cnet = CriticNet(STATE_DIM, n_features)
        opt_c = torch.optim.Adam(cnet.parameters(), lr=LR)

    def sdf_loss(z, X, R, mask, use_critic=True):
        M = sdf_values(mnet(z), X)
        alpha = pricing_errors(M, R, mask)
        loss = torch.nanmean(alpha ** 2)
        if use_critic and cnet is not None:
            with torch.no_grad():
                wc = cnet(z)
            loss = loss + LOSS_FACTOR * critic_alpha(M, wc, X, R, mask) ** 2
        return loss, M

    best_val = float("inf")
    best_state = None
    patience = 0
    epochs_used = 0
    for epoch in range(MAX_EPOCHS):
        # --- critic step (maximize its portfolio's squared pricing error) ---
        if cnet is not None:
            cnet.train()
            z_tr = znet(mac_tr)
            with torch.no_grad():
                M_tr = sdf_values(mnet(z_tr), X_tr_t)
            wc = cnet(z_tr.detach())
            alpha_c = critic_alpha(M_tr, wc, X_tr_t, R_tr_t, mask_tr)
            loss_c = -(alpha_c ** 2)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        # --- SDF step (minimize pricing errors incl. critic portfolio) ---
        znet.train(); mnet.train()
        z_tr = znet(mac_tr)
        loss, _ = sdf_loss(z_tr, X_tr_t, R_tr_t, mask_tr, use_critic=critic)
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()
        epochs_used = epoch + 1

        # --- validation ---
        znet.eval(); mnet.eval()
        if cnet is not None:
            cnet.eval()
        with torch.no_grad():
            z_va = znet(mac_va)
            val_loss, _ = sdf_loss(z_va, X_va_t, R_va_t, mask_va, use_critic=critic)
            val_loss = val_loss.item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = ({k: v.clone() for k, v in znet.state_dict().items()},
                          {k: v.clone() for k, v in mnet.state_dict().items()},
                          {k: v.clone() for k, v in cnet.state_dict().items()}
                          if cnet is not None else None)
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("training failed")
    znet.load_state_dict(best_state[0])
    mnet.load_state_dict(best_state[1])
    if cnet is not None:
        cnet.load_state_dict(best_state[2])
    return znet, mnet, cnet, best_val, epochs_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charset", choices=["sy", "all"], default="sy")
    ap.add_argument("--states", choices=["lstm", "const"], default="lstm")
    ap.add_argument("--critic", action="store_true")
    ap.add_argument("--liq-filter", action="store_true",
                    help="E8: drop stocks in the bottom 5% of mean train-window turnover")
    args = ap.parse_args()

    if args.liq_filter:
        out_name = "e8" if args.charset == "sy" else "e8b"
    elif args.critic:
        out_name = "e5a" if args.charset == "sy" else "e5b"
    elif args.states == "const":
        out_name = "e4b" if args.charset == "sy" else "e4a"
    else:
        out_name = "e2" if args.charset == "sy" else "e3"

    feat_idx = SY_INDICES if args.charset == "sy" else list(range(20))
    n_features = len(feat_idx)
    print(f"== {out_name.upper()}: deep SDF, {n_features} chars, states="
          f"{args.states}, critic={args.critic} ==")

    R_exc, X, macro, common = load_data()
    X_full = X  # keep full 20-char array for the liquidity filter
    T = len(common)
    windows = list(rolling_windows(list(range(T)), train=60, test=12))
    print(f"  {len(windows)} windows, period {common[0]}..{common[-1]}")

    X = np.clip(X[:, :, feat_idx], -10.0, 10.0)
    turnover_full = X_full[:, :, 2]  # turnover = char index 2 (full space)

    pooled_rp, all_alphas, per_win_sharpe = [], [], []
    critic_alphas = []
    n_windows = 0
    for wi, (w_tr, w_te) in enumerate(windows):
        w_va = w_tr[-12:]
        w_tr = w_tr[:-12]
        R_tr, X_tr, mac_tr = R_exc[w_tr], X[w_tr], macro[w_tr]
        R_va, X_va, mac_va = R_exc[w_va], X[w_va], macro[w_va]
        R_te, X_te, mac_te = R_exc[w_te], X[w_te], macro[w_te]

        keep = np.isfinite(R_tr).sum(axis=0) >= 12
        if args.liq_filter:
            tr_turn = np.nanmean(turnover_full[w_tr][:, keep], axis=0)
            thr = np.nanpercentile(tr_turn, 5)
            keep = keep.copy()
            keep[keep] = keep[keep] & (tr_turn > thr)
        R_tr, X_tr = R_tr[:, keep], X_tr[:, keep]
        R_va, X_va = R_va[:, keep], X_va[:, keep]
        R_te, X_te = R_te[:, keep], X_te[:, keep]
        if R_tr.shape[1] < 50:
            print(f"  window {wi}: skipped ({R_tr.shape[1]} stocks)")
            continue

        znet, mnet, cnet, val_loss, epochs_used = train_window(
            R_tr, X_tr, mac_tr, R_va, X_va, mac_va, n_features,
            states=args.states, critic=args.critic)
        print(f"  window {wi} [{common[w_te[0]]}..{common[w_te[-1]]}]: "
              f"val_loss={val_loss:.2e} epochs={epochs_used}")

        znet.eval(); mnet.eval()
        if cnet is not None:
            cnet.eval()
        mu = mac_tr.mean(axis=0); sd = mac_tr.std(axis=0) + 1e-12
        mac_all = torch.from_numpy(
            (np.concatenate([mac_tr, mac_va, mac_te]) - mu) / sd).float()
        X_all = np.concatenate([X_tr, X_va, X_te], axis=0)
        R_all = np.concatenate([R_tr, R_va, R_te], axis=0)
        X_all_t = torch.from_numpy(np.nan_to_num(X_all, nan=0.0)).float()
        R_all_t = torch.from_numpy(np.nan_to_num(R_all, nan=0.0)).float()
        with torch.no_grad():
            z_all = znet(mac_all)
            M_all = sdf_values(mnet(z_all), X_all_t).numpy()
            if cnet is not None:
                wc_all = cnet(z_all).numpy()
        M_te = M_all[-len(w_te):]
        mask_te = np.isfinite(R_te) & np.isfinite(X_te).all(axis=2)
        num = np.where(mask_te, M_te * R_te, 0.0).sum(axis=1)
        den = np.where(mask_te, M_te, 0.0).sum(axis=1)
        rp = np.where(den != 0, num / np.where(den != 0, den, 1.0), np.nan)
        rp = rp[np.isfinite(rp)]
        alphas = [float((M_te[mask_te[:, i], i] * R_te[mask_te[:, i], i]).mean())
                  for i in range(R_te.shape[1])
                  if mask_te[:, i].sum() >= MIN_OBS_ALPHA]
        if cnet is not None:
            wc_te = wc_all[-len(w_te):]
            pr = (wc_te[:, None, :] * X_te).sum(axis=2) * R_te  # (12, N)
            pr = np.where(mask_te, pr, 0.0)
            n_ok = mask_te.sum()
            if n_ok >= 6:
                alpha_c = float((M_te * pr).sum() / n_ok)
                critic_alphas.append(alpha_c)
        if len(rp) >= 6:
            pooled_rp.append(rp)
            per_win_sharpe.append(sharpe_ann(rp))
            all_alphas.extend(alphas)
            n_windows += 1

    if not pooled_rp:
        print("FATAL: no windows completed")
        sys.exit(1)
    rp_all = np.concatenate(pooled_rp)
    sharpe = sharpe_ann(rp_all)
    rms_alpha = float(np.sqrt(np.mean(np.square(all_alphas)))) * 100
    max_alpha = float(np.max(np.abs(all_alphas))) * 100
    critic_alpha_pct = float(np.mean(np.abs(critic_alphas))) * 100 if critic_alphas else math.nan

    # XS explained-variation (same construction as E1 EV)
    ss_res = ss_tot = 0.0
    n_ev = 0
    for wi in range(n_windows):
        w_tr, w_te = windows[wi]
        keep = np.isfinite(R_exc[w_tr[:-12]]).sum(axis=0) >= 12
        R_te_w = R_exc[w_te][:, keep]
        rp_w = pooled_rp[wi]
        for i in range(R_te_w.shape[1]):
            ri = R_te_w[:, i]
            m = np.isfinite(ri) & np.isfinite(rp_w)
            if m.sum() < 8 or np.var(rp_w[m]) < 1e-12:
                continue
            b = np.polyfit(rp_w[m], ri[m], 1)
            e = ri[m] - (b[0] * rp_w[m] + b[1])
            ss_res += float((e ** 2).sum())
            ss_tot += float(((ri[m] - ri[m].mean()) ** 2).sum())
            n_ev += 1
    ev = (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    print(f"\n{out_name.upper()} pooled OOS: n_windows={n_windows} sharpe={sharpe:.3f} "
          f"EV={ev:.4f} rms_alpha={rms_alpha:.3f}% max_alpha={max_alpha:.3f}%"
          + (f" critic_alpha={critic_alpha_pct:.3f}%" if args.critic else ""))

    # ── save ───────────────────────────────────────────────────────────
    out_csv = RES / f"{out_name}_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in [("model", out_name.upper()), ("charset", args.charset),
                     ("states", args.states), ("critic", args.critic),
                     ("n_features", n_features),
                     ("n_windows", n_windows), ("n_oos_months", len(rp_all)),
                     ("sharpe_pooled", f"{sharpe:.4f}"),
                     ("sharpe_mean_win", f"{float(np.mean(per_win_sharpe)):.4f}"),
                     ("ev", f"{ev:.4f}"),
                     ("rms_alpha_pct", f"{rms_alpha:.4f}"),
                     ("max_alpha_pct", f"{max_alpha:.4f}"),
                     ("critic_alpha_pct", f"{critic_alpha_pct:.4f}" if critic_alphas else "")]:
            w.writerow([k, v])
    with open(RES / f"{out_name}_pooled_series.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["oos_return"])
        for v in rp_all:
            w.writerow([f"{v:.6f}"])

    # ── compare with E1 + prior runs ───────────────────────────────────
    print("\n" + "=" * 88)
    print(f"{'model':<11}{'Sharpe':>9}{'EV':>9}{'RMS_alpha%':>11}{'max_alpha%':>11}")
    print("-" * 88)
    print(f"{out_name.upper():<11}{sharpe:>9.3f}{ev:>9.4f}{rms_alpha:>11.3f}{max_alpha:>11.3f}")
    for fname in ["e2_results.csv", "e3_results.csv", "e4a_results.csv",
                  "e4b_results.csv", "e5a_results.csv", "e5b_results.csv"]:
        p = RES / fname
        if not p.exists() or fname == f"{out_name}_results.csv":
            continue
        d = {r[0]: r[1] for r in csv.reader(open(p, encoding="utf-8-sig"))}
        print(f"{d['model']:<11}{float(d['sharpe_pooled']):>9.3f}{float(d['ev']):>9.4f}"
              f"{float(d['rms_alpha_pct']):>11.3f}{float(d['max_alpha_pct']):>11.3f}")
    e1 = {}
    if (RES / "e1_benchmarks.csv").exists():
        for r in csv.DictReader(open(RES / "e1_benchmarks.csv", encoding="utf-8-sig")):
            e1[r["name"]] = r
        for m in ["Market", "FF5", "q-factor", "PCA(5)", "LASSO"]:
            if m in e1:
                r = e1[m]
                print(f"{m:<11}{float(r['sharpe_pooled']):>9.3f}{float(r['ev']):>9.4f}"
                      f"{float(r['rms_alpha_pct']):>11.3f}{float(r['max_alpha_pct']):>11.3f}")
    print("=" * 88)
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
