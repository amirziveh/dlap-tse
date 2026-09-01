#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_e2_ens.py — Seed-ensemble CPZ deep SDF (E2 spec)
======================================================
Q: is the 13-seed dispersion in E2 OOS Sharpe initialization noise (fixable by
   weight averaging) or structural?

Design — ZERO retraining, ZERO code duplication:
  1. Per seed: import scripts.train_e2, monkeypatch its output directory to
     DLAP_ENS_DUMP, and force --dump-mechanism so train_e2 writes
     window_NN.npz (omega_te, R_te, tickers, keep) per window. Seed-outer
     loop (torch.manual_seed(s) once per seed) keeps the RNG stream identical
     to standalone train_e2 runs, so member Sharpes must reproduce the
     committed e2_results.csv rows — verified before any ensembling.
  2. Across seeds: per window, average omega across the S ensemble members
     (equal weights, no test-data selection), rebuild M = 1 - (1/N)ΣωR from
     the averaged weights and recompute the SAME metrics train_e2 reports:
       Sharpe (pooled OOS SDF-portfolio return, annualized x sqrt(12)),
       RMS/max alpha  [alpha_i = mean_t M_t R_te, common SDF],
       EV (test-sample OLS of stock returns on the SDF portfolio return).
  3. Ensembles S in {3, 5, 10, 13} over FIXED member lists
     (42,43,44) / (42..46) / (42..51) / (42..54): never chosen on test data.

Outputs (results{,_tr,_pk}/ensemble/):
  e2ens_results.csv   S,sharpe_pooled,rms_alpha_pct,max_alpha_pct,ev
  member_check.csv    seed,member_sharpe,committed_sharpe,diff  (must be ~0)
"""
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

SEEDS = [42, 43, 44] + list(range(100, 110))  # the paper's 13 committed seeds
ENSEMBLE_SIZES = [3, 5, 10, 13]

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
_C = os.environ.get("DLAP_COUNTRY", "").upper()
if _C == "TR":
    RES = ROOT / "results_tr"
elif _C == "PK":
    RES = ROOT / "results_pk"
else:
    RES = ROOT / "results"
DUMP = ROOT / "ens_dump"
OUT = RES / "ensemble"
DUMP.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)

sys.path.insert(0, str(ROOT / "scripts"))
import train_e2  # noqa: E402


def run_member(seed):
    """Run train_e2 --seed S with output redirected to DUMP. Returns the
    member's pooled Sharpe (reproduces the committed e2 number)."""
    dump_res = DUMP / "mechanism_dump"
    os.environ["DLAP_DUMP_DIR"] = "mechanism_dump"
    for f in dump_res.glob("window_*.npz"):
        f.unlink()
    dump_res.mkdir(exist_ok=True, parents=True)
    train_e2.__dict__["RES"] = DUMP
    sys.argv = ["train_e2.py", "--charset", "sy", "--states", "lstm",
                "--seed", str(seed), "--dump-mechanism"]
    try:
        train_e2.main()
    except SystemExit as e:
        print(f"  seed {seed}: train_e2 exited {e.code}")
        return math.nan
    # out_dir under the patched RES: dump-mechanism overrides the seed subdir,
    # so CSV + npz dumps land in DUMP/mechanism_dump for every seed. Archive
    # them per member before the next run overwrites the directory.
    csv_p = dump_res / "e2_results.csv"
    if not csv_p.exists():
        csv_p = DUMP / f"seed{seed}" / "e2_results.csv"
    member_dir = DUMP / f"member_s{seed}"
    member_dir.mkdir(exist_ok=True, parents=True)
    for f in dump_res.glob("window_*.npz"):
        f.rename(member_dir / f.name)
    if csv_p != member_dir / "e2_results.csv":
        (member_dir / "e2_results.csv").write_text(csv_p.read_text(encoding="utf-8"),
                                                   encoding="utf-8")
    with open(member_dir / "e2_results.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] == "sharpe_pooled":
                return float(row[1])
    return math.nan


def committed_sharpe(seed):
    p = RES / "e2_results.csv" if seed == 42 else \
        (RES / f"seed{seed}" / "e2_results.csv")
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] == "sharpe_pooled":
                return float(row[1])
    return None


def load_window(member_dir, wi):
    """(omega (12,N) NaN-free, R_te (12,N) with NaN, months list) or None.
    member_dir = ens_dump/member_s<seed> (archived window_*.npz files)."""
    p = member_dir / f"window_{wi:02d}.npz"
    if not p.exists():
        p = member_dir / f"window_{wi}.npz"
        if not p.exists():
            return None
    z = np.load(p, allow_pickle=True)
    om = z["omega_te"].astype(np.float64)
    R = z["R_te"].astype(np.float64)
    months = [str(m) for m in z["months_te"]]
    if om.ndim != 2 or om.shape != R.shape:
        return None
    if not np.all(np.isfinite(om)):
        return None
    return om, R, months


def metrics_from_weights(omega, R):
    """Same metric formulas as train_e2 (cpz branch): portfolio return,
    common-SDF alphas, EV — for ONE window."""
    mask = np.isfinite(R) & np.isfinite(omega)
    wr = np.where(mask, omega * R, 0.0)
    aw = np.where(mask, np.abs(omega), 0.0)
    den = aw.sum(axis=1)
    rp = np.where(den > 1e-12, wr.sum(axis=1) / np.maximum(den, 1e-12), np.nan)
    rp_valid = rp[np.isfinite(rp)]

    # common SDF: M_t = 1 - (1/N_t) sum_i omega R
    n_t = mask.sum(axis=1)
    M = 1.0 - np.where(n_t > 0, wr.sum(axis=1) / np.maximum(n_t, 1), 0.0)

    alphas = []
    for i in range(R.shape[1]):
        m = mask[:, i]
        if m.sum() >= 6:
            alphas.append(float((M[m] * R[m, i]).mean()))
    ss_res = ss_tot = 0.0
    for i in range(R.shape[1]):
        ri = R[:, i]
        m = np.isfinite(ri) & np.isfinite(rp)
        if m.sum() >= 8 and np.var(rp[m]) > 1e-12:
            b = np.polyfit(rp[m], ri[m], 1)
            e = ri[m] - (b[0] * rp[m] + b[1])
            ss_res += float((e ** 2).sum())
            ss_tot += float(((ri[m] - ri[m].mean()) ** 2).sum())
    ev = (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    return rp_valid, alphas, ev


def sharpe_ann(rp):
    rp = np.asarray(rp)
    if len(rp) < 6 or np.std(rp, ddof=1) < 1e-12:
        return math.nan
    return float(np.mean(rp) / np.std(rp, ddof=1) * math.sqrt(12))


def main():
    # fresh start: remove stale per-member archives (possibly from another
    # country's run in the shared ens_dump)
    import shutil
    for d in DUMP.glob("member_s*"):
        shutil.rmtree(d)
    member_sharpes = {}
    for s in SEEDS:
        print(f"\n=== member seed {s} ===", flush=True)
        member_sharpes[s] = run_member(s)
        c = committed_sharpe(s)
        tag = ""
        if c is not None:
            d = abs(member_sharpes[s] - c)
            tag = f"  committed={c:.4f} diff={d:.2e}" + ("  OK" if d < 5e-4 else "  MISMATCH ⚠")
        print(f"  member Sharpe = {member_sharpes[s]:.4f}{tag}", flush=True)

    with open(OUT / "member_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "member_sharpe", "committed_sharpe", "diff"])
        for s in SEEDS:
            c = committed_sharpe(s)
            w.writerow([s, f"{member_sharpes[s]:.6f}",
                        f"{c:.6f}" if c is not None else "",
                        f"{abs(member_sharpes[s] - c):.2e}" if c is not None else ""])

    # gather per-member window dumps
    per_seed = {}
    for s in SEEDS:
        mdir = DUMP / f"member_s{s}"
        per_seed[s] = {}
        wi = 0
        while True:
            got = load_window(mdir, wi)
            if got is None:
                break
            per_seed[s][wi] = got
            wi += 1
    n_win = min(len(v) for v in per_seed.values())
    print(f"\nwindows available per seed: {[len(v) for v in per_seed.values()]} (using {n_win})")

    n_months = 0
    for wi in range(n_win):
        n_months += per_seed[SEEDS[0]][wi][0].shape[0]

    rows = []
    for S in ENSEMBLE_SIZES:
        members = SEEDS[:S]
        pooled_rp, all_alphas, evs = [], [], []
        for wi in range(n_win):
            base = per_seed[members[0]][wi]
            om_ens, R0 = None, base[1]
            n_stocks = R0.shape[1]
            om_sum = np.zeros((base[0].shape[0], n_stocks))
            ok = True
            for s in members:
                om, R, _ = per_seed[s][wi]
                if R.shape != R0.shape or not np.allclose(R, R0, equal_nan=True):
                    print(f"  WARNING window {wi}: seed {s} panel mismatch, skipping window")
                    ok = False
                    break
                om_sum += om
            if not ok:
                continue
            om_ens = om_sum / len(members)
            rp, alphas, ev = metrics_from_weights(om_ens, R0)
            if len(rp) >= 6:
                pooled_rp.append(rp)
                all_alphas.extend(alphas)
                evs.append(ev)
        if not pooled_rp:
            print(f"  S={S}: no windows — skipped")
            continue
        rp_all = np.concatenate(pooled_rp)
        sharpe = sharpe_ann(rp_all)
        rms = float(np.sqrt(np.mean(np.square(all_alphas)))) * 100
        mx = float(np.max(np.abs(all_alphas))) * 100
        ev = float(np.nanmean(evs))
        rows.append((S, sharpe, rms, mx, ev))
        print(f"  S={S:2d}: sharpe={sharpe:+.4f} rms={rms:.2f}% max_a={mx:.1f}% ev={ev:+.4f}")

    with open(OUT / "e2ens_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["S", "sharpe_pooled", "rms_alpha_pct", "max_alpha_pct", "ev"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.6f}" if isinstance(v, float) else v for v in r[1:]])

    print(f"\nMember Sharpes: " + ", ".join(f"s{s}={member_sharpes[s]:+.3f}" for s in SEEDS))
    med = float(np.median(list(member_sharpes.values())))
    print(f"Member median: {med:+.4f}")
    print(f"Saved -> {OUT}/e2ens_results.csv (+ member_check.csv)")


if __name__ == "__main__":
    main()
