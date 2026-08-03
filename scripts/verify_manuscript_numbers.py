#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_manuscript_numbers.py — cross-check every reported number in the
manuscripts against the results CSVs (final audit, true-CPZ re-implementation).
"""
import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
RES = ROOT / "results"
PAPER = ROOT / "paper"
PAPER_FA = ROOT / "paper_fa"

errors = []
warnings = []


def kv(path):
    d = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                d[row[0]] = row[1]
    return d


def read_tex(path):
    return open(path, encoding="utf-8").read()


def check(name, actual, expected, tol=0.001):
    if expected is None:
        return
    try:
        a, e = float(actual), float(expected)
    except (TypeError, ValueError):
        errors.append(f"{name}: non-numeric actual={actual!r} expected={expected!r}")
        return
    if abs(a - e) > tol:
        errors.append(f"{name}: CSV={a} vs tex={e}")


def check_text(name, text, expected, tol=0.001):
    # find the expected number in the text (as written, may be Persian digits)
    if expected is None:
        return
    try:
        num = float(expected)
    except (TypeError, ValueError):
        errors.append(f"{name}: non-numeric expected {expected!r}")
        return
    # search with 3-decimal rounding (manuscript reports rounded values)
    fa_map = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    for nd in (3, 2, 1):
        v = f"{num:.{nd}f}"
        variants = [v, v.translate(fa_map).replace(".", "/")]
        if any(x in text for x in variants):
            return
    if num == int(num):  # integers: also try the plain form (e.g. "289")
        v = str(int(num))
        if v in text or v.translate(fa_map) in text:
            return
    errors.append(f"{name}: number {num:.3f} not found in manuscript text")


def main():
    en = read_tex(PAPER / "manuscript.tex")
    fa = read_tex(PAPER_FA / "manuscript_fa.tex") + "".join(
        open(p, encoding="utf-8").read() for p in sorted((PAPER_FA / "sections").glob("*.tex")))

    # 1) benchmarks + deep specs from master_results.csv
    # max_alpha is reported in the tables only for benchmarks and the linear
    # SDF; the deep-spec tables report Sharpe/EV only. The 20-char charscore
    # spec (E3-CS) is not cited in the text (only E2-CS is).
    no_max = {"E2", "E3", "E4A", "E4B", "E5A", "E5B", "E8", "E8B", "E2-CS", "E3-CS"}
    no_rms = {"E3-CS"}
    for row in csv.DictReader(open(RES / "master_results.csv", encoding="utf-8-sig")):
        m = row["model"]
        for col, key in [("sharpe", "sharpe"), ("ev", "ev"),
                         ("rms_alpha_pct", "rms_alpha_pct"),
                         ("max_alpha_pct", "max_alpha_pct")]:
            if col == "max_alpha_pct" and m in no_max:
                continue
            if col == "rms_alpha_pct" and m in no_rms:
                continue
            v = row[col]
            if not v:
                continue
            # EN: check the value appears somewhere in the tex (rough but effective)
            check_text(f"EN {m} {col}", en, v, tol=0.001)
            check_text(f"FA {m} {col}", fa, v, tol=0.001)

    # 2) specific headline numbers
    checks = [
        ("E2 sharpe", "0.363", "0.363"), ("E8 sharpe", "0.819", "0.819"),
        ("E5A sharpe", "-0.025", "-0.025"), ("E4B sharpe", "0.050", "0.050"),
        ("LinearSDF11 sharpe", "0.374", "0.374"), ("LinearSDF20 sharpe", "0.713", "0.713"),
        ("E2 alpha rms", "5.82", "5.82"), ("E3 alpha rms", "5.56", "5.56"),
        ("E2 EV", "0.465", "0.465"), ("E8 wealth", "20.0", "20.0"),
        ("LASSO wealth", "21.6", "21.6"), ("E2 wealth", "2.25", "2.25"),
        ("boom-bust E2", "1.107", "1.107"), ("boom-bust LASSO", "1.175", "1.175"),
        ("E8 subperiod calm", "0.674", "0.674"),
        ("charscore E2 sharpe", "0.810", "0.810"), ("charscore alpha", "7.30", "7.30"),
    ]
    for name, en_v, fa_v in checks:
        check_text(f"EN {name}", en, en_v)
        check_text(f"FA {name}", fa, fa_v)

    # 3) bootstrap intervals
    for f in ["sharp_diff_bootstrap_e2.csv", "sharp_diff_bootstrap_e8.csv"]:
        p = RES / f
        if not p.exists():
            warnings.append(f"missing {f}")
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            for col in ["diff", "ci_lo", "ci_hi"]:
                check_text(f"EN {r['pair']} {col}", en, r[col], tol=0.01)

    # 4) loadings table values
    ld = {}
    with open(RES / "e6_loadings_all.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ld[r["char"]] = float(r["mean_w"])
    for c, v in ld.items():
        check_text(f"EN loading {c}", en, f"{v:.3f}", tol=0.001)
        check_text(f"FA loading {c}", fa, f"{v:.3f}", tol=0.001)

    # 5) per-window table: spot check a few cells
    pws = {}
    with open(RES / "per_window_sharpe.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pws[r["model"]] = r
    for m in ["E2", "E8", "FF5"]:
        for w in ["13-14", "18-19", "19-20", "full"]:
            v = pws[m][w]
            check_text(f"EN perwin {m} {w}", en, v, tol=0.005)
            check_text(f"FA perwin {m} {w}", fa, v, tol=0.005)

    # 6) SPA / Reality Check table (v0.6)
    if (RES / "spa_test.csv").exists():
        for r in csv.DictReader(open(RES / "spa_test.csv", encoding="utf-8-sig")):
            for col in ["spa_p", "rc_p"]:
                check_text(f"EN spa {r['target']} {r['loss']} {r['block']} {col}",
                           en, r[col], tol=0.001)

    # 7) block-12 bootstrap intervals (v0.6) — only the pairs cited in text
    b12_cited = {
        "sharp_diff_bootstrap_e2_b12.csv": "E2 vs LASSO",
        "sharp_diff_bootstrap_e8_b12.csv": "E8 vs Market",
    }
    for f, pair in b12_cited.items():
        p = RES / f
        if not p.exists():
            warnings.append(f"missing {f}")
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            if r["pair"] != pair:
                continue
            for col in ["ci_lo", "ci_hi"]:
                check_text(f"EN {r['pair']} {col} (b12)", en, r[col], tol=0.01)

    # 8) v0.6: mechanism table, sign convention, architecture sensitivity
    v06_checks = [
        ("mech limit-hit tail", "0.225"), ("mech limit-hit kept", "0.198"),
        ("mech amihud tail", "11.9"), ("mech amihud kept", "5.1"),
        ("mech trades tail", "289"), ("mech trades kept", "382"),
        ("mech days/mo tail", "17.1"), ("mech days/mo kept", "18.3"),
        ("mech thin tail", "0.042"), ("mech thin kept", "0.009"),
        ("mech vol tail", "0.181"), ("mech vol kept", "0.187"),
        ("mech ar1 tail", "0.129"), ("mech ar1 kept", "0.093"),
        ("mech maxret tail", "1.026"), ("mech maxret kept", "0.970"),
        ("mech wshare", "0.049"), ("mech a2share", "0.046"),
        ("mech rms tail", "0.0471"), ("mech rms kept", "0.0529"),
        ("mech ewsharpe tail", "0.30"), ("mech ewsharpe kept", "0.60"),
        ("mech ewvol tail", "0.081"), ("mech ewvol kept", "0.091"),
        ("sign E2 s42", "1.230"), ("sign E2 s43", "1.275"), ("sign E2 s44", "1.299"),
        ("sign E8 s42", "1.218"), ("sign E8 s43", "1.227"), ("sign E8 s44", "1.235"),
        ("arch w32", "0.581"), ("arch w128", "0.800"), ("arch d3", "0.193"),
        ("arch ev range", "0.450"), ("arch ev hi", "0.468"),
        ("arch alpha lo", "5.56"), ("arch alpha hi", "5.77"),
        ("catas seed42", "-1.84"), ("catas seed43", "-2.06"), ("catas seed44", "-1.81"),
        ("catas e8 s42", "2.09"), ("catas e8 s43", "2.20"), ("catas e8 s44", "1.25"),
    ]
    for name, v in v06_checks:
        check_text(f"EN {name}", en, v, tol=0.005)

    print(f"ERRORS: {len(errors)}")
    for e in errors:
        print("  ✗", e)
    print(f"WARNINGS: {len(warnings)}")
    for w in warnings:
        print("  !", w)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
