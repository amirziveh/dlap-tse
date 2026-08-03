#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_qfactors.py — DLAP-TSE Phase 1
=====================================
Constructs the HXZ (2015) q-factor model on TSE data:
  MKT  — market excess return (reused from fama-five factors_2x3.csv, same as FF5 benchmark)
  ME   — size factor
  IA   — investment factor (I/A)
  ROE  — profitability factor

Construction (HXZ standard, adapted to TSE):
  - Annual formation at formation_year fy (rebalance ~July), consistent with the
    DLAP-TSE characteristics convention: char known at fy, returns July(fy)..June(fy+1)
  - 2x3 sorts (value-weighted):
      size  = ln(mcap) at June of fy (median split)
      I/A   = investment from cbop_panel (30/70 terciles)
      ROE   = roe from anomaly_signals (30/70 terciles)
  - ME  = avg size spread over the two 2x3 sort sets
  - IA  = 1/2(S_H+B_H) - 1/2(S_L+B_L)   (size x I/A sorts)
  - ROE = 1/2(S_H+B_H) - 1/2(S_L+B_L)   (size x ROE sorts)
  - Breakpoints: all-stock (no NYSE equivalent on TSE — noted for the paper)

Output: data/factors_q.csv
"""
import csv
import os
import math
from collections import defaultdict
from pathlib import Path

FAMA = Path(os.environ.get("FAMA_ROOT", str(Path.home() / "research/fama-five/data")))
OUT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse"))) / "data"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_financial_tickers():
    fin = set()
    for row in read_csv(FAMA / "stock_universe.csv"):
        sector = row.get("sector_name", "") or ""
        if any(w in sector for w in ["بانک", "بیمه", "موسسات اعتباری", "نهادهای مالی"]):
            fin.add(row["ticker"])
    return fin


def load_returns_mcap():
    returns = defaultdict(dict)
    mcap = defaultdict(dict)
    for row in read_csv(FAMA / "processed" / "monthly_returns.csv"):
        t = row["ticker"]
        try:
            y, m = int(row["year"]), int(row["month"])
            returns[t][(y, m)] = float(row["ret_monthly"])
        except (ValueError, KeyError):
            continue
    for row in read_csv(FAMA / "processed" / "market_cap_monthly.csv"):
        t = row["ticker"]
        try:
            y, m = int(row["year"]), int(row["month"])
            mcap[t][(y, m)] = float(row["market_cap"])
        except (ValueError, KeyError):
            continue
    return returns, mcap


def load_chars():
    """fy -> ticker -> {'roe': v}"""
    roe = defaultdict(dict)
    for row in read_csv(FAMA / "mispricing" / "anomaly_signals.csv"):
        fy = row["formation_year"]
        if not fy:
            continue
        v = row.get("roe", "")
        roe[int(fy)][row["ticker"]] = float(v) if v not in ("", "None") else math.nan
    inv = defaultdict(dict)
    for row in read_csv(FAMA / "processed" / "cbop_panel.csv"):
        gy = row.get("gregorian_year", "")
        if not gy:
            continue
        v = row.get("investment", "")
        # same guard as build_characteristics.py: growth ratio > 50 is a
        # unit-mismatch artifact in the Rahavard data, not data
        fv = float(v) if v not in ("", "None") else math.nan
        if fv == fv and math.isfinite(fv) and fv > 50.0:
            fv = math.nan
        inv[int(gy) + 1][row["ticker"]] = fv
    return roe, inv


def formation_size(ticker, fy, mcap):
    """mcap at June of fy (fallback: latest Jan..Jun)"""
    c = mcap.get(ticker, {}).get((fy, 6))
    if c is not None and c > 0:
        return math.log(c)
    for m in range(5, 0, -1):
        c = mcap.get(ticker, {}).get((fy, m))
        if c is not None and c > 0:
            return math.log(c)
    return None


def tercile_break(vals):
    """30/70 tercile breakpoints (all-stock)"""
    s = sorted(v for v in vals if v is not None and math.isfinite(v))
    if len(s) < 6:
        return None, None
    lo = s[max(0, int(len(s) * 0.3) - 1)]
    hi = s[min(len(s) - 1, int(len(s) * 0.7))]
    return lo, hi


def vw_return(stocks, y, m, returns, mcap):
    rs, caps = [], []
    for t in stocks:
        r = returns.get(t, {}).get((y, m))
        if r is None or not math.isfinite(r):
            continue
        c = mcap.get(t, {}).get((y, m))
        if c is None or c <= 0:
            c = 1.0
        rs.append(r)
        caps.append(c)
    if not rs:
        return None
    tc = sum(caps)
    return sum(r * c for r, c in zip(rs, caps)) / tc


def main():
    fin = load_financial_tickers()
    returns, mcap = load_returns_mcap()
    roe_by_fy, inv_by_fy = load_chars()

    # MKT from existing FF5 factors (same benchmark market)
    mkt = {}
    for row in read_csv(FAMA / "factors" / "factors_2x3.csv"):
        mkt[(int(row["year"]), int(row["month"]))] = float(row["Mkt_RF"])

    fys = sorted(set(roe_by_fy) | set(inv_by_fy))
    fys = [f for f in fys if 2002 <= f <= 2026]

    out_rows = []
    cell_counts = {"ME_IA": 0, "IA": 0, "ROE": 0}
    for fy in fys:
        # --- eligible stocks with size + IA + ROE ---
        stocks = {}
        for t in set(roe_by_fy[fy]) | set(inv_by_fy[fy]):
            if t in fin:
                continue
            sz = formation_size(t, fy, mcap)
            if sz is None:
                continue
            stocks[t] = {"size": sz,
                         "ia": inv_by_fy[fy].get(t, math.nan),
                         "roe": roe_by_fy[fy].get(t, math.nan)}
        if len(stocks) < 12:
            continue

        size_med = sorted(s["size"] for s in stocks.values())
        size_med = size_med[len(size_med) // 2]

        # --- 2x3 sort: size x IA ---
        ia_vals = [s["ia"] for s in stocks.values() if math.isfinite(s["ia"])]
        ia_lo, ia_hi = tercile_break(ia_vals)
        cells_ia = {grp: [] for grp in ["SL", "SM", "SH", "BL", "BM", "BH"]}
        for t, s in stocks.items():
            if not math.isfinite(s["ia"]) or ia_lo is None:
                continue
            szg = "S" if s["size"] <= size_med else "B"
            if s["ia"] <= ia_lo:
                iaz = "L"
            elif s["ia"] >= ia_hi:
                iaz = "H"
            else:
                iaz = "M"
            cells_ia[szg + iaz].append(t)

        # --- 2x3 sort: size x ROE ---
        roe_vals = [s["roe"] for s in stocks.values() if math.isfinite(s["roe"])]
        roe_lo, roe_hi = tercile_break(roe_vals)
        cells_roe = {grp: [] for grp in ["SL", "SM", "SH", "BL", "BM", "BH"]}
        for t, s in stocks.items():
            if not math.isfinite(s["roe"]) or roe_lo is None:
                continue
            szg = "S" if s["size"] <= size_med else "B"
            if s["roe"] <= roe_lo:
                rz = "L"
            elif s["roe"] >= roe_hi:
                rz = "H"
            else:
                rz = "M"
            cells_roe[szg + rz].append(t)

        # --- monthly returns July(fy)..June(fy+1) ---
        for off in range(12):
            y = fy + (1 if off >= 6 else 0)
            m = 7 + off if off < 6 else off - 5
            if (y, m) not in mkt:
                continue
            # cell VW returns
            r_ia = {g: vw_return(stocks, y, m, returns, mcap) for g, stocks in cells_ia.items()}
            r_roe = {g: vw_return(stocks, y, m, returns, mcap) for g, stocks in cells_roe.items()}
            has = [v for v in list(r_ia.values()) + list(r_roe.values()) if v is not None]
            if len(has) < 6:
                continue
            # ME: avg size spread across both sort sets
            me_spreads = []
            for cells in (cells_ia, cells_roe):
                s_avg = [r_ia[g] for g in ("SL", "SM", "SH") if r_ia[g] is not None] if cells is cells_ia else \
                        [r_roe[g] for g in ("SL", "SM", "SH") if r_roe[g] is not None]
                b_avg = [r_ia[g] for g in ("BL", "BM", "BH") if r_ia[g] is not None] if cells is cells_ia else \
                        [r_roe[g] for g in ("BL", "BM", "BH") if r_roe[g] is not None]
                if s_avg and b_avg:
                    me_spreads.append(sum(s_avg) / len(s_avg) - sum(b_avg) / len(b_avg))
            me = sum(me_spreads) / len(me_spreads) if me_spreads else None
            # IA factor
            sh, bh, sl, bl = r_ia["SH"], r_ia["BH"], r_ia["SL"], r_ia["BL"]
            ia_f = None
            if all(v is not None for v in (sh, bh, sl, bl)):
                ia_f = 0.5 * (sh + bh) - 0.5 * (sl + bl)
            # ROE factor
            sh, bh, sl, bl = r_roe["SH"], r_roe["BH"], r_roe["SL"], r_roe["BL"]
            roe_f = None
            if all(v is not None for v in (sh, bh, sl, bl)):
                roe_f = 0.5 * (sh + bh) - 0.5 * (sl + bl)
            if me is not None:
                cell_counts["ME_IA"] += 1
            if ia_f is not None:
                cell_counts["IA"] += 1
            if roe_f is not None:
                cell_counts["ROE"] += 1
            out_rows.append({
                "formation_year": fy, "year": y, "month": m,
                "Mkt_RF": mkt[(y, m)],
                "ME": me if me is not None else "",
                "IA": ia_f if ia_f is not None else "",
                "ROE": roe_f if roe_f is not None else "",
            })

    out_rows.sort(key=lambda r: (r["year"], r["month"]))
    with open(OUT / "factors_q.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["formation_year", "year", "month",
                                          "Mkt_RF", "ME", "IA", "ROE"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"q-factor rows: {len(out_rows)} stock-months")
    print(f"  months with ME: {cell_counts['ME_IA']}, IA: {cell_counts['IA']}, ROE: {cell_counts['ROE']}")
    # sanity: mean and t-stats
    import statistics
    for fac in ["ME", "IA", "ROE"]:
        vals = [float(r[fac]) for r in out_rows if r[fac] != ""]
        if vals:
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            t = mu / (sd / math.sqrt(len(vals))) if sd > 0 else 0
            print(f"  {fac:<4} mean {mu * 100:7.3f}%/m  t={t:6.2f}  n={len(vals)}")
    print(f"\nSaved -> {OUT / 'factors_q.csv'}")


if __name__ == "__main__":
    main()
