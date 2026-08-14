"""
demo_dependence.py — why the binomial test was wrong, and what block length does.

Two effects, measured separately on data with no conditional signal:

  1. Cross-sectional dependence: 20 tickers on one market day are not 20
     independent observations. Date-level resampling fixes this exactly.

  2. Overlapping labels: with an h-day target, adjacent dates share h-1 days
     of their return window. Block length must exceed h to retain that
     structure. Date-level resampling with L=1 does NOT fix this.
"""

import numpy as np
import pandas as pd
from scipy import stats

from finance_eval import bootstrap_edge

RNG = np.random.default_rng(42)
N_DAYS, N_TICK = 400, 20


def market_panel(beta=0.9):
    """One market factor + small idiosyncratic noise => strong same-day dependence."""
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=N_DAYS), N_TICK)
    mkt = RNG.normal(0.0004, 0.012, N_DAYS)
    idio = RNG.normal(0, 0.012 * np.sqrt(1 - beta**2) / beta, (N_DAYS, N_TICK))
    y = (np.repeat(mkt, N_TICK).reshape(N_DAYS, N_TICK) * beta + idio).ravel()
    return dates, y


def overlapping_labels(h):
    """Target = cumulative return over the next h days. Adjacent dates overlap."""
    dates_u = pd.bdate_range("2020-01-01", periods=N_DAYS + h)
    r = RNG.normal(0.0004, 0.012, N_DAYS + h)
    y = np.array([r[i:i + h].sum() for i in range(N_DAYS)])
    return np.asarray(dates_u[:N_DAYS]), y


def ci_width(dates, y, pred, base, L, h):
    r = bootstrap_edge(dates, y, pred, base, pred_kind="direction",
                       horizon=h, mean_block_len=L, n_boot=600, seed=1)
    return r.ci_high - r.ci_low


if __name__ == "__main__":
    print("=" * 78)
    print("1. CROSS-SECTIONAL DEPENDENCE  (400 days x 20 tickers, one market factor)")
    print("=" * 78)
    dates, y = market_panel()
    pred = RNG.choice([-1.0, 1.0], size=y.size)
    base = np.ones_like(y)

    hits_m = (np.sign(y) == pred)
    hits_b = (np.sign(y) == base)
    diff = hits_m.astype(int) - hits_b.astype(int)
    n = len(y)

    # what the binomial test implicitly assumes
    se_iid = np.sqrt(diff.var() / n)
    w_iid = 2 * 1.96 * se_iid
    w_date = ci_width(dates, y, pred, base, 1, 1)
    w_blk = ci_width(dates, y, pred, base, 10, 1)

    print(f"  observations: {n:,} rows across {N_DAYS} dates")
    print(f"  CI width assuming iid rows (what binomtest implies): {w_iid:.4f}")
    print(f"  CI width, date resampling  (L=1) :                   {w_date:.4f}"
          f"   ({w_date/w_iid:.1f}x wider)")
    print(f"  CI width, block resampling (L=10):                   {w_blk:.4f}"
          f"   ({w_blk/w_iid:.1f}x wider)")
    print("\n  => treating rows as independent understates uncertainty.")

    print("\n" + "=" * 78)
    print("2. OVERLAPPING LABELS  (400 dates, 1 series, cumulative h-day target)")
    print("=" * 78)
    print(f"  {'horizon':>8} {'L=1':>10} {'L=2h':>10} {'L=4h':>10}   (CI width)")
    for h in (1, 5, 20):
        d, y2 = overlapping_labels(h)
        p2 = RNG.choice([-1.0, 1.0], size=y2.size)
        b2 = np.ones_like(y2)
        w1 = ci_width(d, y2, p2, b2, 1, h)
        w2 = ci_width(d, y2, p2, b2, 2 * h, h)
        w4 = ci_width(d, y2, p2, b2, 4 * h, h)
        print(f"  {h:>8} {w1:>10.4f} {w2:>10.4f} {w4:>10.4f}")

    print("\n  => with h=1 there is no overlap and block length barely matters.")
    print("     As h grows, L=1 understates uncertainty relative to L>h.")
    print("     Block length must exceed the label horizon to retain the")
    print("     overlap structure — it reduces the bias, it does not remove it.")
