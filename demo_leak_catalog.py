"""
demo_leak_catalog.py — which bug produces which inflated number.

Same random-walk data throughout: true predictable signal is ZERO, so the
honest answer is always "edge = 0". Each condition introduces one specific
bug and we measure how much accuracy it manufactures.

Use this as a lookup table. If a reported number matches a row here, that
row names the likely cause.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from finance_eval import (
    SplitSpec, chronological_split, directional_accuracy, fit_scaler_on_train,
)

RNG = np.random.default_rng(11)
N_TICKERS, N_DAYS, LOOKBACK = 20, 1500, 50
SPEC = SplitSpec(train_frac=0.7, horizon=1, embargo=10)


def panel_random_walk(drift=0.0004, vol=0.018):
    dates = pd.bdate_range("2015-01-01", periods=N_DAYS)
    out = []
    for t in range(N_TICKERS):
        r = RNG.normal(drift, vol, N_DAYS)
        out.append(pd.DataFrame({
            "date": dates, "ticker": f"TKR{t:02d}",
            "close": 100 * np.exp(np.cumsum(r)),
        }))
    return pd.concat(out, ignore_index=True)


def windows(panel, off_by_one=False):
    """off_by_one=True lets the window include the target day (classic bug)."""
    Xs, ys, ds = [], [], []
    for _, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date")
        r = g["close"].pct_change().to_numpy()
        d = g["date"].to_numpy()
        for i in range(LOOKBACK, len(g) - 1):
            end = i + 2 if off_by_one else i + 1     # +2 swallows the target
            Xs.append(r[end - LOOKBACK:end])
            ys.append(r[i + 1])
            ds.append(d[i])
    return np.asarray(Xs), np.asarray(ys), pd.to_datetime(pd.Series(ds))


def run(X, y, dates, shuffle=False, global_scale=False):
    if global_scale:
        X = MinMaxScaler().fit_transform(X)
    if shuffle:
        idx = RNG.permutation(len(X))
        cut = int(0.7 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        Xtr, Xte = X[tr], X[te]
    else:
        tr, te = chronological_split(dates, SPEC)
        Xtr, Xte = (X[tr], X[te]) if global_scale else fit_scaler_on_train(X, tr, te)[:2]
    m = LogisticRegression(max_iter=1000).fit(Xtr, (y[tr] > 0).astype(int))
    p = m.predict(Xte)
    return directional_accuracy(y[te], np.where(p == 1, 1.0, -1.0), pred_kind="direction")


def row(name, rep, note):
    base = max(rep.up_share, 1 - rep.up_share)      # strongest constant baseline
    return {
        "condition": name,
        "accuracy": f"{rep.accuracy:.3f}",
        "best_const_baseline": f"{base:.3f}",
        "edge": f"{rep.accuracy - base:+.3f}",
        "diagnosis": note,
    }


if __name__ == "__main__":
    panel = panel_random_walk()
    X, y, d = windows(panel)
    rows = []

    rows.append(row("0. Clean harness", run(X, y, d),
                    "correct — no edge, as expected"))

    rows.append(row("1. Shuffled split", run(X, y, d, shuffle=True),
                    "inflates little for return targets"))

    rows.append(row("2. Global min-max scaling", run(X, y, d, global_scale=True),
                    "inflates little on its own"))

    Xo, yo, do = windows(panel, off_by_one=True)
    rows.append(row("3. Off-by-one window", run(Xo, yo, do),
                    "TARGET IN FEATURES — large, obvious inflation"))

    # 4. bull-market sample, accuracy reported without its base rate
    bull = panel_random_walk(drift=0.0016, vol=0.012)   # strong trend, low vol
    Xb, yb, db = windows(bull)
    rb = run(Xb, yb, db)
    _b = max(rb.up_share, 1 - rb.up_share)
    rows.append({
        "condition": "4. Bull sample, no baseline shown",
        "accuracy": f"{rb.accuracy:.3f}",
        "best_const_baseline": f"{_b:.3f}",
        "edge": f"{rb.accuracy - _b:+.3f}",
        "diagnosis": "accuracy looks high; all of it is the baseline",
    })

    df = pd.DataFrame(rows)
    print("\nRandom-walk data. True edge is ZERO in every row.\n")
    print(df.to_string(index=False))
    print("\nRead the 'edge' column, never the 'accuracy' column.")
print("Baseline here is the best CONSTANT rule; use benchmark_verdict() for the full suite.\n")
