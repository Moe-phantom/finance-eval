"""
demo_leakage.py — how much accuracy is manufactured by pipeline bugs alone.

The data here is a pure random walk. There is NO predictable signal, by
construction. Any accuracy above the base rate is an artifact.

We run the same model twice:
  A) the conventional notebook pipeline (shuffled split, scaler fit on the
     full series) — the setup used in most public LSTM stock notebooks
  B) the harness pipeline (chronological split, purge, train-only scaler)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from finance_eval import (
    SplitSpec, chronological_split, directional_accuracy,
    fit_scaler_on_train, audit, level_warning, benchmark_table, benchmark_verdict,
)

RNG = np.random.default_rng(7)
N_TICKERS, N_DAYS, LOOKBACK = 20, 1500, 50


def make_random_walk_panel() -> pd.DataFrame:
    """Geometric Brownian motion. Zero predictability, mild upward drift."""
    dates = pd.bdate_range("2015-01-01", periods=N_DAYS)
    frames = []
    for t in range(N_TICKERS):
        rets = RNG.normal(0.0004, 0.018, N_DAYS)      # ~10%/yr drift, 28% vol
        price = 100 * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({
            "date": dates,
            "ticker": f"TKR{t:02d}",
            "close": price,
            "volume": RNG.lognormal(15, 0.4, N_DAYS),
        }))
    return pd.concat(frames, ignore_index=True)


def build_windows(panel: pd.DataFrame):
    """Sliding windows of past returns -> next-day return. Standard setup."""
    Xs, ys, ds = [], [], []
    for _, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date")
        r = g["close"].pct_change().to_numpy()
        v = np.log(g["volume"].to_numpy())
        dates = g["date"].to_numpy()
        for i in range(LOOKBACK, len(g) - 1):
            Xs.append(np.concatenate([r[i - LOOKBACK + 1:i + 1], v[i - LOOKBACK + 1:i + 1]]))
            ys.append(r[i + 1])          # next-day return: the honest target
            ds.append(dates[i])
    return np.asarray(Xs), np.asarray(ys), pd.to_datetime(pd.Series(ds))


def pipeline_conventional(X, y):
    """Shuffled split + scaler fitted on everything. The usual notebook."""
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)                      # LEAK 1: sees the future
    idx = RNG.permutation(len(Xs))                    # LEAK 2: shuffled split
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    model = LogisticRegression(max_iter=1000)
    model.fit(Xs[tr], (y[tr] > 0).astype(int))
    pred = model.predict(Xs[te])
    return directional_accuracy(y[te], np.where(pred == 1, 1.0, -1.0), pred_kind="direction")


def pipeline_harness(X, y, dates, spec):
    """Chronological, purged, train-only scaling."""
    tr, te = chronological_split(dates, spec)
    Xtr, Xte, _ = fit_scaler_on_train(X, tr, te)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, (y[tr] > 0).astype(int))
    pred = model.predict(Xte)
    rep = directional_accuracy(y[te], np.where(pred == 1, 1.0, -1.0), pred_kind="direction")
    return rep, dates.iloc[te], y[te], np.where(pred == 1, 1.0, -1.0)


def demo_level_illusion(panel: pd.DataFrame):
    """Predicting the price LEVEL with 'tomorrow = today'."""
    g = panel[panel.ticker == "TKR00"].sort_values("date")
    price = g["close"].to_numpy()
    yt, yp = price[1:], price[:-1]
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, level_warning(yt)


if __name__ == "__main__":
    panel = make_random_walk_panel()
    X, y, dates = build_windows(panel)
    spec = SplitSpec(train_frac=0.7, horizon=1, embargo=10)

    print("=" * 74)
    print("DATA: pure random walk. True predictable signal = ZERO.")
    print(f"      {N_TICKERS} tickers x {N_DAYS} days -> {len(X):,} windows")
    print("=" * 74)

    print("\nA) CONVENTIONAL PIPELINE (shuffled split, scaler on full series)")
    print("  ", pipeline_conventional(X, y))

    print("\nB) HARNESS PIPELINE (chronological, purged, train-only scaler)")
    rep, d_te, y_te, p_te = pipeline_harness(X, y, dates, spec)
    print("  ", rep)
    tbl = benchmark_table(d_te, y_te, p_te, pred_kind="direction",
                          horizon=spec.horizon, n_boot=400)
    print("\n   Benchmark suite:")
    print("   " + tbl.to_string(index=False).replace("\n", "\n   "))
    v = benchmark_verdict(tbl)
    print(f"\n   Verdict: beats strongest baseline "
          f"({v['strongest_baseline']}, acc={v['strongest_baseline_acc']:.4f})? "
          f"{v['beats_strongest']}")


    print("\n" + "=" * 74)
    print("C) THE R-SQUARED ILLUSION: 'tomorrow's price = today's price'")
    r2, lw = demo_level_illusion(panel)
    print(f"   R^2 = {r2:.4f}   level heuristic flagged = {lw['flagged']}"
          f"   (lag-1 ac = {lw['median_ac1']:.3f})")
    print(f"   {lw['note']}")
    print("=" * 74)
