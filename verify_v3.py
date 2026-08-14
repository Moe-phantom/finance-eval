import warnings
import numpy as np, pandas as pd
from finance_eval import _clean_dates, SplitSpec, chronological_split, bootstrap_edge, audit

print("1. tz-aware without market_tz now RAISES")
try:
    _clean_dates(["2024-01-02 10:00:00+11:00"])
    print("   FAIL — no error")
except ValueError as e:
    print(f"   ok: {str(e)[:62]}...")

print("\n2. market_tz preserves the intended market day")
for tz, ts in [("Australia/Sydney","2024-01-02 10:00:00+11:00"),
               ("America/New_York","2024-01-02 20:00:00-05:00"),
               ("UTC","2024-01-02 20:00:00-05:00")]:
    print(f"   {tz:<18} -> {_clean_dates([ts], market_tz=tz).iloc[0].date()}")

print("\n3. naive + market_tz warns instead of silently ignoring")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _clean_dates(["2024-01-02"], market_tz="America/New_York")
    print(f"   {w[0].category.__name__}: {str(w[0].message)[:60]}...")

n_days, n_tick = 200, 20
days = pd.bdate_range("2022-01-01", periods=n_days)
jitter = pd.to_timedelta(np.tile(np.arange(n_tick)*60, n_days), unit="s")
d_intra = pd.Series(np.repeat(days, n_tick)) + jitter
d_clean = pd.Series(np.repeat(days, n_tick))

print("\n4. intraday timestamps now collapse to true observation dates")
for nm, d in [("date-only", d_clean), ("intraday ", d_intra)]:
    tr, te = chronological_split(d, SplitSpec(train_frac=0.7, horizon=1, embargo=10))
    print(f"   {nm}  unique={_clean_dates(d).nunique():>4}  train={len(tr)} test={len(te)}")

rng = np.random.default_rng(0)
mkt = rng.normal(0.0004, 0.012, n_days); idio = rng.normal(0, 0.005, (n_days, n_tick))
y = (np.repeat(mkt, n_tick).reshape(n_days, n_tick) + idio).ravel()
pred = rng.choice([-1.,1.], size=y.size); base = np.ones_like(y)

print("\n5. bootstrap no longer understates uncertainty")
for nm, d in [("date-only", d_clean), ("intraday ", d_intra)]:
    r = bootstrap_edge(d, y, pred, base, pred_kind="direction", horizon=1, n_boot=600, seed=1)
    print(f"   {nm}  n_dates={r.n_dates:>4}  CI width={r.ci_high-r.ci_low:.4f}")

print("\n6. audit's new granularity check, on a deliberately broken panel")
g = np.tile([f"T{i}" for i in range(n_tick)], n_days)
tr, te = chronological_split(d_clean, SplitSpec(train_frac=0.7, horizon=1, embargo=10))
one_per_row = pd.Series(pd.bdate_range("2022-01-01", periods=len(te)))
rep = audit(d_clean.iloc[tr], one_per_row, y[te], SplitSpec(horizon=1, embargo=10),
            target_groups=g[te])
for c in rep.checks:
    if "cross-section" in c["check"]:
        print(f"   {'PASS' if c['pass'] else 'FAIL'}  {c['check']}\n         {c['detail']}")
