# Finance vertical | evaluation harness (v2)

**What this does.** Provides automated checks and explicit assertions for common
sources of leakage, and dependence-aware inference for directional accuracy claims.

**What this does not do.** See `KNOWN_LIMITATIONS` in `finance_eval.py`. Read it
before quoting any number produced here. Passing these checks does not mean a
pipeline is clean, and this harness cannot establish that a model has economic value.

## Files

| File | Purpose |
|---|---|
| `finance_eval.py` | The harness |
| `test_changes.py` | 14 tests verifying each v2 change behaves as claimed |
| `demo_leak_catalog.py` | Which bug produces which inflated number |
| `demo_dependence.py` | Why the binomial test was wrong; what block length does |
| `demo_leakage.py` | End-to-end clean vs conventional comparison |

## Quickstart

```python
from finance_eval import (SplitSpec, chronological_split, fit_scaler_on_train,
                          benchmark_table, benchmark_verdict, audit)

spec = SplitSpec(train_frac=0.7, horizon=1, embargo=10)
tr, te = chronological_split(df["date"], spec)
X_tr, X_te, scaler = fit_scaler_on_train(X, tr, te)

model.fit(X_tr, y[tr])
pred = model.predict(X_te)          # must be {-1, 0, +1} for pred_kind='direction'

tbl = benchmark_table(df["date"].iloc[te], y[te], pred,
                      pred_kind="direction", horizon=spec.horizon)
print(tbl)
print(benchmark_verdict(tbl))       # verdict is vs the STRONGEST baseline only

print(audit(df["date"].iloc[tr], df["date"].iloc[te], y[te], spec,
            target_groups=df["ticker"].iloc[te],
            scaler=scaler, X=X, train_idx=tr,          # VERIFIED
            universe_is_point_in_time=True))           # ASSERTED
```

## Reporting standard

Report the benchmark table plus the audit. A single accuracy figure is not a result.

**The verdict is against the strongest baseline only.** A pure-noise model will
"significantly" beat always-down in an up-drifting sample. Measured here: a random
model scored 0.5102 and beat both always_down and coin_flip with intervals excluding
zero, while failing against always_up (0.5203). Only the hardest baseline carries
information.

## What changed in v2

**Tier A — correctness.** `to_returns` computes within ticker groups and sorts by
date (previously differenced across ticker boundaries, and trusted input order).
The price-level check runs within group (previously correlated across a concatenated
panel, making it meaningless there). An empty audit no longer reports success.
`directional_accuracy` requires an explicit `pred_kind`; passing probabilities as
directions now raises instead of silently scoring everything in [0,1] as "up" —
measured effect on a test case: 0.67 accuracy from nothing. The purge gap is counted
in observation dates, not calendar days; on a business-day series with horizon=5 the
old check saw an 8-day calendar gap where there were 0 observation dates.

**Tier B — inference.** The binomial test is replaced by a stationary block bootstrap
(Politis–Romano) resampling contiguous blocks of dates, paired against a baseline
suite. On a 400-day × 20-ticker panel with one market factor, the iid-row assumption
behind `binomtest` produced a CI width of 0.031 where date-level resampling gave
0.084 — the old p-values were anticonservative by roughly a factor of three in
interval width.

Block resampling **addresses cross-sectional dependence, provides confidence
intervals, and substantially improves inference under overlapping labels.** It does
not fully reproduce the dependence structure when labels overlap. Mean block length
defaults to `2 × horizon`; it must exceed the label horizon. Measured: at horizon=1
block length barely matters (0.143 vs 0.125), at horizon=20 it matters a lot
(L=1: 0.135, L=40: 0.318).

**Tier C — scope.** Audit output is split into VERIFIED (computed by the harness)
and ASSERTED (the caller's word). The scaler check moved from asserted to verified —
it refits a clone on the training rows and compares learned parameters, catching a
full-array fit. Point-in-time universe remains ASSERTED; the harness cannot inspect
constituent history.

## Not implemented, deliberately

Transaction costs, Sharpe, drawdown, PnL. This is a falsification tool. Adding a
backtest engine invites stronger claims, which is the opposite of its purpose.

Walk-forward, validation split, multiple-testing ledger, near-duplicate detection
across the train/test boundary — all needed, none present. See `KNOWN_LIMITATIONS`.
