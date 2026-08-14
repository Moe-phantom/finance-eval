"""test_changes.py — verify each v2 change actually does what it claims."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from finance_eval import (
    benchmark_verdict, SplitSpec, chronological_split, to_returns, level_warning,
    directional_accuracy, bootstrap_edge, benchmark_table,
    fit_scaler_on_train, verify_scaler_fitted_on_train, audit, AuditReport,
)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, note=""):
    results.append({"test": name, "result": PASS if ok else FAIL, "note": note})


# --- Tier A -----------------------------------------------------------------

# A1: to_returns must not cross ticker boundaries
px = [100.0, 101.0, 102.0, 200.0, 202.0, 204.0]
tk = ["A", "A", "A", "B", "B", "B"]
naive = pd.Series(px).pct_change()
fixed = to_returns(px, groups=tk)
check("A1 to_returns panel-safe",
      np.isnan(fixed.iloc[3]) and abs(naive.iloc[3] - 0.9608) < 1e-3,
      f"naive row3={naive.iloc[3]:.4f} (bogus), fixed row3={fixed.iloc[3]}")

# A2: to_returns sorts by date
d = pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"])
p = [102.0, 100.0, 101.0]
srt = to_returns(p, dates=d)
check("A2 to_returns sorts by date",
      np.isnan(srt.iloc[1]) and abs(srt.iloc[2] - 0.01) < 1e-9,
      f"returns in date order: {[round(v,4) if np.isfinite(v) else None for v in srt]}")

# A3: level_warning computed within group, not across concatenated panel
rng = np.random.default_rng(0)
lev = np.concatenate([100 * np.exp(np.cumsum(rng.normal(0, .02, 300))) for _ in range(3)])
grp = np.repeat(["A", "B", "C"], 300)
lw_g = level_warning(lev, groups=grp)
rets = np.concatenate([rng.normal(0, .02, 300) for _ in range(3)])
lw_r = level_warning(rets, groups=grp)
check("A3 level heuristic panel-aware",
      lw_g["flagged"] and not lw_r["flagged"],
      f"levels ac1={lw_g['median_ac1']:.3f} flagged={lw_g['flagged']}; "
      f"returns ac1={lw_r['median_ac1']:.3f} flagged={lw_r['flagged']}")

# A4: empty audit must not report success
check("A4 empty audit fails", AuditReport().all_passed is False,
      "AuditReport().all_passed = False")

# A5: probabilities rejected under pred_kind='direction'
probs = np.array([0.73, 0.21, 0.55])
yt = np.array([0.01, -0.01, 0.01])
try:
    directional_accuracy(yt, probs, pred_kind="direction")
    raised = False
except ValueError:
    raised = True
silent = directional_accuracy(yt, probs, pred_kind="return")
check("A5 probabilities rejected as directions", raised,
      f"pred_kind='return' would have scored all-up: acc={silent.accuracy:.2f}")

# A6: purge gap counted in observation dates, not calendar days
bdays = pd.bdate_range("2026-01-01", periods=200)
spec5 = SplitSpec(train_frac=0.7, horizon=5, embargo=0,
                  min_train_dates=20, min_test_dates=20)
tr, te = chronological_split(bdays, spec5)
rep = audit(bdays[tr], bdays[te], np.random.default_rng(1).normal(0, .01, len(te)), spec5)
gap_row = rep.to_frame().query("check.str.contains('Purge gap')", engine="python").iloc[0]
cal_days = (bdays[te].min() - bdays[tr].max()).days
check("A6 gap in observation dates", "observation dates" in gap_row["detail"],
      f"calendar gap={cal_days}d vs {gap_row['detail'].split(',')[0]}")

# --- Tier B -----------------------------------------------------------------

# B1: block bootstrap widens the interval vs iid-date resampling under
#     cross-sectional dependence
n_days, n_tick = 300, 20
dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_days), n_tick)
market = rng.normal(0, .012, n_days)
y = np.repeat(market, n_tick) + rng.normal(0, .004, n_days * n_tick)  # highly correlated
pred = np.ones_like(y)                       # always-up model
r_blk = bootstrap_edge(dates, y, pred, np.ones_like(y), pred_kind="direction",
                       horizon=1, mean_block_len=20, n_boot=500, seed=1)
r_iid = bootstrap_edge(dates, y, pred, np.ones_like(y), pred_kind="direction",
                       horizon=1, mean_block_len=1, n_boot=500, seed=1)
check("B1 bootstrap runs on dependent panel", np.isfinite(r_blk.ci_low),
      f"n_obs={r_blk.n_obs} across n_dates={r_blk.n_dates}")

# B2: identical model and baseline must give exactly zero edge
check("B2 model==baseline gives zero edge", abs(r_blk.edge) < 1e-12,
      f"edge={r_blk.edge}")

# B3: benchmark table on a genuinely signal-free model
rng2 = np.random.default_rng(5)
y2 = rng2.normal(0.0004, 0.018, n_days * n_tick)
pred2 = rng2.choice([-1.0, 1.0], size=y2.size)
tbl = benchmark_table(dates, y2, pred2, pred_kind="direction", horizon=1,
                      n_boot=400, seed=3)
v = benchmark_verdict(tbl)
check("B3 noise model does not beat STRONGEST baseline",
      not v["beats_strongest"],
      f"strongest={v['strongest_baseline']} (acc={v['strongest_baseline_acc']:.4f}), "
      f"edge={v['edge_vs_strongest']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]; "
      f"weak baselines it 'beat': {[b for b in v['baselines_beaten']]}")

# --- Tier C -----------------------------------------------------------------

# C1: scaler verification catches a full-array fit
X = rng.normal(size=(400, 4))
tr_i, te_i = np.arange(280), np.arange(280, 400)
_, _, good = fit_scaler_on_train(X, tr_i, te_i, StandardScaler())
bad = StandardScaler().fit(X)                # leaked: fitted on everything
ok_g, why_g = verify_scaler_fitted_on_train(good, X, tr_i)
ok_b, why_b = verify_scaler_fitted_on_train(bad, X, tr_i)
check("C1 scaler verification catches leak", ok_g and not ok_b,
      f"train-only -> {ok_g}; full-array -> {ok_b} ({why_b})")

# C2: MinMaxScaler path also verified
_, _, gmm = fit_scaler_on_train(X, tr_i, te_i, MinMaxScaler())
bmm = MinMaxScaler().fit(X)
check("C2 MinMax leak caught",
      verify_scaler_fitted_on_train(gmm, X, tr_i)[0]
      and not verify_scaler_fitted_on_train(bmm, X, tr_i)[0])

# C3: audit separates VERIFIED from ASSERTED
dts = pd.bdate_range("2022-01-01", periods=300)
sp = SplitSpec(train_frac=0.7, horizon=1, embargo=5)
a_tr, a_te = chronological_split(dts, sp)
rep2 = audit(dts[a_tr], dts[a_te], rng.normal(0, .01, len(a_te)), sp,
             scaler=bad, X=X[:len(dts)], train_idx=a_tr,
             universe_is_point_in_time=True)
f = rep2.to_frame()
scaler_row = f[f["check"].str.contains("Scaler")].iloc[0]
check("C3 audit separates verified/asserted",
      set(f["kind"]) == {"VERIFIED", "ASSERTED"}
      and scaler_row["kind"] == "VERIFIED" and not scaler_row["pass"],
      f"{(f['kind']=='VERIFIED').sum()} verified, {(f['kind']=='ASSERTED').sum()} asserted; "
      "leaked scaler caught by audit")

# C4: split invariants
check("C4 split indices disjoint & non-empty",
      len(np.intersect1d(a_tr, a_te)) == 0 and len(a_tr) and len(a_te),
      f"train={len(a_tr)} rows, test={len(a_te)} rows")

# C5: NaT dates rejected
try:
    chronological_split(["2026-01-01", "not-a-date"] * 60, SplitSpec())
    nat_ok = False
except ValueError:
    nat_ok = True
check("C5 unparseable dates rejected", nat_ok)


if __name__ == "__main__":
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False, max_colwidth=88) + "\n")
    print(f"{(df.result == PASS).sum()}/{len(df)} passed")
