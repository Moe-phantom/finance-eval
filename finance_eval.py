"""
finance_eval.py — evaluation harness for the Mantis finance vertical (v2).

WHAT THIS DOES
    Provides automated checks and explicit assertions for common sources of
    leakage, and dependence-aware inference for directional accuracy claims.

WHAT THIS DOES NOT DO
    See KNOWN_LIMITATIONS at the bottom of this file. Read it before quoting
    any number produced here. This harness cannot establish that a model has
    economic value, and passing its checks does not mean a pipeline is clean.

v2 changes
    Tier A  correctness: panel-safe returns, sorted input, panel-safe level
            heuristic, non-empty audit, explicit prediction kinds, purge gap
            measured in observation dates rather than calendar days
    Tier B  inference: stationary block bootstrap paired against a benchmark
            suite, replacing the binomial test
    Tier C  scope: audit results separated into VERIFIED and ASSERTED

v3 changes
    Tier A  date handling: observation dates are normalized to midnight, so
            intraday timestamps can no longer inflate the apparent number of
            independent units. tz-aware input now requires an explicit
            market_tz instead of being converted to UTC, which could move an
            observation across midnight and relabel the market day. Measured
            on a 200-day x 20-ticker panel stamped at per-ticker trade times:
            the harness saw 4,000 "dates" instead of 200 and reported a CI
            width of 0.0535 where the correct figure was 0.1241 - a 2.3x
            understatement, the same magnitude as the binomtest bug that
            Tier B was written to remove.
    Tier C  audit gains a cross-sectional granularity check that catches the
            above when target_groups is supplied.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SplitSpec", "chronological_split",
    "to_returns", "level_warning",
    "DirectionalResult", "directional_accuracy",
    "BootstrapResult", "bootstrap_edge", "benchmark_table",
    "fit_scaler_on_train", "verify_scaler_fitted_on_train",
    "AuditReport", "audit", "benchmark_verdict",
    "KNOWN_LIMITATIONS",
]


# =============================================================================
# Splitting
# =============================================================================

@dataclass(frozen=True)
class SplitSpec:
    """Configuration for a purged chronological split.

    horizon : periods (observation dates) ahead the target looks. The last
              `horizon` training dates have targets reaching into the test
              window and are PURGED.
    embargo : additional observation dates dropped after the purge.
    """
    train_frac: float = 0.7
    horizon: int = 1
    embargo: int = 0
    min_train_dates: int = 30
    min_test_dates: int = 30

    def __post_init__(self) -> None:
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must be in (0, 1)")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.embargo < 0:
            raise ValueError("embargo must be >= 0")


def _clean_dates(dates: Sequence, *, market_tz: str | None = None) -> pd.Series:
    """Coerce to tz-naive OBSERVATION DATES, normalized to midnight. (Tier A)

    The harness treats one unique value as one market day: chronological_split
    splits on it, bootstrap_edge resamples on it, and audit counts the purge
    gap in it. Intraday timestamps therefore inflate the apparent number of
    independent units, so input is normalized to date granularity.

    Timezone-naive input is assumed to already represent market-local
    observation dates.

    market_tz : required when input is tz-aware. Converting to UTC can move an
                observation across midnight (Sydney 10:00 +11:00 becomes the
                previous day) and silently relabel the market day. If UTC
                calendar dates are intended, pass market_tz="UTC".
    """
    try:
        s = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    except ValueError as exc:
        # e.g. mixed UTC offsets across a DST boundary. pandas suggests
        # utc=True, which is exactly the conversion that relabels market days.
        raise ValueError(
            f"dates could not be parsed as a single datetime type ({exc}). "
            "Normalize the offsets upstream, or pass values already localized "
            "to one market timezone."
        ) from exc

    if s.isna().any():
        raise ValueError(f"{int(s.isna().sum())} date(s) could not be parsed (NaT)")

    if isinstance(s.dtype, pd.DatetimeTZDtype):
        if market_tz is None:
            raise ValueError(
                "tz-aware dates require an explicit market_tz (e.g. "
                "'America/New_York'). Converting to UTC can shift the "
                "observation across midnight and relabel the market day. "
                "If UTC calendar dates are intended, pass market_tz='UTC'."
            )
        try:
            s = s.dt.tz_convert(market_tz).dt.tz_localize(None)
        except Exception as exc:
            raise ValueError(f"invalid market_tz: {market_tz!r}") from exc
    elif market_tz is not None:
        # Not an error: naive input is market-local by definition, so there is
        # nothing to convert. But the caller supplied market_tz and may believe
        # a conversion happened, so this is surfaced rather than swallowed.
        warnings.warn(
            f"market_tz={market_tz!r} ignored: these dates are timezone-naive "
            "and are assumed to be market-local observation dates already.",
            UserWarning, stacklevel=2,
        )

    return s.dt.normalize().reset_index(drop=True)


def chronological_split(dates: Sequence, spec: SplitSpec = SplitSpec(), *,
                        market_tz: str | None = None):
    """Return (train_idx, test_idx) positional indices, split on DATES.

    Splitting on dates (not rows) keeps a panel's same-day cross-section
    entirely on one side of the boundary. Dates are normalized to midnight
    first; see _clean_dates for why, and for market_tz.
    """
    d = _clean_dates(dates, market_tz=market_tz)
    unique_dates = np.sort(d.unique())
    n_dates = len(unique_dates)

    cut = int(n_dates * spec.train_frac)
    train_end = cut - spec.horizon          # purge
    test_start = cut + spec.embargo         # embargo

    n_train_dates = max(train_end, 0)
    n_test_dates = max(n_dates - test_start, 0)
    if n_train_dates < spec.min_train_dates:
        raise ValueError(
            f"only {n_train_dates} train dates after purge; "
            f"min_train_dates={spec.min_train_dates}"
        )
    if n_test_dates < spec.min_test_dates:
        raise ValueError(
            f"only {n_test_dates} test dates after embargo; "
            f"min_test_dates={spec.min_test_dates}"
        )

    train_dates = set(unique_dates[:train_end])
    test_dates = set(unique_dates[test_start:])

    train_idx = np.flatnonzero(d.isin(train_dates).to_numpy())
    test_idx = np.flatnonzero(d.isin(test_dates).to_numpy())

    # invariants (Tier A)
    if np.intersect1d(train_idx, test_idx).size:
        raise AssertionError("train and test indices overlap")
    if train_idx.size == 0 or test_idx.size == 0:
        raise AssertionError("empty split")
    return train_idx, test_idx


# =============================================================================
# Targets
# =============================================================================

def to_returns(prices, dates=None, groups=None, log: bool = False, *,
               market_tz: str | None = None) -> pd.Series:
    """Price levels -> returns. Panel-safe and order-safe. (Tier A)

    groups : ticker labels. Returns are computed WITHIN each group, so the
             last price of one ticker is never differenced against the first
             price of the next.
    dates  : if given, rows are sorted by (group, date) before differencing.
    """
    px = pd.Series(np.asarray(prices, dtype=float)).reset_index(drop=True)
    if log and (px <= 0).any():
        raise ValueError("log returns require strictly positive prices")

    frame = pd.DataFrame({"px": px})
    frame["_g"] = np.asarray(groups) if groups is not None else 0
    if dates is not None:
        frame["_d"] = _clean_dates(dates, market_tz=market_tz)
        frame = frame.sort_values(["_g", "_d"], kind="mergesort")

    if log:
        out = np.log(frame["px"]).groupby(frame["_g"], sort=False).diff()
    else:
        out = frame.groupby("_g", sort=False)["px"].pct_change()
    return out.reindex(px.index)                  # restore original row order


def level_warning(y, groups=None, threshold: float = 0.95) -> dict:
    """HEURISTIC (not a test): does the target look like a price level?

    Price levels are near-unit-root; returns are not. Computed WITHIN group
    so a concatenated panel is not correlated across ticker boundaries.

    A target can fail to trip this and still be leaky. This flags one
    specific, common mistake — nothing more.
    """
    s = pd.Series(np.asarray(y, dtype=float))
    g = pd.Series(np.asarray(groups) if groups is not None else 0, index=s.index)

    acs = []
    for _, grp in s.groupby(g, sort=False):
        grp = grp.dropna()
        if len(grp) >= 30:
            ac = grp.autocorr(lag=1)
            if np.isfinite(ac):
                acs.append(ac)
    if not acs:
        return {"checked": False, "flagged": False, "median_ac1": float("nan")}

    med = float(np.median(acs))
    return {
        "checked": True,
        "flagged": bool(med > threshold),
        "median_ac1": med,
        "n_groups": len(acs),
        "note": ("median within-group lag-1 autocorrelation exceeds "
                 f"{threshold}; target may be a price level, on which R^2 is "
                 "not evidence of skill") if med > threshold else "",
    }


# =============================================================================
# Directional accuracy
# =============================================================================

@dataclass
class DirectionalResult:
    n: int
    accuracy: float
    up_share: float
    n_zero_true_dropped: int

    def __str__(self) -> str:
        return (f"n={self.n}  accuracy={self.accuracy:.4f}  "
                f"up_share={self.up_share:.4f}")


def _as_direction(pred, kind: str) -> np.ndarray:
    """Convert predictions to {-1, 0, +1}. Explicit kind required. (Tier A)"""
    p = np.asarray(pred, dtype=float)
    if kind == "direction":
        finite = p[np.isfinite(p)]
        if finite.size and not np.isin(finite, (-1.0, 0.0, 1.0)).all():
            raise ValueError(
                "pred_kind='direction' requires values in {-1, 0, +1}. "
                "If these are predicted returns use pred_kind='return'; "
                "if they are probabilities, threshold them first."
            )
        return p
    if kind == "return":
        return np.sign(p)
    raise ValueError("pred_kind must be 'direction' or 'return'")


def directional_accuracy(y_true, y_pred, pred_kind: str) -> DirectionalResult:
    """Raw directional accuracy. Zero true returns are DROPPED (documented).

    This number is meaningless alone. Pass it through bootstrap_edge() or
    benchmark_table() to compare it against baselines.
    """
    yt = np.sign(np.asarray(y_true, dtype=float))
    yp = _as_direction(y_pred, pred_kind)
    finite = np.isfinite(yt) & np.isfinite(yp)
    n_zero = int((finite & (yt == 0)).sum())
    m = finite & (yt != 0)
    yt, yp = yt[m], yp[m]
    if yt.size == 0:
        raise ValueError("no valid observations")
    return DirectionalResult(
        n=int(yt.size),
        accuracy=float((yt == yp).mean()),
        up_share=float((yt > 0).mean()),
        n_zero_true_dropped=n_zero,
    )


# =============================================================================
# Tier B — stationary block bootstrap
# =============================================================================

@dataclass
class BootstrapResult:
    baseline: str
    model_accuracy: float
    baseline_accuracy: float
    edge: float
    ci_low: float
    ci_high: float
    share_edge_le_zero: float
    n_obs: int
    n_dates: int
    mean_block_len: float
    n_boot: int

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0.0

    def __str__(self) -> str:
        v = "CI excludes 0" if self.ci_excludes_zero else "CI includes 0"
        return (f"vs {self.baseline:<14} acc={self.model_accuracy:.4f} "
                f"base={self.baseline_accuracy:.4f} edge={self.edge:+.4f} "
                f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]  ({v})")


def _date_block_indices(n_dates: int, mean_block: float, rng) -> np.ndarray:
    """Stationary bootstrap (Politis & Romano): geometric block lengths.

    Resamples contiguous blocks of DATES with wraparound. Blocks preserve
    dependence out to roughly the block length; mean block length should
    exceed the label horizon.
    """
    p = 1.0 / max(mean_block, 1.0)
    new_block = rng.random(n_dates) < p
    new_block[0] = True
    block_id = np.cumsum(new_block) - 1
    starts_at = np.flatnonzero(new_block)
    offset = np.arange(n_dates) - starts_at[block_id]
    starts = rng.integers(0, n_dates, size=starts_at.size)
    return (starts[block_id] + offset) % n_dates


def bootstrap_edge(
    dates, y_true, y_pred, baseline_pred, *,
    pred_kind: str,
    baseline_kind: str = "direction",
    baseline_name: str = "baseline",
    horizon: int = 1,
    mean_block_len: float | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    market_tz: str | None = None,
) -> BootstrapResult:
    """Paired model-vs-baseline edge with a dependence-aware interval.

    Resampling whole DATES in contiguous blocks addresses cross-sectional
    dependence exactly (a market-wide day moves in or out as a unit) and
    substantially improves inference under overlapping labels. It does not
    fully reproduce the dependence structure when labels overlap, and it
    assumes the dependence structure is stable across the sample —
    nonstationarity is not addressed here.

    mean_block_len defaults to 2 * horizon (heuristic: block length must
    exceed the label horizon to retain the overlap structure).

    Dates are normalized to midnight before factorizing. Passing intraday
    timestamps would otherwise make each row its own "date" and collapse this
    back toward the iid resampling it replaces.
    """
    d = _clean_dates(dates, market_tz=market_tz)
    yt = np.sign(np.asarray(y_true, dtype=float))
    yp = _as_direction(y_pred, pred_kind)
    yb = _as_direction(baseline_pred, baseline_kind)

    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(yb) & (yt != 0)
    yt, yp, yb, d = yt[m], yp[m], yb[m], d[m].reset_index(drop=True)

    hit_m = (yt == yp).astype(np.float64)
    hit_b = (yt == yb).astype(np.float64)

    codes, uniques = pd.factorize(d, sort=True)
    n_dates = len(uniques)
    if n_dates < 20:
        raise ValueError(f"need >= 20 distinct test dates, got {n_dates}")

    # CSR-style ragged index: observations grouped by date
    order = np.argsort(codes, kind="mergesort")
    counts = np.bincount(codes, minlength=n_dates)
    ptr = np.concatenate([[0], np.cumsum(counts)])[:-1]

    L = float(mean_block_len) if mean_block_len else float(max(2 * horizon, 2))
    rng = np.random.default_rng(seed)
    edges = np.empty(n_boot)

    for b in range(n_boot):
        di = _date_block_indices(n_dates, L, rng)
        lens = counts[di]
        total = int(lens.sum())
        if total == 0:
            edges[b] = 0.0
            continue
        base_ptr = np.repeat(ptr[di], lens)
        within = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
        obs = order[base_ptr + within]
        edges[b] = hit_m[obs].mean() - hit_b[obs].mean()

    lo, hi = np.percentile(edges, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        baseline=baseline_name,
        model_accuracy=float(hit_m.mean()),
        baseline_accuracy=float(hit_b.mean()),
        edge=float(hit_m.mean() - hit_b.mean()),
        ci_low=float(lo), ci_high=float(hi),
        share_edge_le_zero=float((edges <= 0).mean()),
        n_obs=int(yt.size), n_dates=n_dates,
        mean_block_len=L, n_boot=n_boot,
    )


def benchmark_table(
    dates, y_true, y_pred, *, pred_kind: str, horizon: int = 1,
    extra_baselines: Mapping[str, Sequence] | None = None,
    market_tz: str | None = None,
    **kw,
) -> pd.DataFrame:
    """Run the model against a suite of baselines through one pipeline.

    Built-in: always_up, always_down, coin_flip. Pass previous-day sign or a
    momentum rule via extra_baselines (they need panel structure the harness
    does not have).
    """
    yt = np.asarray(y_true, dtype=float)
    rng = np.random.default_rng(kw.get("seed", 0))
    baselines: dict[str, np.ndarray] = {
        "always_up": np.ones_like(yt),
        "always_down": -np.ones_like(yt),
        "coin_flip": rng.choice([-1.0, 1.0], size=yt.size),
    }
    if extra_baselines:
        for k, v in extra_baselines.items():
            baselines[k] = np.sign(np.asarray(v, dtype=float))

    rows = []
    for name, bp in baselines.items():
        r = bootstrap_edge(dates, y_true, y_pred, bp, pred_kind=pred_kind,
                           baseline_name=name, horizon=horizon,
                           market_tz=market_tz, **kw)
        rows.append({
            "baseline": name,
            "model_acc": round(r.model_accuracy, 4),
            "baseline_acc": round(r.baseline_accuracy, 4),
            "edge": round(r.edge, 4),
            "ci_low": round(r.ci_low, 4),
            "ci_high": round(r.ci_high, 4),
            "excludes_0": r.ci_excludes_zero,
        })
    tbl = pd.DataFrame(rows).sort_values("baseline_acc", ascending=False,
                                         ignore_index=True)
    tbl["strongest"] = False
    tbl.loc[0, "strongest"] = True          # highest baseline accuracy
    return tbl


def benchmark_verdict(tbl: pd.DataFrame) -> dict:
    """The verdict is against the STRONGEST baseline only.

    Beating a weak baseline is not evidence. A model that predicts pure noise
    will "significantly" beat always-down in an up-drifting sample, and will
    beat a coin flip whenever the sample is unbalanced. Only the hardest
    baseline in the table carries information.
    """
    if tbl.empty:
        raise ValueError("empty benchmark table")
    row = tbl.loc[tbl["strongest"]].iloc[0]
    beaten = [r.baseline for r in tbl.itertuples() if r.excludes_0]
    return {
        "strongest_baseline": row["baseline"],
        "strongest_baseline_acc": float(row["baseline_acc"]),
        "model_acc": float(row["model_acc"]),
        "edge_vs_strongest": float(row["edge"]),
        "ci": (float(row["ci_low"]), float(row["ci_high"])),
        "beats_strongest": bool(row["excludes_0"]),
        "baselines_beaten": beaten,
        "note": ("" if row["excludes_0"] else
                 "Model does not beat the strongest baseline. Any other "
                 "baseline it beats is not evidence of skill."),
    }


# =============================================================================
# Scaling
# =============================================================================

def fit_scaler_on_train(X, train_idx, test_idx, scaler=None):
    """Fit on train, transform both. Validates indices and finiteness."""
    X = np.asarray(X, dtype=float)
    tr, te = np.asarray(train_idx), np.asarray(test_idx)
    if np.intersect1d(tr, te).size:
        raise ValueError("train_idx and test_idx overlap")
    if tr.size == 0 or te.size == 0:
        raise ValueError("empty train or test index")
    if max(tr.max(), te.max()) >= len(X) or min(tr.min(), te.min()) < 0:
        raise ValueError("index out of range for X")
    if not np.isfinite(X[tr]).all() or not np.isfinite(X[te]).all():
        raise ValueError("X contains NaN or inf")

    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    scaler.fit(X[tr])
    return scaler.transform(X[tr]), scaler.transform(X[te]), scaler


def verify_scaler_fitted_on_train(scaler, X, train_idx) -> tuple[bool, str]:
    """VERIFY (not assert) that a fitted scaler saw only training rows.

    Refits a clone on X[train_idx] and compares learned parameters. A scaler
    fitted on the full array will not match.
    """
    try:
        from sklearn.base import clone
    except ImportError:
        return False, "sklearn unavailable"

    X = np.asarray(X, dtype=float)
    ref = clone(scaler).fit(X[np.asarray(train_idx)])
    params = [a for a in ("mean_", "scale_", "data_min_", "data_max_", "center_")
              if hasattr(scaler, a) and hasattr(ref, a)]
    if not params:
        return False, "no comparable fitted attributes found"
    for a in params:
        if not np.allclose(getattr(scaler, a), getattr(ref, a), equal_nan=True):
            return False, f"{a} differs from a train-only fit"
    return True, f"matches train-only fit on {', '.join(params)}"


# =============================================================================
# Tier C — audit with VERIFIED / ASSERTED separation
# =============================================================================

@dataclass
class AuditReport:
    checks: list = field(default_factory=list)

    def add(self, name: str, passed: bool, kind: str, detail: str = "") -> None:
        if kind not in ("VERIFIED", "ASSERTED"):
            raise ValueError("kind must be VERIFIED or ASSERTED")
        self.checks.append({"check": name, "kind": kind,
                            "pass": bool(passed), "detail": detail})

    @property
    def all_passed(self) -> bool:
        return bool(self.checks) and all(c["pass"] for c in self.checks)  # Tier A

    @property
    def verified_passed(self) -> bool:
        v = [c for c in self.checks if c["kind"] == "VERIFIED"]
        return bool(v) and all(c["pass"] for c in v)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)

    def __str__(self) -> str:
        if not self.checks:
            return "AUDIT EMPTY — no checks were run"
        out = []
        for kind in ("VERIFIED", "ASSERTED"):
            rows = [c for c in self.checks if c["kind"] == kind]
            if not rows:
                continue
            out.append(f"[{kind}]" + ("" if kind == "VERIFIED"
                                      else "  (caller's word, not checked)"))
            for c in rows:
                out.append(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}")
                if c["detail"]:
                    out.append(f"        {c['detail']}")
        n_ast = sum(c["kind"] == "ASSERTED" for c in self.checks)
        out.append("")
        out.append("VERIFIED CHECKS PASSED" if self.verified_passed
                   else "VERIFIED CHECKS FAILED")
        if n_ast:
            out.append(f"{n_ast} item(s) were ASSERTED, not verified.")
        return "\n".join(out)


def audit(
    dates_train, dates_test, y_target, spec: SplitSpec, *,
    target_groups=None,
    scaler=None, X=None, train_idx=None,
    universe_is_point_in_time: bool | None = None,
    market_tz: str | None = None,
) -> AuditReport:
    """Run checks. VERIFIED items are computed; ASSERTED items are not."""
    a = AuditReport()
    dtr = _clean_dates(dates_train, market_tz=market_tz)
    dte = _clean_dates(dates_test, market_tz=market_tz)
    utr, ute = np.sort(dtr.unique()), np.sort(dte.unique())

    a.add("Test window strictly after train window", utr.max() < ute.min(),
          "VERIFIED",
          f"train ends {pd.Timestamp(utr.max()).date()}, "
          f"test starts {pd.Timestamp(ute.min()).date()}")

    a.add("No dates shared between train and test",
          len(set(utr) & set(ute)) == 0, "VERIFIED")

    # Tier A: gap measured in OBSERVATION DATES, not calendar days
    all_dates = np.sort(np.unique(np.concatenate([utr, ute])))
    gap_obs = int(np.searchsorted(all_dates, ute.min())
                  - np.searchsorted(all_dates, utr.max()) - 1)
    a.add("Purge gap covers the label horizon", gap_obs >= spec.horizon - 1,
          "VERIFIED",
          f"gap={gap_obs} observation dates, horizon={spec.horizon}, "
          f"embargo={spec.embargo} (gap counted in dates present in the data, "
          "not calendar days)")

    # Tier A (v3): catch intraday timestamps masquerading as observation dates.
    # In a panel with more than one group, a day carries several rows. One row
    # per date means the cross-section was never grouped, which defeats both
    # date-level splitting and date-level resampling.
    if target_groups is not None:
        n_groups = int(pd.Series(np.asarray(target_groups)).nunique())
        collapsed = n_groups > 1 and len(ute) == len(dte)
        a.add("Test dates are shared across the cross-section", not collapsed,
              "VERIFIED",
              f"{len(dte)} rows, {len(ute)} distinct dates, {n_groups} groups"
              + ("" if not collapsed else
                 " — one date per row despite multiple groups suggests intraday "
                 "timestamps; date-level splitting and resampling are defeated"))

    lw = level_warning(y_target, groups=target_groups)
    if lw["checked"]:
        detail = lw.get("note") or f"median within-group lag-1 ac = {lw['median_ac1']:.3f}"
    else:
        detail = "too few observations per group to check"
    a.add("Target does not look like a price level [HEURISTIC]",
          not lw["flagged"], "VERIFIED", detail)

    if scaler is not None and X is not None and train_idx is not None:
        ok, why = verify_scaler_fitted_on_train(scaler, X, train_idx)
        a.add("Scaler fitted on train only", ok, "VERIFIED", why)

    if universe_is_point_in_time is not None:
        a.add("Universe is point-in-time", universe_is_point_in_time, "ASSERTED",
              "the harness cannot inspect index constituent history")
    return a


# =============================================================================
KNOWN_LIMITATIONS = """
This harness does NOT check, and passing it does NOT establish:

  Feature construction
    - features built from information unavailable at prediction time
    - rolling windows that span the train/test boundary
    - restated fundamentals, revised macro data, as-of availability
    - decision vs. execution timestamps (a 16:05 release cannot be traded
      at 16:00, but a daily dataset will not show that)

  Data integrity
    - split/dividend adjustment of prices
    - corporate actions, ticker reuse, delisted and merged securities
    - point-in-time index constituents (ASSERTED only)
    - the market timezone: for tz-naive input the harness assumes the dates
      are already market-local observation dates and cannot verify it
    - near-duplicate documents split across the train/test boundary

  Evaluation design
    - no validation split; using the test set to choose models, features or
      hyperparameters invalidates it
    - no walk-forward or rolling-origin evaluation; one split can be one regime
    - no regime breakdown (bull/bear, high/low volatility)
    - no multiple-testing ledger; testing many models or subgroups inflates
      false positives regardless of what any single result shows

  Inference
    - observation granularity is assumed DAILY. Dates are normalized to
      midnight, so an intraday design (hourly or bar-level) is silently
      collapsed to days; such designs need block resampling at the bar
      level, which is not implemented
    - the block bootstrap assumes a stable dependence structure;
      nonstationarity is not addressed
    - with overlapping labels it improves on iid resampling but does not
      fully reproduce the dependence structure
    - the level check is a HEURISTIC for one common mistake, not a test

  Economic value
    - no transaction costs, spread, slippage, market impact, turnover,
      borrowing or financing costs
    - no PnL, Sharpe, Sortino, drawdown, or benchmark-relative return
    - a statistically detectable directional edge can still lose money

This tool exists to stop an accuracy figure being reported without a
baseline. It cannot show that a model has alpha.
"""