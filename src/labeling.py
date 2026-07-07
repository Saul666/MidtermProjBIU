"""Turn selected pairs into a supervised learning table (enriched).

One row = one extreme spread event (|z| >= entry). Features use only information
up to the event day; the label looks forward. Beyond the spread state and the
pair relationship, the table now carries three richer blocks:

  * regime / detachment  - long-vs-short correlation, co-membership, local
    cointegration health, relative momentum, sector dispersion, and each leg's
    detachment from its sector.
  * spread stationarity  - trailing ADF statistic and Hurst exponent: is the
    spread still mean-reverting right now?
  * market context       - a volatility-regime signal (real VIX if supplied,
    else a realized-vol proxy), a volume/activity spike, and cross-sectional
    stress (how many pairs are simultaneously extreme, and this pair's rank).

Extra label columns (reverted_5, reverted_20, reverted_safe) support multi-horizon
and risk-adjusted analysis; the model target stays `reverted` (10-day).

Leakage rules are unchanged: selection on the formation window, features look
back, label looks forward, splits by time (with optional purging).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import CONFIG, sector_of, SECTORS
from .features import (log_returns, spread_series, rolling_zscore,
                       rolling_corr, half_life, hurst, adf_stat)


def _sector_dispersion(rets: pd.DataFrame) -> dict:
    out = {}
    cum21 = rets.rolling(21).sum()
    for sec in set(SECTORS.values()):
        members = [t for t in rets.columns if sector_of(t) == sec]
        out[sec] = cum21[members].std(axis=1) if len(members) >= 2 else pd.Series(0.0, index=rets.index)
    return out


def _vix_series(prices, rets, context):
    """Real VIX (reindexed) if provided, else a realized-volatility proxy."""
    market_ret = rets.mean(axis=1)
    if context and context.get("vix") is not None:
        vix = context["vix"].reindex(prices.index).ffill().bfill()
    else:
        vix = (market_ret.rolling(21).std() * np.sqrt(252) * 100).bfill()
    return vix, vix.diff(20)


def build_event_dataset(prices: pd.DataFrame, pairs: pd.DataFrame,
                        formation_end: int, cfg=None, score_end: int | None = None,
                        context: dict | None = None) -> pd.DataFrame:
    """Build the enriched event-level modeling table across all selected pairs."""
    cfg = cfg or CONFIG.signal
    rets = log_returns(prices).reindex(prices.index)
    logp = np.log(prices)
    market_vol = rets.mean(axis=1).rolling(21).std()
    sector_disp = _sector_dispersion(rets)
    vix, vix_chg = _vix_series(prices, rets, context)
    volume = context.get("volume") if context else None
    bw, mw = cfg.baseline_window, cfg.momentum_window
    scoring_start_date = prices.index[formation_end]
    scoring_end_date = prices.index[score_end] if score_end is not None else None

    # ---- pass 1: per-pair spread / z, plus cross-sectional |z| matrix ----
    state = {}
    absz = {}
    for _, p in pairs.iterrows():
        a, b, beta = p["a"], p["b"], p["beta"]
        spread = spread_series(prices, a, b, beta)
        z = rolling_zscore(spread, cfg.z_window)
        state[(a, b)] = {
            "spread": spread, "z": z,
            "spread_vol": spread.rolling(cfg.z_window).std(),
            "corr_recent": rolling_corr(rets[a], rets[b], cfg.z_window),
            "corr_baseline": rolling_corr(rets[a], rets[b], bw),
            "comemb": (rolling_corr(rets[a], rets[b], 21) >= 0.5).astype(float).rolling(126).mean(),
            "vol_a": rets[a].rolling(cfg.z_window).std(),
            "vol_b": rets[b].rolling(cfg.z_window).std(),
            "mom_a": logp[a].diff(mw), "mom_b": logp[b].diff(mw),
        }
        absz[f"{a}/{b}"] = z.abs()
    absz_matrix = pd.DataFrame(absz)

    def _activity(leg):
        if volume is not None and leg in volume.columns:
            v = volume[leg].reindex(prices.index).ffill()
            return v.rolling(20).mean() / v.rolling(60).mean()
        ar = rets[leg].abs()
        return ar.rolling(20).mean() / ar.rolling(60).mean()

    def _detach(leg, other):
        sec = sector_of(leg)
        peers = [t for t in SECTORS if sector_of(t) == sec and t not in (leg, other)]
        if len(peers) < 2:
            return pd.Series(0.0, index=prices.index)
        sm = rets[peers].mean(axis=1)
        return rolling_corr(rets[leg], sm, bw) - rolling_corr(rets[leg], sm, cfg.z_window)

    rows = []
    H = cfg.horizon
    for _, p in pairs.iterrows():
        a, b, beta = p["a"], p["b"], p["beta"]
        s = state[(a, b)]
        spread, z = s["spread"], s["z"]
        z_arr = z.values
        dates = z.index
        n = len(z_arr)
        act_a, act_b = _activity(a), _activity(b)
        det_a, det_b = _detach(a, b), _detach(b, a)
        last_event = -10**9

        for i in range(max(cfg.z_window, bw) + 5, n - max(H, 20)):
            if dates[i] < scoring_start_date:
                continue
            if scoring_end_date is not None and dates[i] >= scoring_end_date:
                break
            zi = z_arr[i]
            if not np.isfinite(zi) or abs(zi) < cfg.entry_z:
                continue
            if i - last_event < cfg.min_gap:
                continue
            last_event = i

            # labels (forward)
            def reverted_within(h):
                w = np.abs(z_arr[i + 1:i + 1 + h])
                return int(np.nanmin(w) <= cfg.exit_z) if len(w) else 0
            # risk-adjusted: revert to exit before |z| blows past 3 within horizon
            fut = z_arr[i + 1:i + 1 + H]
            hit_exit = np.where(np.abs(fut) <= cfg.exit_z)[0]
            hit_stop = np.where(np.abs(fut) >= 3.0)[0]
            first_exit = hit_exit[0] if len(hit_exit) else 10**9
            first_stop = hit_stop[0] if len(hit_stop) else 10**9
            reverted_safe = int(first_exit < first_stop and first_exit < 10**9)

            # local cointegration health
            try:
                from statsmodels.tsa.stattools import coint
                _, coint_recent_p, _ = coint(logp[a].iloc[i - bw:i + 1], logp[b].iloc[i - bw:i + 1])
            except Exception:
                coint_recent_p = np.nan

            win = spread.iloc[max(0, i - 120):i + 1]
            cr, cb = s["corr_recent"].iloc[i], s["corr_baseline"].iloc[i]
            day_absz = absz_matrix.iloc[i]
            n_extreme = int((day_absz >= cfg.entry_z).sum())
            z_rank = float((day_absz < abs(zi)).mean())   # percentile among that day's pairs

            rows.append({
                "pair": f"{a}/{b}", "date": dates[i], "direction": int(np.sign(zi)),
                # spread / signal state
                "abs_z": abs(zi), "z_velocity": zi - z_arr[i - 3],
                "spread_vol": s["spread_vol"].iloc[i],
                "half_life": (lambda h: h if np.isfinite(h) else 999.0)(half_life(spread.iloc[max(0, i - 252):i + 1])),
                "spread_slope": np.polyfit(range(5), spread.iloc[i - 4:i + 1].values, 1)[0],
                # pair relationship
                "corr_recent": cr, "beta": beta, "coint_pvalue": p["coint_pvalue"],
                "market_vol": market_vol.iloc[i] if np.isfinite(market_vol.iloc[i]) else market_vol.median(),
                "ret_a_5d": rets[a].iloc[i - 4:i + 1].sum(), "ret_b_5d": rets[b].iloc[i - 4:i + 1].sum(),
                # regime / detachment
                "mom_a_252": s["mom_a"].iloc[i], "mom_b_252": s["mom_b"].iloc[i],
                "mom_rel": s["mom_a"].iloc[i] - s["mom_b"].iloc[i],
                "corr_baseline": cb, "corr_change": cr - cb, "comembership": s["comemb"].iloc[i],
                "coint_recent_pvalue": coint_recent_p,
                "vol_ratio": s["vol_a"].iloc[i] / s["vol_b"].iloc[i] if s["vol_b"].iloc[i] else np.nan,
                "sector_dispersion": sector_disp[sector_of(a)].iloc[i],
                "detach_max": float(np.nanmax([det_a.iloc[i], det_b.iloc[i]])),
                # spread stationarity
                "adf_stat": adf_stat(win), "hurst": hurst(win),
                # market context
                "vix_level": vix.iloc[i], "vix_change": vix_chg.iloc[i],
                "vol_spike_max": float(np.nanmax([act_a.iloc[i], act_b.iloc[i]])),
                "n_pairs_extreme": n_extreme, "z_rank": z_rank,
                # targets
                "reverted": reverted_within(H),
                "reverted_5": reverted_within(5), "reverted_20": reverted_within(20),
                "reverted_safe": reverted_safe,
            })

    data = pd.DataFrame(rows)
    if data.empty:
        return data
    for col in ["coint_recent_pvalue", "vol_ratio", "comembership", "vix_change", "vol_spike_max", "detach_max"]:
        data[col] = data[col].fillna(data[col].median())
    return data.dropna().sort_values("date").reset_index(drop=True)


def time_split(data: pd.DataFrame, test_fraction: float):
    """Split events by time: earliest (1 - f) train, most recent f test."""
    cut = int(len(data) * (1 - test_fraction))
    return data.iloc[:cut].copy(), data.iloc[cut:].copy()


def purged_time_split(data: pd.DataFrame, test_fraction: float, embargo_days: int = 10):
    """Time split with an embargo: drop train events whose forward label window
    overlaps the test period, removing boundary leakage from the horizon.
    """
    cut = int(len(data) * (1 - test_fraction))
    test = data.iloc[cut:].copy()
    if test.empty:
        return data.iloc[:cut].copy(), test
    test_start = test["date"].min()
    embargo = pd.Timedelta(days=int(embargo_days * 1.6))   # trading->calendar days
    train = data.iloc[:cut]
    train = train[train["date"] < (test_start - embargo)].copy()
    return train, test


def build_walkforward_dataset(prices: pd.DataFrame, formation_days: int = 504,
                              step: int = 126, cfg=None, pair_cfg=None,
                              context: dict | None = None) -> pd.DataFrame:
    """Walk-forward: re-select pairs every ``step`` days and score only the next
    block, so every event is out-of-sample relative to its pair's formation.
    """
    from .pairs import select_pairs
    cfg = cfg or CONFIG.signal
    n = len(prices)
    frames = []
    start = 0
    while start + formation_days + step <= n:
        form = prices.iloc[start:start + formation_days]
        pairs = select_pairs(form, pair_cfg)
        if not pairs.empty:
            ev = build_event_dataset(prices, pairs, formation_end=start + formation_days,
                                     score_end=start + formation_days + step, cfg=cfg, context=context)
            if not ev.empty:
                ev["reform_start"] = prices.index[start]
                frames.append(ev)
        start += step
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
