"""Spread model: opponent-adjusted stat ratings + market-implied ratings -> margin.

Training is always walk-forward for the honest backtest (season Y is predicted
by a model fit only on seasons before Y).  The live model is then refit on all
seasons.  Cover probabilities come from a logistic fit of real outcomes on the
walk-forward edges, so a "56%" means games like it historically covered ~56%.
"""
from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from ratings import (METRICS, RatingSettings, default_settings, fit_ratings, fit_market_ratings, matchup_features,
                      weekly_ratings, weekly_market_ratings)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT


def feature_names(league: str) -> list[str]:
    return [f'gap_{m}' for m in METRICS[league]] + ['market_gap', 'hfa']


# ------------------------------------------------------------------ features
def history_features(games: pd.DataFrame, tg: pd.DataFrame, league: str, settings=None) -> pd.DataFrame:
    """One row per completed regular-season game with as-of-that-week features."""
    wr = weekly_ratings(games, tg, league, settings=settings)
    mr = weekly_market_ratings(games)
    rows = []
    for r in games.itertuples():
        if r.season_type != 'regular' or not r.completed:
            continue
        f = matchup_features(wr.get((r.season, r.week)), r.away, r.home, league, mr.get((r.season, r.week)))
        f.update(id=r.id, season=int(r.season), week=int(r.week), away=r.away, home=r.home,
                 neutral=int(r.neutral), hfa=-(1 - int(r.neutral)), margin=r.away_score - r.home_score,
                 actual_total=r.away_score + r.home_score, spread_home=r.spread_home, total=r.total)
        rows.append(f)
    df = pd.DataFrame(rows)
    if league == 'CFB':
        df = df[(df.away != 'FCS') & (df.home != 'FCS')]
    return df.reset_index(drop=True)


def slate_features(games_hist, tg, slate: pd.DataFrame, league: str, now=None, settings=None):
    """Features for upcoming games using everything completed before `now`."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize('UTC')
    season = int(slate.season.max()) if len(slate) else now.year
    r = fit_ratings(games_hist, tg, now, season, league, settings)
    mr = fit_market_ratings(games_hist, now, season)
    rows = []
    for g in slate.itertuples():
        f = matchup_features(r, g.away, g.home, league, mr)
        f.update(id=g.id, hfa=-(1 - int(g.neutral)))
        rows.append(f)
    return pd.DataFrame(rows), r, mr


# ------------------------------------------------------------------ fitting
def _fit_margin(df, feats):
    m = LinearRegression().fit(df[feats], df['margin'])
    return dict(coef=dict(zip(feats, m.coef_.tolist())), intercept=float(m.intercept_))


def _fit_total(df):
    d = df[df.sum_points.notna() & df.actual_total.notna()]
    if len(d) < 200:
        return None
    m = LinearRegression().fit(d[['sum_points']], d['actual_total'])
    return dict(coef=float(m.coef_[0]), intercept=float(m.intercept_))


def predict_margin(model: dict, feats: pd.DataFrame) -> np.ndarray:
    out = np.full(len(feats), model['margin']['intercept'], dtype=float)
    for k, c in model['margin']['coef'].items():
        out = out + c * feats[k].astype(float).values
    return out


def predict_total(model: dict, feats: pd.DataFrame):
    t = model.get('total')
    if not t or 'sum_points' not in feats:
        return np.full(len(feats), np.nan)
    return t['intercept'] + t['coef'] * feats['sum_points'].astype(float).values


RANK_BANDS = ((0.0, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 1.01))
PSEUDO_GAMES = 20          # shrink each band's hit rate toward 50% by 20 imaginary games


def week_rank_fraction(abs_edges: pd.Series, groups) -> pd.Series:
    """0 = biggest disagreement of the week, ~1 = smallest."""
    r = abs_edges.groupby(groups).rank(ascending=False, method='first')
    n = abs_edges.groupby(groups).transform('count')
    return (r - 1) / n


def rank_table(frac: pd.Series, result: pd.Series) -> list[dict]:
    rows = []
    for lo, hi in RANK_BANDS:
        m = (frac >= lo) & (frac < hi)
        w, l = int((result[m] > 0).sum()), int((result[m] < 0).sum())
        p = (w + PSEUDO_GAMES * 0.5) / (w + l + PSEUDO_GAMES)
        rows.append(dict(lo=lo, hi=hi, w=w, l=l, p=round(p, 4)))
    return rows


def band_prob(table: list[dict], frac: float) -> float:
    for row in table:
        if row['lo'] <= frac < row['hi']:
            return row['p']
    return 0.5


def cover_prob_model_side(model: dict, frac: float, at_open: bool = False) -> float:
    """Historical cover rate for the side the model favors, given this game's
    rank of disagreement within the week (0 = biggest)."""
    key = 'rank_open' if at_open and model.get('calibration', {}).get('rank_open') else 'rank_close'
    return band_prob(model['calibration'][key], frac)


def _rec(x):
    x = np.asarray(x)
    w, l = int((x > 0).sum()), int((x < 0).sum())
    return dict(w=w, l=l, pct=round(w / max(1, w + l), 4))


def walk_forward(df: pd.DataFrame, league: str, first_test: int = 2018, openers: pd.Series | None = None):
    feats = feature_names(league)
    d = df[df[feats].notna().all(axis=1)].copy()
    out = []
    for Y in sorted(d.season.unique()):
        if Y < first_test:
            continue
        tr, te = d[d.season < Y], d[d.season == Y].copy()
        mdl = {'margin': _fit_margin(tr, feats)}
        te['pred'] = predict_margin(mdl, te)
        out.append(te)
    res = pd.concat(out)
    res = res[res.spread_home.notna()].copy()
    res['edge'] = res.pred - res.spread_home
    res['cover'] = np.sign(res.margin - res.spread_home)          # +1 away covered
    res['result'] = np.where(res.edge > 0, res.cover, -res.cover)  # result of betting the model side
    if openers is not None:
        res['open_home'] = res.id.map(openers)
    return res


def backtest_summary(res: pd.DataFrame, league: str, slope: float) -> dict:
    r = res.copy()
    r['ae'] = r.edge.abs()
    r['p'] = 1 / (1 + np.exp(-slope * r.ae))
    s = dict(seasons=f'{int(r.season.min())}-{int(r.season.max())}', games=int(len(r)),
             model_mae=round(float((r.pred - r.margin).abs().mean()), 2),
             market_mae=round(float((r.spread_home - r.margin).abs().mean()), 2),
             every_game=_rec(r.result))
    for k in (1, 2, 3):
        t = r.sort_values('ae', ascending=False).groupby(['season', 'week']).head(k)
        s[f'top{k}_per_week'] = _rec(t.result)
    big = {'CFB': (3, 6, 10), 'NFL': (1.5, 3, 4.5)}[league]
    s['by_edge'] = {}
    for lo, hi in zip((0,) + big, big + (99,)):
        m = (r.ae >= lo) & (r.ae < hi)
        s['by_edge'][f'{lo}-{hi if hi < 99 else "+"}'] = _rec(r.result[m])
    # veto: how often the side the model is against covers
    s['veto'] = {}
    for x in big:
        against = np.concatenate([r.cover[r.edge <= -x].values, -r.cover[r.edge >= x].values])
        s['veto'][f'{x}+'] = _rec(against)
    s['by_season'] = {int(k): _rec(g.result) for k, g in r.groupby('season')}
    s['p_max'] = round(float(r.p.max()), 3)
    s['p_bands'] = {}
    for lo, hi in ((0.5, 0.52), (0.52, 0.54), (0.54, 0.56), (0.56, 1.0)):
        m = (r.p >= lo) & (r.p < hi)
        s['p_bands'][f'{lo:.2f}-{hi:.2f}'] = dict(n=int(m.sum()), **_rec(r.result[m]))
    if 'open_home' in r and r.open_home.notna().any():
        o = r[r.open_home.notna() & ((r.spread_home - r.open_home).abs() <= 7)].copy()
        o['eo'] = o.pred - o.open_home
        o['ro'] = np.where(o.eo > 0, np.sign(o.margin - o.open_home), -np.sign(o.margin - o.open_home))
        o['clv'] = np.sign(o.eo) * (o.spread_home - o.open_home)
        o['aeo'] = o.eo.abs()
        s['vs_open'] = dict(games=int(len(o)), every_game=_rec(o.ro), avg_clv=round(float(o.clv.mean()), 2),
                            moved_our_way=round(float((o.clv > 0).mean()), 3),
                            moved_against=round(float((o.clv < 0).mean()), 3))
        for k in (2, 3):
            t = o.sort_values('aeo', ascending=False).groupby(['season', 'week']).head(k)
            s['vs_open'][f'top{k}_per_week'] = dict(**_rec(t.ro), avg_clv=round(float(t.clv.mean()), 2))
    return s


def train(df: pd.DataFrame, league: str, openers: pd.Series | None = None, settings=None) -> dict:
    """Walk-forward backtest + calibration, then refit on everything for live use."""
    feats = feature_names(league)
    res = walk_forward(df, league, openers=openers)
    dec = res[res.cover != 0]
    lr = LogisticRegression(C=1.0, fit_intercept=False).fit(dec[['edge']].values, (dec.cover > 0).astype(int))
    slope = float(lr.coef_[0][0])
    groups = [res.season, res.week]
    frac = week_rank_fraction(res.edge.abs(), groups)
    calibration = dict(slope=slope, rank_close=rank_table(frac, res.result),
                       method='historical cover rate by weekly rank of model-vs-line disagreement '
                              f'(walk-forward 2018+; each band shrunk toward 50% by {PSEUDO_GAMES} games)')
    if 'open_home' in res and res.open_home.notna().any():
        o = res[res.open_home.notna() & ((res.spread_home - res.open_home).abs() <= 7)].copy()
        eo = o.pred - o.open_home
        ro = np.where(eo > 0, np.sign(o.margin - o.open_home), -np.sign(o.margin - o.open_home))
        fo = week_rank_fraction(eo.abs(), [o.season, o.week])
        calibration['rank_open'] = rank_table(fo, pd.Series(ro, index=o.index))
    full = df[df[feats].notna().all(axis=1)]
    model = dict(league=league, version='18.0', trained_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 seasons=[int(full.season.min()), int(full.season.max())], games=int(len(full)),
                 features=feats, margin=_fit_margin(full, feats), total=_fit_total(full),
                 calibration=calibration,
                 residual_sd=float(np.std(full.margin - predict_margin({'margin': _fit_margin(full, feats)}, full))),
                 rating_settings=(settings or default_settings(league)).__dict__,
                 backtest=backtest_summary(res, league, slope))
    return model


def save_model(model: dict, path: Path | None = None):
    path = path or MODEL_DIR / f"model_{model['league']}.json"
    path.write_text(json.dumps(model, indent=1))
    return path


def load_model(league: str, store=None) -> dict | None:
    if store is not None:
        m = store.get_json(f'models/model_{league}.json')
        if m:
            return m
    p = MODEL_DIR / f'model_{league}.json'
    return json.loads(p.read_text()) if p.exists() else None
