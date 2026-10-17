"""Opponent-adjusted team ratings (offense and defense) for several stats.

For each stat m and every completed team-game before the cutoff:
    stat(team offense in game) = mu + O[team] + D[opponent] + h * venue
solved by weighted ridge regression.  Weights decay with game age, and games
from earlier seasons are discounted further, so early-season ratings lean on
last year (regressed toward average by the ridge penalty).

O[team] > 0: better than average offense for that stat.
D[team] > 0: defense ALLOWS more than average (worse defense).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

METRICS = {
    'CFB': ['points', 'sr', 'expl', 'ypp'],
    'NFL': ['points', 'sr', 'expl', 'ypp', 'pass_epa', 'rush_epa'],
}


@dataclass
class RatingSettings:
    half_life_days: float = 240.0      # game weight halves every N days (tuned on 2016-2019 only)
    prior_season_factor: float = 0.6   # extra multiplier for games from earlier seasons
    window_days: float = 700.0         # ignore games older than this
    ridge: dict = field(default_factory=lambda: {'points': 2.0, 'sr': 2.0, 'expl': 2.0, 'ypp': 2.0,
                                                  'pass_epa': 2.0, 'rush_epa': 2.0})


def default_settings(league: str) -> RatingSettings:
    """Tuned on 2016-2019 only (prediction error), never on later seasons."""
    if league == 'NFL':
        return RatingSettings(half_life_days=240.0, prior_season_factor=0.6,
                              ridge={'points': 12.0, 'sr': 0.05, 'expl': 0.05, 'ypp': 0.05,
                                     'pass_epa': 0.05, 'rush_epa': 0.05})
    return RatingSettings(half_life_days=240.0, prior_season_factor=0.6,
                          ridge={'points': 0.25, 'sr': 0.25, 'expl': 0.25, 'ypp': 0.25})


def _fit_metric(rows: pd.DataFrame, m: str, teams: list[str], w: np.ndarray, ridge: float):
    ok = rows[m].notna().values
    if ok.sum() < 30:
        return None
    r = rows[ok]; ww = w[ok]
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(r), 2 * n + 2))
    X[:, 0] = 1.0                                   # mu
    X[np.arange(len(r)), 1 + r['team'].map(idx).values] = 1.0
    X[np.arange(len(r)), 1 + n + r['opp'].map(idx).values] = 1.0
    X[:, -1] = r['venue'].values * 0.5              # h = home-minus-away difference
    y = r[m].values.astype(float)
    sw = np.sqrt(ww)[:, None]
    A = X * sw; b = y * sw[:, 0]
    # ridge = prior strength in "games": every rating starts as if it had `ridge`
    # average-weight games at league average (unitless, same for every stat)
    reg = np.full(X.shape[1], ridge); reg[0] = 0.0; reg[-1] = 0.0
    beta = np.linalg.solve(A.T @ A + np.diag(reg) + 1e-9 * np.eye(X.shape[1]), A.T @ b)
    return dict(mu=beta[0], O=dict(zip(teams, beta[1:1 + n])), D=dict(zip(teams, beta[1 + n:1 + 2 * n])),
                h=beta[-1])


def fit_ratings(games: pd.DataFrame, tg: pd.DataFrame, cutoff, season: int, league: str,
                settings: RatingSettings | None = None):
    """Ratings using only games completed before `cutoff` (a UTC timestamp)."""
    s = settings or default_settings(league)
    g = games[(games.kickoff < cutoff) & games.completed &
              (games.kickoff > cutoff - pd.Timedelta(days=s.window_days))]
    if len(g) < 40:
        return None
    rows = tg[tg.id.isin(g.id)].merge(g[['id', 'kickoff', 'season']], on='id')
    rows = rows[rows.team.notna() & rows.opp.notna()]
    age = (cutoff - rows['kickoff']).dt.total_seconds().values / 86400.0
    w = 0.5 ** (age / s.half_life_days)
    w = w * np.where(rows['season'].values < season, s.prior_season_factor, 1.0)
    # weight rate stats by plays (bounded) so tiny samples count less
    teams = sorted(set(rows.team) | set(rows.opp), key=str)
    out = {'teams': teams, 'metrics': {}, 'games_used': int(len(g))}
    for m in METRICS[league]:
        mw = w.copy()
        if m != 'points':
            plays = rows['plays'].fillna(rows['plays'].median()).clip(20, 90).values
            mw = mw * plays / 65.0
        fit = _fit_metric(rows, m, teams, mw, s.ridge.get(m, 6.0))
        if fit:
            out['metrics'][m] = fit
    # team game counts this season (for sample-size flags)
    cur = rows[rows.season == season]
    out['games_this_season'] = cur.groupby('team').size().to_dict()
    return out


def fit_market_ratings(games: pd.DataFrame, cutoff, season: int, half_life_days: float = 100.0,
                       ridge: float = 0.5):
    """Team ratings implied by earlier CLOSING spreads (what the market thinks).

    spread_home(away-minus-home margin) = R[away] - R[home] - hfa * (not neutral)
    """
    g = games[(games.kickoff < cutoff) & games.spread_home.notna() &
              (games.kickoff > cutoff - pd.Timedelta(days=500))]
    if len(g) < 40:
        return None
    g = g[g.away.notna() & g.home.notna()]
    teams = sorted(set(g.away) | set(g.home), key=str); idx = {t: i for i, t in enumerate(teams)}
    age = (cutoff - g.kickoff).dt.total_seconds().values / 86400.0
    w = 0.5 ** (age / half_life_days) * np.where(g.season.values < season, 0.5, 1.0)
    X = np.zeros((len(g), len(teams) + 1))
    X[np.arange(len(g)), g.away.map(idx).values] = 1.0
    X[np.arange(len(g)), g.home.map(idx).values] = -1.0
    X[:, -1] = -(1 - g.neutral.astype(int).values)
    sw = np.sqrt(w)[:, None]
    A = X * sw; b = g.spread_home.values * sw[:, 0]
    reg = np.full(X.shape[1], ridge); reg[-1] = 0.0
    beta = np.linalg.solve(A.T @ A + np.diag(reg), A.T @ b)
    return dict(R=dict(zip(teams, beta[:-1])), hfa=float(beta[-1]))


def matchup_features(r: dict, away: str, home: str, league: str, mr: dict | None = None) -> dict:
    """Neutral-field expected stat gaps (away minus home) for one matchup."""
    f = {}
    if mr is not None:
        R = mr['R']
        f['market_gap'] = R[away] - R[home] if away in R and home in R else np.nan
    if r is None:
        return f
    for m, fit in r['metrics'].items():
        O, D = fit['O'], fit['D']
        if away not in O or home not in O:
            f[f'gap_{m}'] = np.nan
            if m == 'points':
                f['sum_points'] = np.nan
            continue
        exp_away = fit['mu'] + O[away] + D[home]
        exp_home = fit['mu'] + O[home] + D[away]
        f[f'gap_{m}'] = exp_away - exp_home
        if m == 'points':
            f['sum_points'] = exp_away + exp_home
    f['away_games'] = r['games_this_season'].get(away, 0)
    f['home_games'] = r['games_this_season'].get(home, 0)
    return f


def team_table(r: dict, league: str) -> pd.DataFrame:
    """Readable per-team ratings (for the Ratings tab)."""
    if not r:
        return pd.DataFrame()
    rows = []
    for t in r['teams']:
        row = {'team': t}
        for m, fit in r['metrics'].items():
            row[f'off_{m}'] = fit['O'].get(t, np.nan)
            row[f'def_{m}'] = -fit['D'].get(t, np.nan)      # flip sign: higher = better defense
        row['games'] = r['games_this_season'].get(t, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def weekly_ratings(games: pd.DataFrame, tg: pd.DataFrame, league: str, seasons=None,
                   settings: RatingSettings | None = None):
    """Ratings as of the first kickoff of each (season, week).  Used for history/backtests."""
    out = {}
    sel = games if seasons is None else games[games.season.isin(seasons)]
    for (season, week), wk in sel.groupby(['season', 'week']):
        cutoff = wk.kickoff.min() - pd.Timedelta(hours=1)
        out[(season, week)] = fit_ratings(games, tg, cutoff, season, league, settings)
    return out


def weekly_market_ratings(games: pd.DataFrame, seasons=None):
    out = {}
    sel = games if seasons is None else games[games.season.isin(seasons)]
    for (season, week), wk in sel.groupby(['season', 'week']):
        cutoff = wk.kickoff.min() - pd.Timedelta(hours=1)
        out[(season, week)] = fit_market_ratings(games, cutoff, season)
    return out
