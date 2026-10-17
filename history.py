"""Load and normalize game + team-game data into one shape for both leagues.

games table (one row per game)
    id, league, season, week, season_type, kickoff (UTC), away, home, neutral,
    away_score, home_score, completed, spread_home, total
    spread_home is the closing (or latest) home spread: negative = home favored.

team-game table (one row per team per game, offense view)
    id, team, opp, venue (+1 home, -1 away, 0 neutral), points, plays,
    ypp, sr (success rate 0-1), expl (CFB explosiveness / NFL EPA per play),
    pass_epa, rush_epa (NFL only), giveaways
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HIST = ROOT                      # all files live in one folder
FCS = 'FCS'          # all FCS opponents share one rating in CFB


def _boolish(s: pd.Series) -> pd.Series:
    """True/False/'True'/'false'/1/0/NaN -> clean bool (CSV round trips and older pandas differ)."""
    return s.map(lambda v: str(v).strip().lower() in ('true', '1', '1.0', 'yes')).astype(bool)


# ---------------------------------------------------------------- CFB
def cfb_games_frame(games: pd.DataFrame, lines: pd.DataFrame | None) -> pd.DataFrame:
    g = games.copy()
    g['id'] = g['id'].astype(str)
    g['league'] = 'CFB'
    g['kickoff'] = pd.to_datetime(g['kickoff'], utc=True, format='ISO8601', errors='coerce')
    for side in ('away', 'home'):
        cls = g.get(f'{side}_classification')
        if cls is not None:
            g.loc[cls.fillna('fbs').str.lower() != 'fbs', side] = FCS
    g = g[g['away'].notna() & g['home'].notna()].copy()            # TBD / placeholder games
    g['away'] = g['away'].astype(str); g['home'] = g['home'].astype(str)
    for c in ('away_score', 'home_score'):
        g[c] = pd.to_numeric(g[c], errors='coerce')
    g['neutral'] = _boolish(g['neutral'])
    g['completed'] = _boolish(g['completed']) & g.away_score.notna() & g.home_score.notna()
    g['season_type'] = g['season_type'].fillna('regular') if 'season_type' in g else 'regular'
    if lines is not None and len(lines):
        ln = lines.copy(); ln['id'] = ln['id'].astype(str)
        ln = ln.drop_duplicates('id').set_index('id')
        g['spread_home'] = g['id'].map(ln['spread_home'])
        g['total'] = g['id'].map(ln['total'])
    else:
        g['spread_home'] = np.nan; g['total'] = np.nan
    cols = ['id', 'league', 'season', 'week', 'season_type', 'kickoff', 'away', 'home', 'neutral',
            'away_score', 'home_score', 'completed', 'spread_home', 'total']
    return g[cols].drop_duplicates('id', keep='last').sort_values('kickoff').reset_index(drop=True)


def cfb_team_games(games: pd.DataFrame, advanced: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Offense rows for FBS teams.  FCS teams only appear as opponents (pooled)."""
    adv = advanced.copy(); adv['id'] = adv['id'].astype(str)
    adv = adv.rename(columns={'offense.successRate': 'sr', 'offense.explosiveness': 'expl',
                              'offense.plays': 'adv_plays', 'defense.successRate': 'sr_allowed',
                              'defense.explosiveness': 'expl_allowed', 'defense.plays': 'adv_plays_allowed'})
    bx = box.copy(); bx['id'] = bx['id'].astype(str)
    g = games.set_index('id')
    rows = []
    for side, opp_side, venue in (('away', 'home', -1), ('home', 'away', 1)):
        part = games[['id', side, opp_side, f'{side}_score', 'neutral', 'completed']].rename(
            columns={side: 'team', opp_side: 'opp', f'{side}_score': 'points'})
        part['venue'] = np.where(part['neutral'], 0, venue)
        rows.append(part)
    tg = pd.concat(rows, ignore_index=True)
    tg = tg[tg.completed.astype(bool)]
    for c in ('sr', 'expl', 'adv_plays', 'sr_allowed', 'expl_allowed', 'adv_plays_allowed'):
        if c in adv:
            adv[c] = pd.to_numeric(adv[c], errors='coerce')
    for c in ('plays', 'yards', 'giveaways'):
        if c in bx:
            bx[c] = pd.to_numeric(bx[c], errors='coerce')
    bx['team'] = bx['team'].astype(str); adv['team'] = adv['team'].astype(str)
    # CFBD (and our weekly cache) can list the same game twice; one row per team per game
    adv = adv.drop_duplicates(['id', 'team'], keep='last')
    bx = bx.drop_duplicates(['id', 'team'], keep='last')
    tg = tg.drop_duplicates(['id', 'team'], keep='last')
    tg = tg.merge(adv[['id', 'team', 'sr', 'expl', 'adv_plays', 'sr_allowed', 'expl_allowed', 'adv_plays_allowed']],
                  on=['id', 'team'], how='left')
    tg = tg.merge(bx[['id', 'team', 'plays', 'yards', 'giveaways']], on=['id', 'team'], how='left')
    tg['ypp'] = tg['yards'] / tg['plays'].where(tg['plays'] > 0)
    # FCS offense rows: use the FBS opponent's 'allowed' advanced numbers when available
    tg = tg.reset_index(drop=True)
    fcs = (tg['team'] == FCS).to_numpy()
    if fcs.any():
        mirror = tg.loc[~fcs, ['id', 'team', 'sr_allowed', 'expl_allowed', 'adv_plays_allowed']].rename(
            columns={'team': 'opp', 'sr_allowed': 'm_sr', 'expl_allowed': 'm_expl', 'adv_plays_allowed': 'm_plays'})
        tg = tg.merge(mirror.drop_duplicates(['id', 'opp']), on=['id', 'opp'], how='left')
        fcs = (tg['team'] == FCS).to_numpy()            # recompute after the merge so rows line up
        for col, src in (('sr', 'm_sr'), ('expl', 'm_expl'), ('adv_plays', 'm_plays')):
            tg[col] = np.where(fcs, tg[src], tg[col])
        tg = tg.drop(columns=['m_sr', 'm_expl', 'm_plays'])
    tg['plays'] = tg['plays'].fillna(tg['adv_plays'])
    tg['pass_epa'] = np.nan; tg['rush_epa'] = np.nan
    keep = ['id', 'team', 'opp', 'venue', 'points', 'plays', 'ypp', 'sr', 'expl', 'pass_epa', 'rush_epa', 'giveaways']
    return tg[keep].reset_index(drop=True)


# ---------------------------------------------------------------- NFL
NFL_ALIASES = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA', 'JAC': 'JAX', 'WSH': 'WAS'}


def nfl_code(x):
    x = str(x).upper().removeprefix('NFL:')
    return NFL_ALIASES.get(x, x)


def nfl_games_frame(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g['id'] = g['id'].astype(str)
    g['league'] = 'NFL'
    g['kickoff'] = pd.to_datetime(g['kickoff'], utc=True, format='ISO8601', errors='coerce')
    g['away'] = g['away'].map(nfl_code); g['home'] = g['home'].map(nfl_code)
    for c in ('away_score', 'home_score'):
        g[c] = pd.to_numeric(g[c], errors='coerce')
    g['neutral'] = _boolish(g['neutral'])
    g['completed'] = _boolish(g['completed']) & g.away_score.notna() & g.home_score.notna()
    if 'spread_home' not in g:
        g['spread_home'] = -g['line_away']           # line_away = away spread (+ = away getting points)
    if 'total' not in g:
        g['total'] = g.get('market_total')
    cols = ['id', 'league', 'season', 'week', 'season_type', 'kickoff', 'away', 'home', 'neutral',
            'away_score', 'home_score', 'completed', 'spread_home', 'total']
    out = g[cols].sort_values('kickoff').reset_index(drop=True)
    return out


def nfl_team_games(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    st = stats.copy(); st['id'] = st['id'].astype(str); st['team'] = st['team'].map(nfl_code)
    rows = []
    for side, opp_side, venue in (('away', 'home', -1), ('home', 'away', 1)):
        part = games[['id', side, opp_side, f'{side}_score', 'neutral', 'completed']].rename(
            columns={side: 'team', opp_side: 'opp', f'{side}_score': 'points'})
        part['venue'] = np.where(part['neutral'], 0, venue)
        rows.append(part)
    tg = pd.concat(rows, ignore_index=True)
    tg = tg[tg.completed].merge(st, on=['id', 'team'], how='left')
    bad = tg['invalid_core_plays'].fillna(0) > 0
    p = tg['plays'].where((tg['plays'] > 0) & ~bad)
    tg['ypp'] = tg['yards'] / p
    tg['sr'] = tg['successes'] / p
    tg['expl'] = tg['epa_sum'] / p                    # EPA per play
    tg['pass_epa'] = tg['pass_epa_sum'] / tg['dropbacks'].where(tg['dropbacks'] > 0)
    tg['rush_epa'] = tg['rush_epa_sum'] / tg['rush_plays'].where(tg['rush_plays'] > 0)
    tg['giveaways'] = -tg['net_turnovers']           # net turnovers lost (approximation)
    keep = ['id', 'team', 'opp', 'venue', 'points', 'plays', 'ypp', 'sr', 'expl', 'pass_epa', 'rush_epa', 'giveaways']
    return tg[keep].reset_index(drop=True)


# ---------------------------------------------------------------- loaders
def load_history(league: str):
    """Return (games, team_games) from the bundled history files."""
    if league == 'CFB':
        games = pd.read_csv(HIST / 'cfb_games.csv.gz')
        lines = pd.read_csv(HIST / 'cfb_lines.csv.gz')
        adv = pd.read_csv(HIST / 'cfb_advanced.csv.gz')
        box = pd.read_csv(HIST / 'cfb_box.csv.gz')
        g = cfb_games_frame(games, lines)
        tg = cfb_team_games(g, adv, box)
    else:
        games = pd.read_csv(HIST / 'nfl_games.csv.gz')
        stats = pd.read_csv(HIST / 'nfl_team_games.csv.gz')
        g = nfl_games_frame(games)
        tg = nfl_team_games(g, stats)
    return g, tg


def combine(base_games, base_tg, new_games, new_tg):
    """Merge freshly downloaded games/team-games over the bundled history (new wins)."""
    new_games = new_games.copy()
    new_games['kickoff'] = pd.to_datetime(new_games['kickoff'], utc=True, format='ISO8601', errors='coerce')
    g = pd.concat([base_games[~base_games.id.isin(new_games.id)], new_games], ignore_index=True)
    g = g[g.kickoff.notna()]
    tg = pd.concat([base_tg[~base_tg.id.isin(new_tg.id)], new_tg], ignore_index=True)
    return g.sort_values('kickoff').reset_index(drop=True), tg.reset_index(drop=True)
