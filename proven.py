"""What has actually beaten the closing line, measured on 2016-2025 games (Oct 16 2026).

The Vegas-anchored model starts at the market line and only moves off it for these spots.
Every reason shown in the app carries its real record from the same test, so you can see which
of your spots are proven and which are just context.

How it was measured: every game side in power-conference games (college) / every game (NFL),
graded against the CLOSING spread, using our own definitions of each spot (rules.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# points the anchored model moves toward the side (about half the side's average cover margin - shrunk)
PROVEN = {
    'CFB': {'rvr_fav': dict(pts=1.5, rate=0.573, record='240-179', label='favorite in a ranked-vs-ranked game')},
    'NFL': {'dog_vs_look': dict(pts=1.0, rate=0.538, record='370-318', label='underdog vs a favorite looking ahead')},
}
# your other spots, same test (context - not proven on their own)
RECORDS = {
    'CFB': {'lookahead': ('50.1%', '841-837'), 'letdown': ('48.8%', '250-262'), 'trap': ('51.5%', '53-50'),
            'move': ('50.8%', '1335-1295'), 'rival_fav': ('52.8%', '314-281')},
    'NFL': {'lookahead': ('51.1%', '582-557'), 'letdown': ('49.8%', '211-213'), 'trap': ('47.0%', '55-62'),
            'move': (None, None), 'rival_fav': ('48.4%', '455-486')},
}
# the stats model's gap with the market barely comes true (slope 0.03-0.06 in college, none in the NFL)
STATS_WEIGHT = {'CFB': 0.05, 'NFL': 0.0}


def record_text(league: str, key: str) -> str:
    rate, rec = RECORDS.get(league, {}).get(key, (None, None))
    return f' ({rate} since 2016)' if rate else ''


def proven_for_side(league: str, side: str, fl: dict, line_side: float) -> list[dict]:
    """Proven spots backing `side` (fl = rules.flag_games row as a dict)."""
    other = 'home' if side == 'away' else 'away'
    out = []
    if line_side is None or pd.isna(line_side):
        return out
    if league == 'CFB':
        if fl.get(f'{side}_rank') is not None and pd.notna(fl.get(f'{side}_rank')) and \
                fl.get(f'{other}_rank') is not None and pd.notna(fl.get(f'{other}_rank')) and line_side < 0:
            out.append(dict(key='rvr_fav', **PROVEN['CFB']['rvr_fav']))
    else:
        if line_side > 0 and fl.get(f'{other}_lookahead'):
            out.append(dict(key='dog_vs_look', **PROVEN['NFL']['dog_vs_look']))
    return out


def anchor(slate: pd.DataFrame, flags: dict, league: str) -> pd.DataFrame:
    """Vegas-anchored model line for every game.

    model_home = market home spread
                 - points for proven spots backing home  + points for proven spots backing away
                 + a small slice of the stats model's disagreement (STATS_WEIGHT)
    Most games land exactly on the market; the few that don't are the bets."""
    s = slate.copy()
    adj, notes = [], []
    for r in s.to_dict('records'):
        cur = r.get('current_home')
        fl = flags.get(r['id'], {})
        a, n = 0.0, {'away': [], 'home': []}
        if cur is not None and pd.notna(cur):
            for side in ('away', 'home'):
                line_side = -cur if side == 'away' else cur
                for p in proven_for_side(league, side, fl, line_side):
                    a += p['pts'] if side == 'away' else -p['pts']     # +: model likes away more than market
                    n[side].append(p)
        adj.append(a); notes.append(n)
    s['proven_adjust'] = adj
    s['proven'] = notes
    w = STATS_WEIGHT.get(league, 0.0)
    stats_gap = (s['stats_home'] - s['current_home']).fillna(0.0)
    s['model_home'] = s['current_home'] + s['proven_adjust'] + w * stats_gap
    return s


def side_prob(league: str, row: dict, side: str) -> float:
    """Cover chance for `side`: the proven spot's record, else a coin flip."""
    ps = (row.get('proven') or {}).get(side) or []
    if ps:
        return float(np.mean([p['rate'] for p in ps])) - 0.01        # small shrink toward 50%
    return 0.5
