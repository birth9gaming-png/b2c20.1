"""Situational spots, defined the same way for live games and for the history test.

lookahead : team is favored now (CFB 7+, NFL 3+), its NEXT game (within 9 days)
            is against an opponent rated within 3 points of it or better, and that
            next opponent is much stronger than today's (CFB 14+, NFL 7+ points).
letdown   : team's PREVIOUS game (within 9 days) was an outright win as an
            underdog of 3+, and it is favored today.
sandwich  : both at once.
Ratings used to size opponents are the market-implied ratings (from earlier
closing lines), so the definitions use only information known before kickoff.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FAV_CUT = {'CFB': 7.0, 'NFL': 3.0}
STEP_UP = {'CFB': 14.0, 'NFL': 7.0}


def team_schedule(games: pd.DataFrame) -> dict:
    sched = {}
    for r in games.itertuples():
        for side, team, opp in (('away', r.away, r.home), ('home', r.home, r.away)):
            sched.setdefault((r.season, team), []).append(r)
    for k in sched:
        sched[k].sort(key=lambda x: x.kickoff)
    return sched


def spots_for_game(g, league: str, sched: dict, ratings: dict | None, current_home=None) -> dict:
    """Flags for both teams of one game.  `current_home` = today's home spread (else g.spread_home)."""
    R = (ratings or {}).get('R', {})
    line = current_home if current_home is not None and pd.notna(current_home) else g.spread_home
    out = {}
    for side, team, opp in (('away', g.away, g.home), ('home', g.home, g.away)):
        info = dict(lookahead=False, letdown=False, sandwich=False, next_opp=None, prev=None, rest_days=None)
        games = sched.get((g.season, team), [])
        pos = next((i for i, x in enumerate(games) if x.id == g.id), None)
        if pos is None or line is None or pd.isna(line):
            out[side] = info
            continue
        my_margin = (line if side == 'away' else -line)        # team's expected margin today
        nxt = games[pos + 1] if pos + 1 < len(games) else None
        prv = games[pos - 1] if pos > 0 else None
        if prv is not None:
            info['rest_days'] = (g.kickoff - prv.kickoff).days
        if nxt is not None and (nxt.kickoff - g.kickoff).days <= 9:
            nopp = nxt.home if nxt.away == team else nxt.away
            info['next_opp'] = nopp
            if team in R and nopp in R and opp in R:
                info['lookahead'] = bool(my_margin >= FAV_CUT[league] and R[nopp] - R[team] >= -3
                                         and R[nopp] - R[opp] >= STEP_UP[league])
        if prv is not None and (g.kickoff - prv.kickoff).days <= 9 and prv.completed and pd.notna(prv.spread_home):
            pside = 'away' if prv.away == team else 'home'
            exp = prv.spread_home if pside == 'away' else -prv.spread_home
            act = (prv.away_score - prv.home_score) * (1 if pside == 'away' else -1)
            popp = prv.home if pside == 'away' else prv.away
            info['prev'] = dict(opp=popp, won=bool(act > 0), was_dog_by=round(-exp, 1) if exp < 0 else 0.0)
            info['letdown'] = bool(exp <= -3 and act > 0 and my_margin > 0)
        info['sandwich'] = info['lookahead'] and info['letdown']
        out[side] = info
    return out


def describe(side_info: dict, team: str) -> list[str]:
    msgs = []
    if side_info.get('lookahead'):
        msgs.append(f"{team} in a LOOKAHEAD spot (big game vs {side_info['next_opp']} next)")
    if side_info.get('letdown'):
        p = side_info.get('prev') or {}
        msgs.append(f"{team} in a LETDOWN spot (won outright as a {p.get('was_dog_by', 0):g}-pt dog vs {p.get('opp')})")
    if side_info.get('sandwich'):
        msgs.append(f"{team} in a SANDWICH spot")
    return msgs


def history_records(games: pd.DataFrame, league: str, market_ratings_by_week: dict) -> dict:
    """ATS record of FADING the team in each spot, at closing lines (for display)."""
    sched = team_schedule(games)
    recs = {k: [0, 0] for k in ('lookahead', 'letdown', 'sandwich')}
    for g in games.itertuples():
        if g.season_type != 'regular' or not g.completed or pd.isna(g.spread_home):
            continue
        if league == 'CFB' and 'FCS' in (g.away, g.home):
            continue
        s = spots_for_game(g, league, sched, market_ratings_by_week.get((g.season, g.week)))
        cover = np.sign((g.away_score - g.home_score) - g.spread_home)     # +1 away covered
        if cover == 0:
            continue
        for k in recs:
            a, h = s['away'][k], s['home'][k]
            if a == h:
                continue
            won = (cover < 0) if a else (cover > 0)                         # fade the flagged team
            recs[k][0 if won else 1] += 1
    return {k: dict(w=v[0], l=v[1], pct=round(v[0] / max(1, sum(v)), 4)) for k, v in recs.items()}
