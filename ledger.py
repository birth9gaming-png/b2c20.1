"""Pick log: what you bet, at what number, what the model said, the closing line and the result.

CLV (closing-line value) in points = your number minus the closing number for
your side.  Positive = you beat the close.  Over a few hundred bets, average
CLV is the most reliable sign of a real edge — far faster than win-loss record.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd

KEY = 'ledger/picks.json'


def load(store) -> list[dict]:
    return store.get_json(KEY, []) or []


def save_all(store, picks: list[dict]):
    store.put_json(KEY, picks)


def add(store, pick: dict) -> dict:
    picks = load(store)
    pick = dict(pick)
    pick.setdefault('pick_id', uuid.uuid4().hex[:10])
    pick.setdefault('logged_at', datetime.now(timezone.utc).isoformat(timespec='seconds'))
    pick.setdefault('status', 'open')
    picks.append(pick)
    save_all(store, picks)
    return pick


def remove(store, pick_id: str):
    save_all(store, [p for p in load(store) if p.get('pick_id') != pick_id])


def profit(odds: float, units: float, result: str) -> float:
    if result == 'P' or result is None:
        return 0.0
    if result == 'L':
        return -units
    o = float(odds or -110)
    return units * (100 / abs(o) if o < 0 else o / 100)


def grade(store, games_by_league: dict, closing_by_league: dict) -> int:
    """Fill result + closing line for finished games.  Returns how many picks changed."""
    picks = load(store)
    changed = 0
    for p in picks:
        if p.get('status') == 'graded':
            continue
        g = games_by_league.get(p['league'])
        if g is None:
            continue
        row = g[g.id == str(p['game_id'])]
        if row.empty:
            continue
        r = row.iloc[0]
        # closing line: last saved odds snapshot before kickoff, else the schedule's line
        close_home = closing_by_league.get(p['league'], {}).get(str(p['game_id']))
        if close_home is None or pd.isna(close_home):
            close_home = r.spread_home if pd.notna(r.spread_home) and r.completed else None
        if close_home is not None and pd.notna(close_home):
            close_side = close_home if p['side'] == 'home' else -close_home
            p['closing_line'] = float(close_side)
            p['clv'] = round(float(p['line']) - float(close_side), 2)
        if bool(r.completed):
            margin_side = (r.home_score - r.away_score) if p['side'] == 'home' else (r.away_score - r.home_score)
            cover = margin_side + float(p['line'])
            p['result'] = 'W' if cover > 0 else 'L' if cover < 0 else 'P'
            p['profit'] = round(profit(p.get('odds', -110), float(p.get('units', 1)), p['result']), 3)
            p['final'] = f"{r.away} {int(r.away_score)} – {r.home} {int(r.home_score)}"
            p['status'] = 'graded'
            changed += 1
    if changed or picks:
        save_all(store, picks)
    return changed


def find_closes(store, picks: list[dict]) -> dict:
    """{league: {game_id: closing home spread}} from the last saved pre-kickoff snapshot."""
    from odds import load_snapshots, line_history, match_team
    out = {}
    for lg in {p['league'] for p in picks if p.get('status') != 'graded'}:
        kicks = [pd.Timestamp(p['kickoff']) for p in picks if p['league'] == lg and p.get('status') != 'graded']
        kicks = [k.tz_localize('UTC') if k.tzinfo is None else k for k in kicks]
        span = (max(kicks) - min(kicks)).days + 21
        h = line_history(load_snapshots(store, lg, since_days=span, now=max(kicks) + pd.Timedelta(days=1)))
        if h.empty:
            continue
        h['commence'] = pd.to_datetime(h['commence'], utc=True, format='ISO8601')
        for p in picks:
            if p['league'] != lg or p.get('status') == 'graded':
                continue
            ko = pd.Timestamp(p['kickoff'])
            ko = ko.tz_localize('UTC') if ko.tzinfo is None else ko
            cand = h[(h.commence - ko).abs() < pd.Timedelta(hours=36)]
            for _, e in cand.groupby('event_id'):
                a = match_team(e.away_team.iloc[0], [p['away'], p['home']])
                hm = match_team(e.home_team.iloc[0], [p['away'], p['home']])
                if a and hm and {a, hm} == {p['away'], p['home']}:
                    v = float(e.spread_home.iloc[-1])
                    out.setdefault(lg, {})[str(p['game_id'])] = v if hm == p['home'] else -v
                    break
    return out


def summary(picks: list[dict], by: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame([p for p in picks if p.get('status') == 'graded'])
    if df.empty:
        return pd.DataFrame()
    def agg(g):
        w, l, pu = (g.result == 'W').sum(), (g.result == 'L').sum(), (g.result == 'P').sum()
        risk = g['units'].astype(float).sum()
        clv = g['clv'].dropna() if 'clv' in g else pd.Series(dtype=float)
        return pd.Series({'Record': f'{w}-{l}' + (f'-{pu}' if pu else ''),
                          'Win %': round(100 * w / max(1, w + l), 1),
                          'Units': round(g['profit'].sum(), 2),
                          'ROI %': round(100 * g['profit'].sum() / max(1e-9, risk), 1),
                          'Avg CLV (pts)': round(clv.mean(), 2) if len(clv) else np.nan,
                          'Beat close %': round(100 * (clv > 0).mean(), 1) if len(clv) else np.nan})
    if by and by in df:
        return df.groupby(by).apply(agg).reset_index()
    return agg(df).to_frame().T
