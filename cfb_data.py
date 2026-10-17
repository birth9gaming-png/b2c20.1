"""College football data from CollegeFootballData.com (CFBD API v2).

Free tier = 1,000 calls/month.  Every call is counted in storage so the app can
show usage.  Completed weeks are cached, so a normal refresh costs ~4 calls.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

BASE = 'https://api.collegefootballdata.com'


def _f(row, *names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default


def norm(name) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


class CFBD:
    def __init__(self, key: str, store=None):
        self.key = key
        self.store = store

    def _count(self):
        if self.store is None:
            return
        month = datetime.now(timezone.utc).strftime('%Y-%m')
        k = f'usage/cfbd_{month}.json'
        u = self.store.get_json(k, {'calls': 0})
        u['calls'] += 1
        self.store.put_json(k, u)

    def get(self, path, **params):
        if not self.key:
            raise RuntimeError('Add CFBD_API_KEY in Streamlit Secrets to load college data.')
        r = requests.get(BASE + path, params=params, timeout=60,
                         headers={'Authorization': f'Bearer {self.key}', 'Accept': 'application/json'})
        self._count()
        if r.status_code == 401:
            raise RuntimeError('CFBD rejected the API key (401). Check CFBD_API_KEY in Secrets.')
        if r.status_code == 429:
            raise RuntimeError('CFBD monthly or rate limit reached (429). Try again later.')
        r.raise_for_status()
        return r.json()

    def usage_this_month(self) -> int:
        if self.store is None:
            return 0
        month = datetime.now(timezone.utc).strftime('%Y-%m')
        return self.store.get_json(f'usage/cfbd_{month}.json', {'calls': 0})['calls']


# ---------------------------------------------------------------- normalizers
def games_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append(dict(
            id=str(_f(r, 'id')), season=int(_f(r, 'season')), week=int(_f(r, 'week')),
            season_type=str(_f(r, 'seasonType', 'season_type', default='regular')),
            kickoff=_f(r, 'startDate', 'start_date'),
            away=_f(r, 'awayTeam', 'away_team'), home=_f(r, 'homeTeam', 'home_team'),
            away_classification=str(_f(r, 'awayClassification', 'away_classification', 'awayDivision', default='fbs') or 'fbs').lower(),
            home_classification=str(_f(r, 'homeClassification', 'home_classification', 'homeDivision', default='fbs') or 'fbs').lower(),
            neutral=bool(_f(r, 'neutralSite', 'neutral_site', default=False)),
            completed=bool(_f(r, 'completed', default=False)),
            away_score=_f(r, 'awayPoints', 'away_points'), home_score=_f(r, 'homePoints', 'home_points'),
            venue_id=_f(r, 'venueId', 'venue_id'), venue=_f(r, 'venue'),
            conference_game=bool(_f(r, 'conferenceGame', 'conference_game', default=False)),
            away_conf=_f(r, 'awayConference', 'away_conference'), home_conf=_f(r, 'homeConference', 'home_conference'),
        ))
    g = pd.DataFrame(out)
    if len(g):
        g['away_score'] = pd.to_numeric(g['away_score'], errors='coerce')
        g['home_score'] = pd.to_numeric(g['home_score'], errors='coerce')
    return g


def verified_home_spread(q: dict, away: str, home: str):
    """CFBD 'spread' is the home handicap, but verify with formattedSpread when present."""
    spread = _f(q, 'spread')
    try:
        spread = float(spread) if spread is not None else None
    except (TypeError, ValueError):
        spread = None
    fmt = str(_f(q, 'formattedSpread', 'formatted_spread', default='') or '')
    m = re.fullmatch(r'\s*(.*?)\s*([+-]?\d+(?:\.\d+)?)\s*', fmt)
    if m and spread is not None:
        name, val = norm(m.group(1)), float(m.group(2))
        if name and abs(abs(val) - abs(spread)) < 1e-6:
            if name == norm(home) or norm(home).startswith(name) or name.startswith(norm(home)):
                return val
            if name == norm(away) or norm(away).startswith(name) or name.startswith(norm(away)):
                return -val
    return spread


def lines_frame(rows: list[dict]) -> pd.DataFrame:
    """Per game: median close/current home spread, median opening home spread, totals."""
    out = []
    for r in rows:
        gid, away, home = str(_f(r, 'id')), _f(r, 'awayTeam', 'away_team'), _f(r, 'homeTeam', 'home_team')
        spreads, opens, totals, topen = [], [], [], []
        for q in _f(r, 'lines', default=[]) or []:
            s = verified_home_spread(q, away, home)
            if s is not None:
                spreads.append(s)
            so = _f(q, 'spreadOpen', 'spread_open')
            try:
                so = float(so) if so is not None else None
            except (TypeError, ValueError):
                so = None
            if so is not None and s is not None:
                # the opener uses the same sign convention as the provider's raw 'spread'
                raw = _f(q, 'spread')
                try:
                    flip = float(raw) != s
                except (TypeError, ValueError):
                    flip = False
                so = -so if flip else so
                if abs(so - s) <= 10:
                    opens.append(so)
            for key, dest in (('overUnder', totals), ('overUnderOpen', topen)):
                v = _f(q, key, key.replace('overUnder', 'over_under'))
                try:
                    if v is not None and float(v) > 0:
                        dest.append(float(v))
                except (TypeError, ValueError):
                    pass
        out.append(dict(id=gid, spread_home=np.median(spreads) if spreads else np.nan,
                        open_home=np.median(opens) if opens else np.nan,
                        total=np.median(totals) if totals else np.nan,
                        total_open=np.median(topen) if topen else np.nan, n_books=len(spreads)))
    return pd.DataFrame(out)


def advanced_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        off, de = _f(r, 'offense', default={}) or {}, _f(r, 'defense', default={}) or {}
        out.append({'id': str(_f(r, 'gameId', 'game_id', 'id')), 'season': _f(r, 'season'), 'week': _f(r, 'week'),
                    'team': _f(r, 'team'),
                    'offense.successRate': _f(off, 'successRate', 'success_rate'),
                    'offense.explosiveness': _f(off, 'explosiveness'), 'offense.plays': _f(off, 'plays'),
                    'defense.successRate': _f(de, 'successRate', 'success_rate'),
                    'defense.explosiveness': _f(de, 'explosiveness'), 'defense.plays': _f(de, 'plays')})
    return pd.DataFrame(out)


def box_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        gid = str(_f(r, 'id'))
        for t in _f(r, 'teams', default=[]) or []:
            stats = {norm(s.get('category')): s.get('stat') for s in t.get('stats', []) if isinstance(s, dict)}

            def num(k):
                try:
                    return float(stats.get(k))
                except (TypeError, ValueError):
                    return None
            att = num('passingattempts')
            if att is None and re.fullmatch(r'\d+\s*[-/]\s*\d+', str(stats.get('completionattempts', ''))):
                att = float(re.split(r'[-/]', str(stats['completionattempts']))[-1])
            rush = num('rushingattempts')
            yards = num('totalyards')
            if yards is None and num('netpassingyards') is not None and num('rushingyards') is not None:
                yards = num('netpassingyards') + num('rushingyards')
            out.append(dict(id=gid, team=_f(t, 'team', 'school'), yards=yards,
                            plays=(att + rush) if att is not None and rush is not None else None,
                            giveaways=num('turnovers')))
    return pd.DataFrame(out)


def venues_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for v in rows:
        loc = _f(v, 'location', default={}) or {}
        lat = _f(v, 'latitude', default=_f(loc, 'x')) if isinstance(loc, dict) else _f(v, 'latitude')
        lon = _f(v, 'longitude', default=_f(loc, 'y')) if isinstance(loc, dict) else _f(v, 'longitude')
        out.append(dict(venue_id=_f(v, 'id'), name=_f(v, 'name'), city=_f(v, 'city'), state=_f(v, 'state'),
                        lat=lat, lon=lon, dome=bool(_f(v, 'dome', default=False))))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- season refresh
def refresh_season(api: CFBD, store, season: int, weeks_done: set[int] | None = None, log=print):
    """Download the current season: schedule, lines (with openers), advanced + box stats.

    Box scores are fetched one completed week at a time and cached, so each
    refresh only pays for weeks not yet stored.
    """
    from storage import read_frame, write_frame
    prefix = f'cache/cfb/{season}'
    games = games_frame(api.get('/games', year=season, seasonType='regular'))
    write_frame(store, f'{prefix}/games.csv.gz', games)
    lines = lines_frame(api.get('/lines', year=season, seasonType='regular'))
    write_frame(store, f'{prefix}/lines.csv.gz', lines)
    adv = advanced_frame(api.get('/stats/game/advanced', year=season, seasonType='regular'))
    write_frame(store, f'{prefix}/advanced.csv.gz', adv)
    box = read_frame(store, f'{prefix}/box.csv.gz')
    have = set() if box is None else set(box['id'].astype(str))
    done = games[games.completed]
    need_weeks = sorted(set(done[~done.id.isin(have)].week))
    parts = [box] if box is not None else []
    for w in need_weeks:
        parts.append(box_frame(api.get('/games/teams', year=season, week=int(w), seasonType='regular')))
        log(f'CFB box scores week {w} downloaded')
    if parts:
        box = pd.concat(parts, ignore_index=True).drop_duplicates(['id', 'team'], keep='last')
        write_frame(store, f'{prefix}/box.csv.gz', box)
    try:
        import injuries
        store.put_json(f'{prefix}/key_players.json',
                       injuries.key_players_cfb(api.get('/stats/player/season', year=season, seasonType='regular')))
    except Exception as exc:
        log(f'CFB key players not loaded ({exc})')
    try:                                                   # AP poll: needed for your trap / lookahead / letdown rules
        import rules
        store.put_json(f'{prefix}/polls.json', rules.polls_from_cfbd(api.get('/rankings', year=season, seasonType='regular')))
    except Exception as exc:
        log(f'CFB rankings not loaded ({exc})')
    if not store.exists('cache/cfb/venues.csv.gz'):
        write_frame(store, 'cache/cfb/venues.csv.gz', venues_frame(api.get('/venues')))
    store.put_json(f'{prefix}/refreshed.json', {'at': datetime.now(timezone.utc).isoformat()})
    return games, lines, adv, box
