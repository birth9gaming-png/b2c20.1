"""The Odds API: live spreads/totals, saved snapshots, line history and team matching.

Free plan = 500 credits/month.  One league call with spreads+totals and the
bookmaker list in config costs 2 credits.  Each saved snapshot is the history
used for opening lines, line movement and closing-line value.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

import config
from nfl_data import TEAM_NAMES

API = 'https://api.the-odds-api.com/v4/sports/{sport}/odds'
ET = ZoneInfo('America/New_York')


def norm(name) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


# Odds API school names that don't start with the CFBD name
CFB_ALIASES = {
    'appalachianstatemountaineers': 'App State', 'appalachianstate': 'App State',
    'connecticuthuskies': 'UConn', 'uconnhuskies': 'UConn',
    'massachusettsminutemen': 'UMass', 'umassminutemen': 'UMass',
    'hawaiirainbowwarriors': "Hawai'i", 'hawaiiwarriors': "Hawai'i",
    'sanjosestatespartans': 'San José State', 'sanjosestate': 'San José State',
    'louisianamonroewarhawks': 'UL Monroe', 'ulmonroewarhawks': 'UL Monroe',
    'louisianaragincajuns': 'Louisiana', 'louisianalafayetteragincajuns': 'Louisiana',
    'texasamaggies': 'Texas A&M', 'texasamcommerce': 'East Texas A&M',
    'floridaatlanticowls': 'Florida Atlantic', 'floridainternationalpanthers': 'Florida International',
    'fiupanthers': 'Florida International', 'utsaroadrunners': 'UT San Antonio',
    'texassanantonioroadrunners': 'UT San Antonio', 'utepminers': 'UTEP',
    'ucfknights': 'UCF', 'centralfloridaknights': 'UCF', 'smumustangs': 'SMU',
    'southernmethodistmustangs': 'SMU', 'tcuhornedfrogs': 'TCU', 'byucougars': 'BYU',
    'lsutigers': 'LSU', 'usctrojans': 'USC', 'uclabruins': 'UCLA', 'unlvrebels': 'UNLV',
    'uabblazers': 'UAB', 'olemissrebels': 'Ole Miss', 'mississippirebels': 'Ole Miss',
    'miamiohredhawks': 'Miami (OH)', 'miamiohioredhawks': 'Miami (OH)', 'miamihurricanes': 'Miami',
    'southernmississippigoldeneagles': 'Southern Miss', 'southernmissgoldeneagles': 'Southern Miss',
    'nccharlotte49ers': 'Charlotte', 'charlotte49ers': 'Charlotte',
    'sambhoustonstatebearkats': 'Sam Houston', 'samhoustonbearkats': 'Sam Houston',
    'gaststatepanthers': 'Georgia State', 'kennesawstateowls': 'Kennesaw State',
    'jacksonvillestategamecocks': 'Jacksonville State', 'pittsburghpanthers': 'Pittsburgh',
    'pittpanthers': 'Pittsburgh', 'northcarolinastatewolfpack': 'NC State', 'ncstatewolfpack': 'NC State',
    'armyblackknights': 'Army', 'armywestpointblackknights': 'Army', 'navymidshipmen': 'Navy',
    'southfloridabulls': 'South Florida', 'usfbulls': 'South Florida',
    'middletennesseeblueraiders': 'Middle Tennessee', 'mtsublueraiders': 'Middle Tennessee',
    'westernkentuckyhilltoppers': 'Western Kentucky', 'wkuhilltoppers': 'Western Kentucky',
    'ulmwarhawks': 'UL Monroe', 'bostoncollegeeagles': 'Boston College',
}


def _old_aliases() -> dict:
    """Name aliases carried over from v17 (normalized name -> CFBD name)."""
    import json
    from history import HIST
    p = HIST / 'odds_aliases.json'
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return {k: v.split(':', 1)[1] for k, v in raw.items() if isinstance(v, str) and v.startswith('CFB:')}


OLD_ALIASES = _old_aliases()


def match_team(odds_name: str, candidates: list[str], aliases: dict | None = None) -> str | None:
    """Best CFBD/NFL name for an Odds API team name among `candidates`."""
    n = norm(odds_name)
    if odds_name in TEAM_NAMES:
        code = TEAM_NAMES[odds_name]
        return code if code in candidates else None
    for table in (aliases or {}, CFB_ALIASES, OLD_ALIASES):
        hit = table.get(n)
        if hit and hit in candidates:
            return hit
    best, best_len = None, 0
    for c in candidates:
        cn = norm(c)
        if cn and n.startswith(cn) and len(cn) > best_len:
            best, best_len = c, len(cn)
    return best


class OddsAPI:
    def __init__(self, key: str, store=None):
        self.key = key
        self.store = store
        self.last_headers = {}

    def fetch(self, league: str) -> list[dict]:
        if not self.key:
            raise RuntimeError('Add ODDS_API_KEY in Streamlit Secrets to load sportsbook lines.')
        r = requests.get(API.format(sport=config.ODDS_SPORT[league]), timeout=60, params=dict(
            apiKey=self.key, markets=config.ODDS_MARKETS, oddsFormat='american',
            bookmakers=','.join(config.BOOKMAKERS)))
        self.last_headers = {k: r.headers.get(k) for k in ('x-requests-remaining', 'x-requests-used', 'x-requests-last')}
        if r.status_code == 401:
            raise RuntimeError('The Odds API rejected the key (401). Check ODDS_API_KEY in Secrets.')
        if r.status_code == 429:
            raise RuntimeError('The Odds API rate/quota limit reached (429).')
        r.raise_for_status()
        events = r.json()
        if self.store is not None:
            now = datetime.now(timezone.utc)
            self.store.put_json(f"odds/{league}/{now.strftime('%Y-%m-%dT%H-%M-%S')}.json.gz",
                                dict(fetched_at=now.isoformat(), league=league, events=events))
            self.store.put_json('usage/odds_api.json', dict(at=now.isoformat(), **self.last_headers))
        return events


# ---------------------------------------------------------------- snapshot -> rows
def quotes_from_events(events: list[dict], fetched_at: str) -> pd.DataFrame:
    """Flatten one snapshot into rows: event, book, market, side, point, price."""
    rows = []
    for e in events:
        for b in e.get('bookmakers', []):
            for m in b.get('markets', []):
                for o in m.get('outcomes', []):
                    side = ('home' if o.get('name') == e.get('home_team') else
                            'away' if o.get('name') == e.get('away_team') else
                            str(o.get('name', '')).lower())
                    rows.append(dict(event_id=e['id'], commence=e.get('commence_time'),
                                     away_team=e.get('away_team'), home_team=e.get('home_team'),
                                     book=b.get('key'), market=m.get('key'), side=side,
                                     point=o.get('point'), price=o.get('price'),
                                     book_update=b.get('last_update'), fetched_at=fetched_at))
    return pd.DataFrame(rows)


def load_snapshots(store, league: str, since_days: int = 14, now=None) -> pd.DataFrame:
    keys = store.list(f'odds/{league}/')
    now = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc))
    cutoff = (now - timedelta(days=since_days)).strftime('%Y-%m-%dT%H-%M-%S')
    upto = (now + timedelta(minutes=1)).strftime('%Y-%m-%dT%H-%M-%S')
    parts = []
    for k in keys:
        stamp = k.rsplit('/', 1)[-1].split('.json')[0]
        if stamp < cutoff or stamp > upto:
            continue
        snap = store.get_json(k)
        if snap and snap.get('events'):
            parts.append(quotes_from_events(snap['events'], snap['fetched_at']))
    if not parts:
        return pd.DataFrame()
    q = pd.concat(parts, ignore_index=True)
    q['fetched_at'] = pd.to_datetime(q['fetched_at'], utc=True, format='ISO8601')
    q['commence'] = pd.to_datetime(q['commence'], utc=True, format='ISO8601')
    return q


def line_history(q: pd.DataFrame) -> pd.DataFrame:
    """Per event and snapshot: consensus home spread, sharp-book home spread, total, best prices."""
    if q.empty:
        return pd.DataFrame()
    sp = q[(q.market == 'spreads') & (q.side == 'home') & q.point.notna()]
    sp = sp[sp.fetched_at < sp.commence]                               # pre-kickoff quotes only
    rows = []
    for (eid, t), g in sp.groupby(['event_id', 'fetched_at']):
        sharp = g[g.book.isin(config.SHARP_BOOKS)]
        rows.append(dict(event_id=eid, fetched_at=t, commence=g.commence.iloc[0],
                         away_team=g.away_team.iloc[0], home_team=g.home_team.iloc[0],
                         spread_home=float(np.median(g.point)), sharp_home=float(np.median(sharp.point)) if len(sharp) else np.nan,
                         n_books=int(g.book.nunique())))
    h = pd.DataFrame(rows)
    tot = q[(q.market == 'totals') & (q.side == 'over') & q.point.notna() & (q.fetched_at < q.commence)]
    if len(tot):
        tt = tot.groupby(['event_id', 'fetched_at']).point.median().rename('total').reset_index()
        h = h.merge(tt, on=['event_id', 'fetched_at'], how='left')
    else:
        h['total'] = np.nan
    return h.sort_values(['event_id', 'fetched_at']).reset_index(drop=True)


def current_board(q: pd.DataFrame) -> pd.DataFrame:
    """Latest snapshot per event, with opening (first seen) line and best available prices."""
    h = line_history(q)
    if h.empty:
        return h
    first = h.groupby('event_id').first()
    last = h.groupby('event_id').last()
    out = last.copy()
    out['open_home_seen'] = first['spread_home']
    out['open_seen_at'] = first.index.map(lambda e: h[h.event_id == e].fetched_at.min())
    out['total_open_seen'] = first['total']
    out['snapshots'] = h.groupby('event_id').size()
    # best number for each side in the latest snapshot of each event
    latest_t = q.groupby('event_id').fetched_at.max().rename('latest')
    lq = q.merge(latest_t, left_on='event_id', right_index=True)
    lq = lq[(lq.fetched_at == lq.latest) & (lq.market == 'spreads')]
    for side in ('away', 'home'):
        s = lq[lq.side == side].copy()
        if s.empty:
            out[f'best_{side}_point'] = np.nan; out[f'best_{side}_price'] = np.nan; out[f'best_{side}_book'] = None
            continue
        s = s.sort_values(['point', 'price'], ascending=[False, False]).groupby('event_id').first()
        out[f'best_{side}_point'] = s['point']; out[f'best_{side}_price'] = s['price']; out[f'best_{side}_book'] = s['book']
    return out.reset_index()


def closing_line(h: pd.DataFrame, event_id: str):
    """Last pre-kickoff consensus home spread for an event."""
    e = h[h.event_id == event_id]
    return float(e.spread_home.iloc[-1]) if len(e) else np.nan


def attach_events(slate: pd.DataFrame, board: pd.DataFrame, league: str) -> pd.DataFrame:
    """Match Odds API events to schedule games by date (ET, +/-1 day) and team names."""
    if board.empty or slate.empty:
        out = slate.copy()
        out['event_id'] = None
        return out
    slate = slate.copy()
    slate['et_date'] = slate.kickoff.dt.tz_convert(ET).dt.date
    eids = []
    for g in slate.itertuples():
        best = None
        for b in board.itertuples():
            d = pd.Timestamp(b.commence).tz_convert(ET).date()
            if abs((d - g.et_date).days) > 1:
                continue
            a = match_team(b.away_team, [g.away, g.home]); h = match_team(b.home_team, [g.away, g.home])
            if a == g.away and h == g.home:
                best = b.event_id; break
            if a == g.home and h == g.away:               # listed reversed (neutral sites)
                best = b.event_id; break
        eids.append(best)
    slate['event_id'] = eids
    return slate.drop(columns=['et_date'])
