"""DraftKings betting splits: parse a copy/paste of the DK Network splits page,
then turn bets% / handle% plus line movement into simple flags.

Typical pasted block (one value per line or tab-separated):
    HOU Texans @ JAX Jaguars
    10/18, 09:30AM
    Spread
    Odds  % Handle  % Bets
    JAX Jaguars -2.5   -120   74%   76%
    HOU Texans +2.5    +100   26%   24%
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
import pandas as pd

import html as html_mod

import requests

import config

DK_URL = 'https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/'
DK_GROUP = {'NFL': 88808, 'CFB': 87637}
VSIN_URL = {'NFL': 'https://data.vsin.com/nfl/betting-splits/',
            'CFB': 'https://data.vsin.com/college-football/betting-splits/'}
HEADERS = {'User-Agent': 'Mozilla/5.0 (BornToCover personal splits reader; one request per refresh)'}

MINUS = '−'
ODDS_RE = re.compile(r'^[+\-]\d{3,4}$|^EVEN$', re.I)
PCT_RE = re.compile(r'^(\d{1,3})%$')
HEADER_RE = re.compile(r'^(.+?)\s+(?:@|at)\s+(.+)$')
MARKETS = {'moneyline': 'moneyline', 'spread': 'spread', 'total': 'total', 'point spread': 'spread', 'totals': 'total'}
SKIP = {'odds', '% handle', '% bets', 'handle', 'bets', '%handle', '%bets'}


ROW_RE = re.compile(r'^(.*?\S)\s+([+\-]\d{3,4}|EVEN)\s+(\d{1,3}%)\s+(\d{1,3}%)$', re.I)


def _tokens(text: str) -> list[str]:
    text = text.replace(MINUS, '-').replace('–', '-').replace('\xa0', ' ')
    out = []
    for line in text.splitlines():
        for part in re.split(r'\t+|\s{3,}', line.strip()):
            part = part.strip()
            if not part:
                continue
            m = ROW_RE.match(part)                      # "Memphis -7 -108 90% 77%" on one line
            if m:
                out.extend(m.groups())
            else:
                out.append(part)
    # a matchup header split over three text nodes: "Memphis", "@", "Tulane"
    merged, i = [], 0
    while i < len(out):
        if (i + 2 < len(out) and out[i + 1].lower() in ('@', 'at')
                and '%' not in out[i] and '%' not in out[i + 2]):
            merged.append(f'{out[i]} @ {out[i + 2]}')
            i += 3
        else:
            merged.append(out[i])
            i += 1
    return merged


def _split_label(label: str):
    """'JAX Jaguars -2.5' -> ('JAX Jaguars', -2.5); 'Over 42.5' -> ('Over', 42.5)."""
    m = re.match(r'^(.*?)\s*([+\-]?\d+(?:\.\d+)?|PK|pk)$', label)
    if m and m.group(1):
        v = m.group(2)
        return m.group(1).strip(), 0.0 if v.lower() == 'pk' else float(v)
    return label.strip(), None


def parse_dk(text: str) -> list[dict]:
    """Return one dict per game: {away, home, start, spread:[...], total:[...], moneyline:[...]}."""
    toks = _tokens(text)
    games, cur, market = [], None, None
    i = 0
    while i < len(toks):
        t = toks[i]
        low = t.lower()
        h = HEADER_RE.match(t)
        if h and not PCT_RE.match(t) and not ODDS_RE.match(t) and '%' not in t:
            cur = dict(away=h.group(1).strip(), home=h.group(2).strip(), start=None,
                       spread=[], total=[], moneyline=[])
            games.append(cur); market = None; i += 1
            continue
        if cur is None:
            i += 1; continue
        if low in MARKETS:
            market = MARKETS[low]; i += 1; continue
        if low in SKIP or re.match(r'^\d{1,2}/\d{1,2}', t):
            if re.match(r'^\d{1,2}/\d{1,2}', t) and cur['start'] is None:
                cur['start'] = t
            i += 1; continue
        # side entry: label, odds, pct, pct  (label may already contain the line)
        if market and i + 3 < len(toks) + 1:
            nxt = toks[i + 1:i + 4]
            if len(nxt) == 3 and ODDS_RE.match(nxt[0]) and PCT_RE.match(nxt[1]) and PCT_RE.match(nxt[2]):
                name, line = _split_label(t)
                odds = 100 if nxt[0].upper() == 'EVEN' else int(nxt[0])
                cur[market].append(dict(name=name, line=line, odds=odds,
                                        handle=float(PCT_RE.match(nxt[1]).group(1)),
                                        bets=float(PCT_RE.match(nxt[2]).group(1))))
                i += 4; continue
        i += 1
    return [g for g in games if g['spread'] or g['total'] or g['moneyline']]


def side_of(name: str, away: str, home: str) -> str | None:
    """Which side a DK label refers to, given the DK header's away/home strings."""
    n = name.lower().strip()
    if n == away.lower().strip():
        return 'away'
    if n == home.lower().strip():
        return 'home'
    a, h = set(away.lower().split()), set(home.lower().split())
    words = set(n.split())
    sa, sh = len(words & a), len(words & h)
    return 'away' if sa > sh else 'home' if sh > sa else None


def spread_splits(game: dict) -> dict | None:
    """{'away': {...}, 'home': {...}} for the spread market of one parsed game."""
    out = {}
    for e in game.get('spread', []):
        s = side_of(e['name'], game['away'], game['home'])
        if s:
            out[s] = e
    return out if len(out) == 2 else None


def flags(spl: dict | None, open_home: float | None, current_home: float | None) -> dict:
    """Sharp/public/RLM flags for the spread.

    sharp_side : side where handle% - bets% >= SHARP_GAP (fewer, bigger bets)
    public_side: side with >= 65% of bets
    rlm_side   : minority-of-bets side whose number got WORSE since open
                 (books moved toward it despite the public on the other side)
    """
    res = dict(sharp_side=None, public_side=None, rlm_side=None, notes=[])
    if not spl:
        return res
    for side in ('away', 'home'):
        e = spl[side]
        if e['handle'] - e['bets'] >= config.SHARP_GAP:
            res['sharp_side'] = side
            res['notes'].append(f"{side}: {e['handle']:.0f}% of money on {e['bets']:.0f}% of bets")
        if e['bets'] >= 65:
            res['public_side'] = side
    if open_home is not None and current_home is not None and pd.notna(open_home) and pd.notna(current_home):
        move = current_home - open_home              # <0: home number got worse, >0: away number got worse
        if move <= -0.5 and spl['home']['bets'] < 45:
            res['rlm_side'] = 'home'
        elif move >= 0.5 and spl['away']['bets'] < 45:
            res['rlm_side'] = 'away'
        if res['rlm_side']:
            e = spl[res['rlm_side']]
            res['notes'].append(f"reverse line move toward {res['rlm_side']} ({e['bets']:.0f}% of bets, line moved {abs(move):.1f})")
    return res


def save(store, league: str, parsed: list[dict], matches: dict):
    now = datetime.now(timezone.utc)
    store.put_json(f"splits/{league}/{now.strftime('%Y-%m-%dT%H-%M-%S')}.json",
                   dict(captured_at=now.isoformat(), games=parsed, matches=matches))


def latest(store, league: str) -> dict:
    """game_id -> (spread splits dict, captured_at) from the most recent paste that matched it."""
    out = {}
    for k in store.list(f'splits/{league}/'):
        snap = store.get_json(k) or {}
        for idx, gid in (snap.get('matches') or {}).items():
            try:
                g = snap['games'][int(idx)]
            except (IndexError, ValueError, KeyError):
                continue
            spl = spread_splits(g)
            if gid and spl:
                if g.get('_reversed'):
                    spl = {'away': spl['home'], 'home': spl['away']}
                out[gid] = (spl, snap.get('captured_at'))
    return out


def _name_score(dk_name: str, team: str, league: str) -> float:
    """How well a DraftKings label matches one schedule team (0-1)."""
    from odds import match_team, norm
    from nfl_data import CODE_TO_NAME
    if league == 'NFL':
        full = CODE_TO_NAME.get(team, team)
        nick = full.split()[-1].lower()
        words = dk_name.lower().split()
        return 1.0 if nick in words or team.lower() in words else 0.0
    if match_team(dk_name, [team]) == team:
        return 1.0
    a, b = norm(dk_name), norm(team)
    if b and b in a:
        return 0.8
    wa, wb = set(dk_name.lower().split()), set(team.lower().split())
    return 0.5 * len(wa & wb) / max(1, len(wb))


def best_match(parsed: dict, slate: list[dict], league: str):
    """(game_id, reversed, score) of the schedule game a parsed DK game refers to."""
    best = (None, False, 0.0)
    for g in slate:
        s1 = _name_score(parsed['away'], g['away'], league) + _name_score(parsed['home'], g['home'], league)
        s2 = _name_score(parsed['away'], g['home'], league) + _name_score(parsed['home'], g['away'], league)
        if s1 > best[2]:
            best = (g['id'], False, s1)
        if s2 > best[2]:
            best = (g['id'], True, s2)
    return best if best[2] >= 1.5 else (None, False, best[2])


# ---------------------------------------------------------------- automatic download
def html_to_text(page: str) -> str:
    """Every HTML tag becomes a line break, so each text node lands on its own line."""
    page = re.sub(r'(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>', '\n', page)
    page = re.sub(r'(?s)<[^>]+>', '\n', page)
    page = html_mod.unescape(page)
    lines = [ln.strip() for ln in page.splitlines()]
    return '\n'.join(ln for ln in lines if ln)


def _http_get(url, params):
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_dk(league: str, max_pages: int = 12, get=None) -> list[dict]:
    """Every game on the DraftKings Network splits page for the next 7 days (10 per page)."""
    get = get or _http_get
    games, seen = [], set()
    for page in range(1, max_pages + 1):
        text = html_to_text(get(DK_URL, dict(tb_eg=DK_GROUP[league], tb_edate='n7days', tb_emt=0, tb_page=page)))
        found = [g for g in parse_dk(text) if (g['away'], g['home'], g['start']) not in seen]
        if not found:
            break
        for g in found:
            seen.add((g['away'], g['home'], g['start']))
        games += found
    return games


NUM = re.compile(r'^[+\-]?\d+(?:\.\d+)?$|^PK$|^EVEN$', re.I)


def _cells(row_html: str) -> list[str]:
    cells = re.findall(r'(?is)<t[dh][^>]*>(.*?)</t[dh]>', row_html)
    return [re.sub(r'\s+', ' ', html_mod.unescape(re.sub(r'(?s)<[^>]+>', ' ', c))).strip().replace(MINUS, '-')
            for c in cells]


def parse_vsin(page: str) -> list[dict]:
    """VSiN's DraftKings splits table: two rows per game (away first, then home).
    Columns: icon | team | spread | handle% | bets% | total | handle% | bets% | ML | handle% | bets%"""
    rows = []
    for tr in re.findall(r'(?is)<tr[^>]*>(.*?)</tr>', page):
        cells = [c for c in _cells(tr) if c]
        if sum(bool(PCT_RE.match(c)) for c in cells) < 2:
            continue
        idx = next((i for i, c in enumerate(cells) if re.search(r'[A-Za-z]{2,}', c)
                    and not PCT_RE.match(c) and not NUM.match(c)), None)
        if idx is None:
            continue
        rest = cells[idx + 1:]
        if len(rest) < 3 or not NUM.match(rest[0]) or not PCT_RE.match(rest[1]) or not PCT_RE.match(rest[2]):
            continue
        name = re.sub(r'^\(?\d+\)?\s+', '', cells[idx]).strip()          # drop rank prefixes like "12 "
        line = 0.0 if rest[0].upper() in ('PK', 'EVEN') else float(rest[0])
        rows.append(dict(name=name, line=line, odds=None,
                         handle=float(PCT_RE.match(rest[1]).group(1)), bets=float(PCT_RE.match(rest[2]).group(1))))
    return [dict(away=a['name'], home=h['name'], start=None, spread=[a, h], total=[], moneyline=[])
            for a, h in zip(rows[0::2], rows[1::2])]


def fetch_vsin(league: str, get=None) -> list[dict]:
    return parse_vsin((get or _http_get)(VSIN_URL[league], dict(source='DK')))


def auto_refresh(store, league: str, slate_rows: list[dict], log=print, sources=None) -> dict:
    """Download splits (DraftKings Network, else VSiN's copy of DraftKings), match to games, save."""
    parsed, source, errors = [], None, []
    for name, fn in sources or (('DraftKings Network', fetch_dk), ('VSiN (DraftKings data)', fetch_vsin)):
        try:
            parsed = fn(league)
        except Exception as exc:                      # network or page-layout problem: try the next source
            errors.append(f'{name}: {exc}')
            parsed = []
        if parsed:
            source = name
            break
    matches = {}
    for k, g in enumerate(parsed):
        gid, rev, score = best_match(g, slate_rows, league)
        if gid:
            g['_reversed'] = bool(rev)
            matches[str(k)] = gid
    if parsed:
        now = datetime.now(timezone.utc)
        store.put_json(f"splits/{league}/{now.strftime('%Y-%m-%dT%H-%M-%S')}.json",
                       dict(captured_at=now.isoformat(), source=source, games=parsed, matches=matches))
    log(f"Splits: matched {len(matches)} of {len(parsed)} games from {source or 'no source'}"
        + (f" — {'; '.join(errors)}" if errors and not parsed else ''))
    return dict(source=source, games=len(parsed), matched=len(matches), errors=errors)


def last_capture(store, league: str):
    keys = store.list(f'splits/{league}/')
    return (store.get_json(keys[-1]) or {}).get('captured_at') if keys else None
