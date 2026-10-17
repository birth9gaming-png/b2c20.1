"""Injury reports + each team's key players, so the model knows who is missing.

Sources (both allow automated reads in robots.txt):
  NFL - ESPN's public injuries feed (JSON)
  CFB - Covers.com college football injury report (one page, updated daily)
Key players come from this season's stats:
  NFL - play-by-play, last 4 games: main QB (dropbacks), top RB (carries), top 2 targets
  CFB - CFBD season player stats: main QB (attempts), top RB (carries), top 2 receivers (catches)
A key player listed Out / Doubtful / IR / Suspended counts as missing.  A QB
listed Questionable is flagged too, because the market moves hard on QBs.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
import pandas as pd
import requests

ESPN_NFL = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries'
COVERS = {'CFB': 'https://www.covers.com/sport/football/ncaaf/injuries',
          'NFL': 'https://www.covers.com/sport/football/nfl/injuries'}
HEADERS = {'User-Agent': 'Mozilla/5.0 (BornToCover personal injury reader; one request per refresh)'}

SUFFIX = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
STATUS_RE = re.compile(r'^(Out For Season|Out|Questionable|Doubtful|Probable|Day-To-Day|Injured Reserve|IR|Suspended|'
                       r'Suspension|Game Time Decision|Inactive)\b\s*-?\s*(.*?)\s*(?:\(([^)]*)\))?\s*$', re.I)
POS_RE = re.compile(r'^(QB|RB|FB|WR|TE|OL|OT|OG|G|T|C|DL|DE|DT|NT|LB|ILB|OLB|EDGE|CB|S|FS|SS|DB|K|P|LS|ATH|KR|PR)$', re.I)


def name_key(name: str) -> str:
    """'Jalen Hurts' / 'J.Hurts' / 'J. Hurts' -> 'j hurts'.  'Amon-Ra St. Brown' -> 'a brown'."""
    s = re.sub(r"[^a-z.\s'-]", '', str(name).lower()).replace('.', ' ').replace('-', ' ')
    parts = [p for p in s.split() if p and p not in SUFFIX]
    if len(parts) < 2:
        return parts[0] if parts else ''
    return f'{parts[0][0]} {parts[-1]}'


def is_out(status: str) -> bool:
    s = str(status).lower().strip()
    return bool(re.match(r'^(out|doubtful|injured reserve|ir\b|suspen|reserve|pup|inactive)', s))


# ---------------------------------------------------------------- fetch
def _get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def parse_espn(payload: dict) -> pd.DataFrame:
    from history import nfl_code
    rows = []
    for team in payload.get('injuries', []):
        for e in team.get('injuries', []):
            a = e.get('athlete') or {}
            abbr = ((a.get('team') or {}).get('abbreviation')) or ''
            pos = ((a.get('position') or {}).get('abbreviation')) or ''
            rows.append(dict(team=nfl_code(abbr) if abbr else team.get('displayName'), player=a.get('displayName'),
                             pos=pos, status=e.get('status', ''), date=e.get('date'),
                             note=e.get('shortComment') or ''))
    return pd.DataFrame(rows)


# Covers school names that differ from CFBD names (checked against the live page, Oct 2026)
COVERS_CFB = {'FIU Panthers': 'Florida International', 'FIU': 'Florida International', 'JSU': 'Jacksonville State',
              'Louisiana-Monroe': 'UL Monroe', 'Miami (FL)': 'Miami', 'Mississippi': 'Ole Miss',
              'MS State': 'Mississippi State', 'ND State': 'North Dakota State', 'NIU': 'Northern Illinois',
              'San Jose State': 'San José State', 'UMass': 'Massachusetts', 'Hawaii': "Hawai'i",
              'Miami (OH)': 'Miami (OH)', 'Southern Mississippi': 'Southern Miss', 'Connecticut': 'UConn',
              'Appalachian State': 'App State', 'ULM': 'UL Monroe', 'UL Lafayette': 'Louisiana',
              'Sam Houston State': 'Sam Houston', 'Texas-San Antonio': 'UTSA', 'Texas-El Paso': 'UTEP'}


def _clean(fragment: str) -> str:
    import html as html_mod
    return ' '.join(html_mod.unescape(re.sub(r'(?s)<[^>]+>', ' ', fragment)).split())


def covers_team(league: str, school: str, nick: str = '') -> str:
    """Covers school + nickname -> our team name (CFBD school name / NFL code)."""
    school, nick = school.strip(), nick.strip()
    if league == 'NFL':
        from odds import TEAM_NAMES
        from rules import DIV_OF
        by_nick = {k.split()[-1].lower(): v for k, v in TEAM_NAMES.items() if v in DIV_OF}
        if nick.lower() in by_nick:
            return by_nick[nick.lower()]
        full = f'{school} {nick}'.strip()
        return TEAM_NAMES.get(full, school)
    return COVERS_CFB.get(school, COVERS_CFB.get(f'{school} {nick}'.strip(), school))


def _status(text: str):
    """'Questionable - Knee' -> ('Questionable', 'Knee').  Unknown wording keeps the part before the dash."""
    m = STATUS_RE.match(text)
    if m:
        st = m.group(1).title()
        return ('IR' if st == 'Ir' else st), m.group(2)
    head, _, tail = text.partition(' - ')
    return head.strip().title(), tail.strip()


def _parse_covers_html(page: str, league: str) -> list[dict]:
    """Structured read: each team is <a id="School"></a><section> ... <table> rows (Player | POS | Status)."""
    import html as html_mod
    parts = re.split(r'(?is)<a\s+id="([^"]*)"\s*>\s*</a>\s*<section\b[^>]*>', page)
    rows = []
    for k in range(1, len(parts) - 1, 2):
        school, body = html_mod.unescape(parts[k]), parts[k + 1]
        nm = re.search(r'(?is)CoversMatchups-teamName.*?<span[^>]*>(.*?)</span>', body)
        nick = _clean(nm.group(1)) if nm else ''
        team = covers_team(league, school, nick)
        last = None
        for tr in re.findall(r'(?is)<tr\b[^>]*>(.*?)</tr>', body):
            tds = re.findall(r'(?is)<td\b[^>]*>(.*?)</td>', tr)
            if len(tds) >= 3:
                bold = re.search(r'(?is)<b[^>]*>(.*?)</b>', tds[2])
                cell = _clean(tds[2])
                st_text = _clean(bold.group(1)) if bold else re.sub(r'\(.*$', '', cell).strip()
                if not st_text:
                    continue
                status, injury = _status(st_text)
                date = re.search(r'\(\s*([^)]*?)\s*\)', cell)
                last = dict(team=team, school=f'{school} {nick}'.strip(), player=_clean(tds[0]),
                            pos=_clean(tds[1]).upper(), status=status, injury=injury,
                            date=date.group(1) if date else None, note='')
                rows.append(last)
            elif len(tds) == 1 and last is not None and not last['note']:
                last['note'] = _clean(tds[0])[:240]          # the blurb row under each player
    return rows


def _parse_covers_text(page: str, league: str) -> list[dict]:
    """Fallback if the page layout changes: walk the text lines.

    Team block:  ...School / Nickname / (n) / Player / POS / Status
    Player row:  Name / POS / 'Status - Injury' / '(' / 'date)' / note
    """
    from splits import html_to_text
    lines = [' '.join(ln.split()) for ln in html_to_text(page).splitlines()]
    rows, team, school = [], None, ''
    for i, ln in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ''
        h = re.match(r'^(.+?)\s*\((\d+)\)$', ln)
        if (re.match(r'^\(\d+\)$', ln) and nxt == 'Player') or (h and not STATUS_RE.match(ln) and nxt == 'Player'):
            if h:                                         # 'Air Force Falcons (2)' on one line
                name_parts = [h.group(1)]
            else:                                         # 'Air Force' / 'Falcons' / '(2)'
                name_parts = [x for x in lines[max(0, i - 2):i] if re.match(r"^[A-Za-z][A-Za-z .&'()-]*$", x)]
            school = ' '.join(name_parts)
            sch, nick = (name_parts[0], name_parts[1]) if len(name_parts) == 2 else (school, '')
            team = covers_team(league, sch, nick)
            continue
        if team is None or len(ln) > 60 or i < 2 or not STATUS_RE.match(ln):
            continue
        if not POS_RE.match(lines[i - 1]):                # every Covers row has a POS cell; notes never do
            continue
        status, injury = _status(ln)
        tail = ' '.join(lines[i + 1:i + 3])
        date = re.match(r'^\(\s*([^)]*?)\s*\)', tail)
        rows.append(dict(team=team, school=school, player=lines[i - 2], pos=lines[i - 1].upper(), status=status,
                         injury=injury, date=date.group(1) if date else None, note=''))
    return rows


def parse_covers(page: str, league: str = 'CFB') -> pd.DataFrame:
    """Covers injury page -> one row per listed player (team, player, pos, status, injury, date, note)."""
    rows = _parse_covers_html(page, league)
    if not rows:
        rows = _parse_covers_text(page, league)
    return pd.DataFrame(rows, columns=['team', 'school', 'player', 'pos', 'status', 'injury', 'date', 'note'])


def fetch(league: str) -> tuple[pd.DataFrame, str]:
    """Injury rows for a league and the source used."""
    errors = []
    if league == 'NFL':
        try:
            df = parse_espn(_get(ESPN_NFL).json())
            if len(df):
                return df, 'ESPN'
        except Exception as exc:
            errors.append(f'ESPN: {exc}')
    try:
        df = parse_covers(_get(COVERS[league]).text, league)
        if len(df):
            return df, 'Covers'
        errors.append('Covers: page loaded but no injury rows were recognized')
    except Exception as exc:
        errors.append(f'Covers: {type(exc).__name__}: {exc}')
    raise RuntimeError('; '.join(errors) or 'no injury rows found')


def refresh(store, league: str, log=print):
    df, src = fetch(league)
    store.put_json(f'injuries/{league}.json', dict(at=datetime.now(timezone.utc).isoformat(), source=src,
                                                   rows=df.to_dict('records')))
    log(f'Injuries: {len(df)} report lines from {src}')


def load(store, league: str) -> tuple[pd.DataFrame, str | None, str | None]:
    v = store.get_json(f'injuries/{league}.json')
    if not v:
        return pd.DataFrame(columns=['team', 'player', 'pos', 'status']), None, None
    return pd.DataFrame(v['rows']), v.get('at'), v.get('source')


# ---------------------------------------------------------------- key players
def key_players_nfl(pbp: pd.DataFrame, last_n_weeks: int = 4) -> dict:
    """team -> list of {name, role} from recent play-by-play."""
    from history import nfl_code
    df = pbp[pbp['season_type'] == 'REG'].copy()
    if df.empty:
        return {}
    df['week'] = pd.to_numeric(df['week'], errors='coerce')
    recent = df[df['week'] > df['week'].max() - last_n_weeks]
    recent = recent[recent['posteam'].notna()]
    recent['posteam'] = recent['posteam'].map(nfl_code)
    out = {}
    for team, g in recent.groupby('posteam'):
        roles = []
        qb = g[pd.to_numeric(g.get('qb_dropback'), errors='coerce') == 1]['passer_player_name'].dropna().value_counts()
        if len(qb):
            roles.append(dict(name=qb.index[0], role='QB'))
        rb = g[g['play_type'] == 'run']['rusher_player_name'].dropna()
        rb = rb[~rb.isin([r['name'] for r in roles])].value_counts()
        if len(rb):
            roles.append(dict(name=rb.index[0], role='RB1'))
        tg = g['receiver_player_name'].dropna().value_counts()
        for k, n in enumerate(tg.index[:2]):
            roles.append(dict(name=n, role=f'Target{k + 1}'))
        out[team] = roles
    return out


def key_players_cfb(rows: list[dict]) -> dict:
    """CFBD /stats/player/season rows -> team -> key players."""
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    df['stat'] = pd.to_numeric(df['stat'], errors='coerce')
    cat = df['category'].str.lower(); typ = df['statType'].str.upper()
    out = {}

    def top(mask, n, role):
        s = df[mask].sort_values('stat', ascending=False)
        for team, g in s.groupby('team'):
            for k, r in enumerate(g.head(n).itertuples()):
                out.setdefault(team, []).append(dict(name=r.player, role=role if n == 1 else f'{role}{k + 1}'))
    top((cat == 'passing') & (typ == 'ATT'), 1, 'QB')
    top((cat == 'rushing') & (typ == 'CAR'), 1, 'RB1')
    top((cat == 'receiving') & (typ == 'REC'), 2, 'Target')
    # a QB who also leads in carries should not double as RB1
    for team, roles in out.items():
        qb = {name_key(r['name']) for r in roles if r['role'] == 'QB'}
        out[team] = [r for r in roles if not (r['role'] == 'RB1' and name_key(r['name']) in qb)]
    return out


# ---------------------------------------------------------------- per game
def team_report(team: str, rows: pd.DataFrame, keys: list[dict]) -> dict:
    """Missing key players + other notable outs for one team (rows already filtered to the team)."""
    keymap = {name_key(k['name']): k['role'] for k in keys or []}
    missing, questionable, others = [], [], []
    for r in rows.itertuples():
        k = name_key(r.player)
        role = keymap.get(k)
        label = f"{r.player} ({r.pos or '?'}) {r.status}"
        if role and is_out(r.status):
            missing.append(dict(player=r.player, role=role, status=r.status))
        elif role and role == 'QB' and 'question' in str(r.status).lower():
            questionable.append(dict(player=r.player, role=role, status=r.status))
        elif is_out(r.status) and str(r.pos).upper() in ('QB', 'RB', 'WR', 'TE'):
            others.append(label)
    return dict(missing=missing, qb_out=any(m['role'] == 'QB' for m in missing),
                qb_questionable=bool(questionable), questionable=questionable, others=others[:6],
                known=bool(keys))


def resolve_teams(df: pd.DataFrame, league: str, candidates) -> pd.DataFrame:
    """Map report team names onto the schedule's team names, once per board.

    Exact names first.  Otherwise the longest schedule name the report name starts with, compared against
    EVERY team, so 'Florida State Seminoles' can only become Florida State, never Florida."""
    if df.empty or 'team' not in df:
        return df
    from odds import match_team, norm
    cands = sorted({str(c) for c in candidates if pd.notna(c)})
    by_norm = {norm(c): c for c in cands}
    mapping = {}
    for name in df['team'].dropna().unique():
        n = str(name)
        if n in by_norm.values():
            mapping[name] = n
        elif norm(n) in by_norm:
            mapping[name] = by_norm[norm(n)]
        else:
            mapping[name] = match_team(n, cands)
    out = df.copy()
    if league == 'CFB' and 'school' in out:              # use the fuller 'School Nickname' when the short name missed
        miss = out['team'].map(mapping).isna()
        out.loc[miss, 'team'] = out.loc[miss, 'school'].map(lambda x: match_team(str(x), cands))
        out.loc[~miss, 'team'] = out.loc[~miss, 'team'].map(mapping)
    else:
        out['team'] = out['team'].map(mapping)
    return out


def match_team_rows(df: pd.DataFrame, team: str, league: str) -> pd.DataFrame:
    """Rows for one team.  Call resolve_teams() first so names are exact."""
    if df.empty or 'team' not in df:
        return df
    return df[df['team'] == team]


def describe(rep: dict, team: str) -> list[str]:
    out = []
    for m in rep.get('missing', []):
        out.append(f"{team}: {m['role']} {m['player']} {m['status'].upper()}")
    for q in rep.get('questionable', []):
        out.append(f"{team}: QB {q['player']} questionable")
    return out
