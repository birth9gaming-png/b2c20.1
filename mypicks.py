"""Your betting profile: import Pikkit history, grade every spread pick, and score this week's games
by how they match the spots where YOU win.

What it learns from your graded picks (each pick counted once, parlay legs included):
  - your hit rate by league and by line size (favorite 0.5-3.5, 4-10, 10.5-21, 21+; underdog 0.5-3.5, 4-10, 10.5+)
  - teams you know (picked 2+ times before) - you hit 77% on those vs 60% on others
  - rivalry games, home vs road
Every rate is shrunk toward your overall record, so a bucket with 10 picks can't swing the score much.
"""
from __future__ import annotations
import io
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HISTORY_FILE = HERE / 'my_picks.csv'
UNIT = 25.0

BUCKETS = [(-999, -21, 'Favorite 21+'), (-21, -10.5, 'Favorite 10.5-21'), (-10.5, -4, 'Favorite 4-10'),
           (-4, 0, 'Favorite 0.5-3.5'), (0, 4, 'Underdog 0.5-3.5'), (4, 10.5, 'Underdog 4-10'), (10.5, 999, 'Underdog 10.5+')]


def bucket(line: float) -> str:
    if line is None or pd.isna(line):
        return 'Pick-em'
    if line == 0:
        return 'Pick-em'
    for lo, hi, name in BUCKETS:
        if (lo < line <= hi) if line < 0 else (lo <= line < hi):
            return name
    return 'Pick-em'


# ---------------------------------------------------------------- Pikkit import
def _strip(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c))


CFB_FIX = {'miami fl': 'Miami', 'miami florida': 'Miami', 'miami (fl)': 'Miami', 'miami oh': 'Miami (OH)',
           'miami ohio': 'Miami (OH)', 'miami (oh)': 'Miami (OH)', 'san jose state': 'San José State', 'hawaii': "Hawai'i",
           'appalachian state': 'App State', 'north carolina state': 'NC State', 'southern mississippi': 'Southern Miss',
           'louisiana monroe': 'UL Monroe', 'ul lafayette': 'Louisiana', 'louisiana lafayette': 'Louisiana',
           'connecticut': 'UConn', 'massachusetts': 'UMass', 'utsa': 'UT San Antonio', 'texas san antonio': 'UT San Antonio',
           'fiu': 'Florida International', 'fau': 'Florida Atlantic', 'usf': 'South Florida', 'pitt': 'Pittsburgh',
           'sam houston state': 'Sam Houston'}
SPREAD_RE = re.compile(r'^(?P<team>.+?) (?P<line>[+-]?\d+(?:\.\d+)?) (?P<kind>Spread(?: Alternate)?)(?P<extra>.*?) (?P<away>[^@]+?) @ (?P<home>.+)$')
ML_RE = re.compile(r'^(?:Money Line - )?(?P<team>.+?) (?:Moneyline|Money Line|Winner)(?P<extra>.*?) (?P<away>[^@]+?) @ (?P<home>.+)$')


def _team(name, cands, league):
    from nfl_data import CODE_TO_NAME
    from odds import match_team
    n = _strip(name).strip()
    if league == 'NFL':
        nick = {v.split()[-1].lower(): k for k, v in CODE_TO_NAME.items()}
        for t in n.lower().split()[::-1]:
            if t in nick and nick[t] in cands:
                return nick[t]
        return None
    fix = CFB_FIX.get(n.lower())
    if fix and fix in cands:
        return fix
    for c in cands:
        if _strip(c).lower() == n.lower():
            return c
    return match_team(n, cands)


def parse_leg(leg: str):
    if re.search(r'1st Half|2nd Half|Quarter|\b1H\b|Live', leg, re.I):
        return None
    m = SPREAD_RE.match(leg)
    if m and m['extra'].strip() in ('', 'Betting'):
        return dict(kind='alt spread' if 'Alternate' in m['kind'] else 'spread', team=m['team'],
                    line=float(m['line']), away=m['away'], home=m['home'])
    m = ML_RE.match(leg)
    if m and m['extra'].strip() == '':
        return dict(kind='moneyline', team=m['team'], line=np.nan, away=m['away'], home=m['home'])
    return None


def import_pikkit(csv_bytes: bytes, games_by_league: dict) -> tuple[pd.DataFrame, dict]:
    """Pikkit transactions.csv -> graded football picks (one row per unique pick) + money summary."""
    d = pd.read_csv(io.BytesIO(csv_bytes), dtype={'bet_id': str})
    d = d[d['status'].astype(str).str.startswith('SETTLED') & (d['status'] != 'SETTLED_VOID')].copy()
    d['placed'] = pd.to_datetime(d['time_placed_iso'], utc=True, format='ISO8601')
    lg = d['leagues'].fillna('')
    summary = dict(bets=len(d), staked=float(d['amount'].sum()), profit=float(d['profit'].sum()),
                   by_type={t: dict(bets=int(len(g)), staked=float(g['amount'].sum()), profit=float(g['profit'].sum()))
                            for t, g in d.groupby('type')},
                   by_league={k: dict(bets=int(len(g)), staked=float(g['amount'].sum()), profit=float(g['profit'].sum()))
                              for k, g in d.groupby(lg)})
    rows = []
    for b in d.itertuples():
        leagues = [x for x in ('NFL', 'CFB') if (x if x == 'NFL' else 'NCAAFB') in str(b.leagues)]
        if not leagues:
            continue
        for leg in str(b.bet_info).split(' | '):
            p = parse_leg(leg)
            if not p:
                continue
            for league in leagues:
                g = games_by_league.get(league)
                if g is None:
                    continue
                win = g[(g.kickoff >= b.placed - pd.Timedelta(hours=6)) & (g.kickoff <= b.placed + pd.Timedelta(days=9))]
                hit = None
                for r in win.itertuples():
                    if _team(p['away'], [r.away, r.home], league) == r.away and _team(p['home'], [r.away, r.home], league) == r.home:
                        hit = r
                        break
                if hit is None:
                    continue
                t = _team(p['team'], [hit.away, hit.home], league)
                if t is None:
                    continue
                side = 'away' if t == hit.away else 'home'
                rows.append(dict(placed=b.placed, league=league, season=int(hit.season), week=int(hit.week), game_id=str(hit.id),
                                 kickoff=hit.kickoff, away=hit.away, home=hit.home, team=t, side=side, kind=p['kind'],
                                 line=p['line'], neutral=bool(hit.neutral), bet_type=b.type,
                                 completed=bool(hit.completed), away_score=hit.away_score, home_score=hit.home_score))
                break
    picks = pd.DataFrame(rows)
    if picks.empty:
        return picks, summary
    picks = picks.drop_duplicates(['game_id', 'team', 'kind', 'line'])
    return grade(picks), summary


def grade(picks: pd.DataFrame) -> pd.DataFrame:
    p = picks.copy()
    margin = np.where(p.side == 'away', p.away_score - p.home_score, p.home_score - p.away_score)
    raw = np.where(p.kind == 'moneyline', margin, margin + p.line.fillna(0))
    p['result'] = np.where(~p.completed.astype(bool) | pd.isna(raw), None, np.where(raw > 0, 'W', np.where(raw < 0, 'L', 'P')))
    return p


def load_history(store=None) -> pd.DataFrame:
    """Imported history: stored copy (latest upload) if present, else the bundled file."""
    if store is not None:
        v = store.get_json('mypicks/history.json')
        if v and v.get('rows'):
            df = pd.DataFrame(v['rows'])
            df['placed'] = pd.to_datetime(df['placed'], utc=True, format='ISO8601')
            return df
    if HISTORY_FILE.exists():
        df = pd.read_csv(HISTORY_FILE)
        df['placed'] = pd.to_datetime(df['placed'], utc=True, format='ISO8601')
        return df
    return pd.DataFrame()


def bundled_summary():
    import json
    p = HERE / 'my_picks_summary.json'
    return json.loads(p.read_text()) if p.exists() else None


def save_history(store, picks: pd.DataFrame, summary: dict):
    store.put_json('mypicks/history.json', dict(rows=picks.astype(object).where(picks.notna(), None).to_dict('records'),
                                                summary=summary))


# ---------------------------------------------------------------- profile
def _rate(w, l, prior, k):
    return (w + k * prior) / (w + l + k)


def build_profile(picks: pd.DataFrame, k_league=20, k_bucket=15) -> dict:
    """Shrunk hit rates from your graded spread picks (standard + alt)."""
    s = picks[picks.kind.isin(['spread', 'alt spread']) & picks.result.isin(['W', 'L'])].copy()
    s = s.sort_values('placed')
    w_all, l_all = int((s.result == 'W').sum()), int((s.result == 'L').sum())
    overall = _rate(w_all, l_all, 0.5, 10)
    prof = dict(overall=dict(w=w_all, l=l_all, rate=overall), league={}, bucket={}, followed={}, rivalry={}, venue={})
    for lg, g in s.groupby('league'):
        w, l = int((g.result == 'W').sum()), int((g.result == 'L').sum())
        prof['league'][lg] = dict(w=w, l=l, rate=_rate(w, l, overall, k_league))
    s['bucket'] = s.line.map(bucket)
    for (lg, b), g in s.groupby(['league', 'bucket']):
        w, l = int((g.result == 'W').sum()), int((g.result == 'L').sum())
        base = prof['league'].get(lg, {}).get('rate', overall)
        prof['bucket'][f'{lg}|{b}'] = dict(w=w, l=l, rate=_rate(w, l, base, k_bucket))
    for b, g in s.groupby('bucket'):            # both leagues pooled, used when a league bucket is thin
        w, l = int((g.result == 'W').sum()), int((g.result == 'L').sum())
        prof['bucket'][f'ALL|{b}'] = dict(w=w, l=l, rate=_rate(w, l, overall, k_bucket))
    # teams you know: picked 2+ times BEFORE in the SAME season (last year's roster is a different team)
    seen, flag = {}, []
    for r in s.itertuples():
        k = (r.team, r.season)
        flag.append(seen.get(k, 0) >= 2)
        seen[k] = seen.get(k, 0) + 1
    s['followed'] = flag
    cur = picks[picks.season == picks.season.max()] if 'season' in picks and len(picks) else picks
    counts = cur.team.value_counts()
    prof['teams'] = counts[counts >= 3].to_dict()          # teams you've bet 3+ times THIS season
    for key, mask in (('followed', s.followed), ('other', ~s.followed)):
        w, l = int((s.result[mask] == 'W').sum()), int((s.result[mask] == 'L').sum())
        prof['followed'][key] = dict(w=w, l=l, rate=_rate(w, l, overall, 20))
    for key, mask in (('road', s.side == 'away'), ('home', s.side == 'home')):
        w, l = int((s.result[mask] == 'W').sum()), int((s.result[mask] == 'L').sum())
        prof['venue'][key] = dict(w=w, l=l, rate=_rate(w, l, overall, 30))
    prof['n'] = len(s)
    return prof


def spot(profile: dict, league: str, team: str, line: float, side: str, rival: bool = False) -> tuple[float, list[str]]:
    """Your expected hit rate betting `team` at `line`, from your own history, plus the reasons."""
    if not profile or not profile.get('n'):
        return 0.5, ['No pick history loaded']
    reasons = []
    b = bucket(line)
    lgrate = profile['league'].get(league, profile['overall'])['rate']
    bk = profile['bucket'].get(f'{league}|{b}') or profile['bucket'].get(f'ALL|{b}')
    rate = bk['rate'] if bk else lgrate
    if bk:
        kind, rng = (b.split(' ', 1) + [''])[:2]
        where = (' in the NFL' if league == 'NFL' else ' in college') if f'{league}|{b}' in profile['bucket'] else ''
        reasons.append(f"you're {bk['w']}-{bk['l']} on {kind.lower()}s of {rng}{where}")
    known = team in profile.get('teams', {})
    f = profile['followed']['followed' if known else 'other']
    rate += 0.5 * (f['rate'] - profile['overall']['rate'])
    if known:
        reasons.append(f"a team you know ({profile['teams'][team]} picks this season; you're {f['w']}-{f['l']} on teams you know)")
    v = profile['venue']['road' if side == 'away' else 'home']
    rate += 0.5 * (v['rate'] - profile['overall']['rate'])
    if rival:
        reasons.append('rivalry game')
    return float(np.clip(rate, 0.40, 0.80)), reasons


def ledger_as_history(picks: list[dict]) -> pd.DataFrame:
    """Picks you log in the app, in the same shape as the imported history (graded ones only)."""
    rows = []
    for p in picks:
        if p.get('status') != 'graded' or p.get('result') not in ('W', 'L', 'P'):
            continue
        rows.append(dict(placed=pd.Timestamp(p['logged_at']), league=p['league'], season=p.get('season'), week=p.get('week'),
                         game_id=str(p['game_id']), away=p['away'], home=p['home'], team=p['team'], side=p['side'],
                         kind='spread', line=float(p['line']), neutral=False, bet_type='straight (app)', result=p['result']))
    return pd.DataFrame(rows)


def _rank_spots_by_record(board: pd.DataFrame, profile: dict, league: str) -> pd.DataFrame:
    """v19.3 ranking (your record only).  Kept for comparison; the app uses rank_spots below."""
    import rules as rules_mod
    if board.empty:
        return board
    conf = rules_mod.load_conferences() if league == 'CFB' else None
    rows = []
    for r in board.to_dict('records'):
        line_h = r.get('current_home')
        if line_h is None or pd.isna(line_h):
            continue
        riv = rules_mod.rivals(league, r['away'], r['home'])
        opts = {}
        for side in ('away', 'home'):
            team = r[side]
            line = -line_h if side == 'away' else line_h
            rate, why = spot(profile, league, team, line, side, riv)
            opts[side] = (rate, why, line, team)
        a, h = opts['away'][0], opts['home'][0]
        model_side = r.get('model_side')
        if abs(a - h) < 0.03 and model_side in ('away', 'home'):
            pick = model_side                       # close call: lean with the model
        else:
            pick = 'away' if a > h else 'home'
        rate, why, line, team = opts[pick]
        other = 'home' if pick == 'away' else 'away'
        edge = r.get('edge')
        model_txt = '—'
        if edge is not None and pd.notna(edge):
            toward = edge if pick == 'away' else -edge
            model_txt = (f'model agrees (+{toward:.1f} pts)' if toward >= 0.5 else
                         f'model disagrees ({toward:.1f} pts)' if toward <= -0.5 else 'model neutral')
        inj = (r.get('injuries') or {}).get(pick) or {}
        known = team in profile.get('teams', {})
        rows.append(dict(id=r['id'], kickoff=r['kickoff'], season=r.get('season'), week=r.get('week'), away=r['away'], home=r['home'], side=pick, team=team, line=line,
                         spot_rate=rate, reasons=why, known_team=known, rival=riv, model=model_txt,
                         model_toward=(edge if pick == 'away' else -edge) if edge is not None and pd.notna(edge) else np.nan,
                         other_team=r[other], other_rate=opts[other][0], key_injury=bool(inj.get('missing') or inj.get('qb_questionable')),
                         power=(league == 'NFL') or rules_mod.is_power(conf, r['away'], int(r['season']))
                         or rules_mod.is_power(conf, r['home'], int(r['season']))))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(['spot_rate', 'known_team', 'model_toward'], ascending=[False, False, False]).reset_index(drop=True)


def _user_flags(games, league, board, extra_polls=None):
    """Your own trap / lookahead / letdown definitions (rules.py) for the games on the board."""
    import rules as rules_mod
    polls = rules_mod.load_polls(extra_polls) if league == 'CFB' else {}
    if games is None or board.empty:
        return {}
    seasons = set(int(x) for x in board['season'].dropna().unique())
    g = games[games['season'].isin(seasons)].copy()
    if g.empty:
        return {}
    if 'spread_home' not in g:
        g['spread_home'] = np.nan
    cur = {r['id']: r.get('current_home') for r in board.to_dict('records')}
    try:
        f = rules_mod.flag_games(g, league, polls=polls, current_home=cur)
    except Exception as exc:
        print(f'rule flags failed: {type(exc).__name__}: {exc}')
        return {}
    return f.set_index('id').to_dict('index')


SIGNAL_TEXT = {'Line moving its way': 'line moved toward them',
               'Bigger-money side': 'bigger DraftKings money on them',
               'Reverse line move': 'reverse line move toward them'}


def side_reasons(side: str, r: dict, uf: dict | None = None, league: str | None = None) -> tuple[list[str], list[str]]:
    """Real reasons FOR betting `side`, and the same reasons for the other side.  Each carries its record
    against the closing line since 2016, so proven spots (⭐) stand apart from context.

    Proven spots (Vegas-anchored model), line moved toward the side, opponent in a lookahead / letdown spot,
    your trap rule, bigger DraftKings money or a reverse line move on the side."""
    import proven as pv
    from board import signals
    uf = uf or r.get('rule_flags') or {}
    league = league or r.get('league') or 'CFB'
    other = 'home' if side == 'away' else 'away'

    def reasons(sd, od):
        out = []
        ps = (r.get('proven') or {}).get(sd) or []
        for p in ps:
            out.append(f"⭐ {p['label']}: {100 * p['rate']:.1f}% ({p['record']}) since 2016")
        proven_look = any(p['key'] == 'dog_vs_look' for p in ps)
        for h in signals(sd, r):
            if h == 'Line moving its way':
                out.append('line moved toward them' + pv.record_text(league, 'move'))
            elif h == 'Bigger-money side':
                out.append('bigger DraftKings money on them (no splits history to test yet)')
            elif h == 'Reverse line move':
                out.append('reverse line move toward them (no splits history to test yet)')
            elif 'lookahead' in h:
                if not proven_look:
                    out.append('opponent in a lookahead spot' + pv.record_text(league, 'lookahead'))
            elif 'letdown' in h:
                out.append('opponent in a letdown spot' + pv.record_text(league, 'letdown'))
            else:
                out.append(h.replace('Opponent in', 'opponent in a'))
        if uf.get(f'{od}_lookahead') and not proven_look and not any('lookahead' in x for x in out):
            out.append(f"opponent looking ahead to {uf.get(f'{od}_next')}" + pv.record_text(league, 'lookahead'))
        if uf.get(f'{od}_letdown') and not any('letdown' in x for x in out):
            out.append(f"opponent in a letdown spot after beating {uf.get(f'{od}_prev')}" + pv.record_text(league, 'letdown'))
        if uf.get('trap') and uf.get('fav') == sd:
            out.append('your trap rule: the "worse" team is favored, back the favorite' + pv.record_text(league, 'trap'))
        return out
    return reasons(side, other), reasons(other, side)


def moved_against(side: str, r: dict) -> float:
    """Points the market moved AWAY from `side` since the open (positive = moving against you)."""
    mv = r.get('move_home')                      # current - open home spread; > 0 = home got more points
    if mv is None or pd.isna(mv):
        return 0.0
    return float(mv) if side == 'home' else -float(mv)


def rank_spots(board: pd.DataFrame, profile: dict, league: str, games: pd.DataFrame | None = None,
               extra_polls: list | None = None) -> pd.DataFrame:
    """Every game: the side that fits your spots AND has a real reason behind it.

    A side qualifies when it has at least one reason (line moved its way, opponent in a lookahead/letdown
    spot, your trap rule, bigger DraftKings money / reverse line move) and the market has not moved 3+
    points against it.  Order: qualifying first, then net reasons (for minus against), then your record.
    The model is shown as a note only."""
    import rules as rules_mod
    if board.empty:
        return board
    conf = rules_mod.load_conferences() if league == 'CFB' else None
    flags = _user_flags(games, league, board, extra_polls)
    rows = []
    for r in board.to_dict('records'):
        line_h = r.get('current_home')
        if line_h is None or pd.isna(line_h):
            continue
        riv = rules_mod.rivals(league, r['away'], r['home'])
        uf = flags.get(r['id'], {})
        opts = {}
        for side in ('away', 'home'):
            team = r[side]
            line = -line_h if side == 'away' else line_h
            rate, why = spot(profile, league, team, line, side, riv)
            pro, con = side_reasons(side, r, uf or r.get('rule_flags'), league)
            against = moved_against(side, r)
            ok = bool(pro) and against < 3 and not any(x.startswith('⭐') for x in con)   # no proven spot against
            n_pv = sum(x.startswith('⭐') for x in pro)
            opts[side] = dict(rate=rate, why=why, line=line, team=team, pro=pro, con=con, against=against, ok=ok,
                              proven=n_pv, key=(ok, n_pv, len(pro) - len(con), rate))
        pick = 'away' if opts['away']['key'] > opts['home']['key'] else 'home'
        o = opts[pick]
        other = 'home' if pick == 'away' else 'away'
        edge = r.get('edge')
        toward = (edge if pick == 'away' else -edge) if edge is not None and pd.notna(edge) else np.nan
        model_txt = '—' if pd.isna(toward) else (f'model agrees (+{toward:.1f} pts)' if toward >= 0.5 else
                                                 f'model disagrees ({toward:.1f} pts)' if toward <= -0.5 else 'model neutral')
        inj = (r.get('injuries') or {}).get(pick) or {}
        rows.append(dict(id=r['id'], kickoff=r['kickoff'], season=r.get('season'), week=r.get('week'), away=r['away'],
                         home=r['home'], side=pick, team=o['team'], line=o['line'], spot_rate=o['rate'], reasons=o['why'],
                         for_reasons=o['pro'], against_reasons=o['con'], net=len(o['pro']) - len(o['con']),
                         qualifies=o['ok'], moved_against=o['against'], proven_n=o['proven'],
                         known_team=o['team'] in profile.get('teams', {}), rival=riv, model=model_txt, model_toward=toward,
                         other_team=r[other], other_rate=opts[other]['rate'],
                         key_injury=bool(inj.get('missing') or inj.get('qb_questionable')),
                         power=(league == 'NFL') or rules_mod.is_power(conf, r['away'], int(r['season']))
                         or rules_mod.is_power(conf, r['home'], int(r['season']))))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(['qualifies', 'proven_n', 'net', 'spot_rate'], ascending=[False, False, False, False]).reset_index(drop=True)
