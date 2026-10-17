"""Your style: a model trained ONLY on your own picks.

It answers two questions for every game on the board:
  1. Which side would you bet?   A logistic model learns your habits from the weeks you bet: it compares
     the sides you picked with every side you passed on (line, favorite/dog, home/road, ranked teams,
     rivalry, lookahead / letdown / trap spots, line move, records, kickoff time, teams you've bet THIS season).
  2. How would you play it?      Spread, alt spread or moneyline, the way you usually do for that kind of line
     (college moneylines only - your NFL moneylines are a coin flip, so NFL uses spread / alt only).
No stats model and no market data decide anything here; it is a copy of you.  The model never sees whether
a bet won, so testing it on old seasons is a fair test of whether your habits win.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FORMS = {'CFB': ('spread', 'alt spread', 'moneyline'), 'NFL': ('spread', 'alt spread')}
BUCKETS = [(-99, -14, 'favorite 14+'), (-14, -7, 'favorite 7-14'), (-7, -3, 'favorite 3-7'), (-3, 0, 'favorite 0-3'),
           (0, 3, 'underdog 0-3'), (3, 7, 'underdog 3-7'), (7, 99, 'underdog 7+')]
COMMON = ['fav', 'absl', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'home', 'road', 'rival', 'own_look', 'own_letdown',
          'opp_look', 'opp_letdown', 'trap_fav', 'trap_dog', 'wpct_diff', 'recent_team']
FEATS = {'CFB': COMMON + ['own_rank', 'opp_rank', 'own_power', 'opp_power', 'move_toward', 'night', 'noon'],
         'NFL': COMMON + ['primetime', 'late']}
MIN_PICKS = 15


def bucket_of(line: float) -> int:
    for k, (lo, hi, _) in enumerate(BUCKETS):
        if lo <= line < hi or (k == len(BUCKETS) - 1 and line >= lo):
            return k
    return 3


def bucket_name(line: float) -> str:
    return BUCKETS[bucket_of(line)][2]


def _side_line(spread_home, side):
    return -spread_home if side == 'away' else spread_home


# ---------------------------------------------------------------- features
def side_features(games: pd.DataFrame, league: str, picks: pd.DataFrame | None = None,
                  current_home: dict | None = None, opens: dict | None = None, only_ids=None,
                  polls: dict | None = None) -> pd.DataFrame:
    """One row per (game, side).  `games` = full schedule of the seasons involved (for lookahead/letdown).

    current_home : {id: home spread now} for upcoming games (else the schedule's spread_home)
    opens        : {id: opening home spread} (college line move)
    only_ids     : limit the output rows to these game ids
    """
    import rules
    g = games.copy()
    g['id'] = g['id'].astype(str)
    if 'spread_home' not in g:
        g['spread_home'] = np.nan
    cur = {str(k): v for k, v in (current_home or {}).items()}
    flags = rules.flag_games(g, league, polls=polls, current_home=cur).set_index('id')
    conf = rules.load_conferences() if league == 'CFB' else None
    opens = {str(k): v for k, v in (opens or {}).items()}
    # your picks this season by team, with time placed - "teams you've been watching"
    seen = {}
    if picks is not None and len(picks):
        pk = picks[picks['league'] == league]
        for r in pk.itertuples():
            seen.setdefault((int(r.season), r.team), []).append(pd.Timestamp(r.placed))
    want = set(str(i) for i in only_ids) if only_ids is not None else None
    rows = []
    for r in g.itertuples():
        if want is not None and r.id not in want:
            continue
        sh = cur.get(r.id, r.spread_home)
        if sh is None or pd.isna(sh):
            continue
        fl = flags.loc[r.id]
        ko = r.kickoff if pd.notna(r.kickoff) else pd.Timestamp('2000-01-01', tz='UTC')
        et = ko.tz_convert('America/New_York')
        op = opens.get(r.id, np.nan)
        mv = (sh - op) if op is not None and pd.notna(op) else np.nan
        done = bool(r.completed) and pd.notna(r.away_score) and pd.notna(r.home_score)
        for side, opp in (('away', 'home'), ('home', 'away')):
            line = _side_line(sh, side)
            team, oteam = getattr(r, side), getattr(r, opp)
            b = bucket_of(line)
            wg, og = fl[f'{side}_games'], fl[f'{opp}_games']
            margin = ((r.away_score - r.home_score) * (1 if side == 'away' else -1)) if done else np.nan
            prior = [t for t in seen.get((int(r.season), team), []) if t < ko - pd.Timedelta(hours=12)]
            row = dict(id=r.id, season=int(r.season), week=int(r.week), kickoff=ko, side=side, team=team, opp=oteam,
                       line=line, bucket=b, margin=margin,
                       fav=float(line < 0), absl=min(abs(line), 35.0), home=float(side == 'home' and not r.neutral),
                       road=float(side == 'away' and not r.neutral), rival=float(fl['rival']),
                       own_look=float(fl[f'{side}_lookahead']), own_letdown=float(fl[f'{side}_letdown']),
                       opp_look=float(fl[f'{opp}_lookahead']), opp_letdown=float(fl[f'{opp}_letdown']),
                       trap_fav=float(bool(fl['trap']) and fl['fav'] == side),
                       trap_dog=float(bool(fl['trap']) and fl['fav'] == opp),
                       wpct_diff=(fl[f'{side}_wins'] / wg if wg else .5) - (fl[f'{opp}_wins'] / og if og else .5),
                       recent_team=float(min(len(prior), 3)),
                       own_rank=float(pd.notna(fl.get(f'{side}_rank'))), opp_rank=float(pd.notna(fl.get(f'{opp}_rank'))),
                       own_power=float(league == 'NFL' or rules.is_power(conf, team, int(r.season))),
                       opp_power=float(league == 'NFL' or rules.is_power(conf, oteam, int(r.season))),
                       move_toward=float(np.clip(-mv if side == 'home' else mv, -4, 4)) if pd.notna(mv) else 0.0,
                       night=float(et.hour >= 19), noon=float(et.hour < 13),
                       primetime=float(et.hour >= 19 or et.dayofweek in (0, 3)),   # Sun/Mon/Thu night games
                       late=float(16 <= et.hour < 19))
            for k in range(len(BUCKETS)):
                row[f'b{k}'] = float(b == k)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- training
def usable_picks(picks: pd.DataFrame, league: str) -> pd.DataFrame:
    if picks is None or picks.empty:
        return pd.DataFrame()
    p = picks[(picks['league'] == league) & picks['kind'].isin(FORMS[league])].copy()
    p['game_id'] = p['game_id'].astype(str)
    return p


def train(picks: pd.DataFrame, games: pd.DataFrame, league: str, opens: dict | None = None,
          feedback: list | None = None) -> dict | None:
    """Fit the copy of you for one league.  Returns None if there are too few picks."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    p = usable_picks(picks, league)
    if len(p) < MIN_PICKS:
        return None
    seasons = sorted(p['season'].dropna().astype(int).unique())
    g = games[games['season'].isin(seasons)]
    rows = side_features(g, league, picks=picks, opens=opens)
    if rows.empty:
        return None
    picked = set(zip(p['game_id'], p['side']))
    rows['picked'] = [int((i, s) in picked) for i, s in zip(rows['id'], rows['side'])]
    weeks = set(zip(p['season'].astype(int), p['week'].astype(int)))
    T = rows[[(a, b) in weeks for a, b in zip(rows['season'], rows['week'])]].copy()
    if league == 'CFB':
        T = T[(T['own_power'] + T['opp_power']) > 0]          # you skip Group of 5-only games
    if T['picked'].sum() < MIN_PICKS:
        return None
    feats = FEATS[league]
    X = T[feats].to_numpy(float)
    y = T['picked'].to_numpy(int)
    w = np.ones(len(T))
    # your thumbs up / down on suggested picks: strong, direct lessons (weight 3)
    fb = [f for f in (feedback or []) if f.get('league') == league and f.get('feats')]
    if fb:
        X = np.vstack([X, np.array([[float(f['feats'].get(c, 0.0)) for c in feats] for f in fb])])
        y = np.r_[y, [1 if f['vote'] > 0 else 0 for f in fb]]
        w = np.r_[w, np.full(len(fb), 3.0)]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    m = LogisticRegression(C=0.3, max_iter=3000).fit((X - mu) / sd, y, sample_weight=w)
    raw = m.decision_function((X - mu) / sd)
    # leave-one-season-out: can it recognise your picks in a season it did not see?
    aucs = []
    for s in sorted(T['season'].unique()):
        tr, te = T[T['season'] != s], T[T['season'] == s]
        if te['picked'].sum() < 5 or tr['picked'].sum() < 10:
            continue
        Xt = tr[feats].to_numpy(float)
        mt = LogisticRegression(C=0.3, max_iter=3000).fit((Xt - mu) / sd, tr['picked'])
        aucs.append(float(roc_auc_score(te['picked'], mt.decision_function((te[feats].to_numpy(float) - mu) / sd))))
    return dict(league=league, feats=feats, mu=mu.tolist(), sd=sd.tolist(), coef=m.coef_[0].tolist(),
                intercept=float(m.intercept_[0]), train_scores=np.sort(raw).tolist(),
                n_picks=int(T['picked'].sum()), n_sides=int(len(T)), seasons=seasons, n_feedback=len(fb),
                recognise=float(np.mean(aucs)) if aucs else None, forms=form_table(p, games, league))


def score(model: dict, rows: pd.DataFrame) -> np.ndarray:
    X = rows[model['feats']].to_numpy(float)
    return ((X - np.array(model['mu'])) / np.array(model['sd'])) @ np.array(model['coef']) + model['intercept']


def style_pct(model: dict, s: np.ndarray) -> np.ndarray:
    """0-100: how much a side looks like your picks, vs every side you could have picked before."""
    ref = np.array(model['train_scores'])
    return np.round(100 * np.searchsorted(ref, s, side='right') / len(ref)).astype(int)


def explain(model: dict, row: dict, top: int = 3) -> list[str]:
    """The habits that push this side up the most."""
    text = {'fav': 'a favorite', 'home': 'at home', 'road': 'on the road', 'rival': 'rivalry game',
            'own_look': None, 'own_letdown': None, 'opp_look': 'opponent looking ahead',
            'opp_letdown': 'opponent in a letdown spot', 'trap_fav': 'your trap rule (worse team favored)',
            'trap_dog': None, 'recent_team': "a team you've bet this season", 'own_rank': 'ranked team',
            'opp_rank': 'ranked opponent', 'own_power': 'power-conference team', 'opp_power': 'power-conference opponent',
            'move_toward': 'line moved toward them', 'night': 'night game', 'noon': 'noon kickoff',
            'primetime': 'primetime game', 'late': 'late-afternoon game',
            'wpct_diff': None, 'absl': None}
    for k, (_, _, nm) in enumerate(BUCKETS):
        text[f'b{k}'] = f'{nm} line'
    contrib = []
    for f, c, mu, sd in zip(model['feats'], model['coef'], model['mu'], model['sd']):
        v = float(row.get(f, 0.0))
        z = (v - mu) / sd * c
        if text.get(f) and v != 0 and z > 0.05:
            contrib.append((z, text[f]))
    return [t for _, t in sorted(contrib, reverse=True)[:top]]


# ---------------------------------------------------------------- how you play it
def form_table(p: pd.DataFrame, games: pd.DataFrame, league: str) -> dict:
    """Per line bucket: which leg type you use, how many points you buy on alts, and your record."""
    g = games.copy(); g['id'] = g['id'].astype(str)
    sp = g.set_index('id')['spread_home'].to_dict() if 'spread_home' in g else {}
    out = {}
    rows = []
    for r in p.itertuples():
        gl = sp.get(str(r.game_id))
        if gl is None or pd.isna(gl):
            if r.kind == 'spread' and pd.notna(r.line):
                gl_side = float(r.line)
            else:
                continue
        else:
            gl_side = _side_line(gl, r.side)
        rows.append(dict(b=bucket_of(gl_side), kind=r.kind, buy=(r.line - gl_side) if r.kind == 'alt spread' and pd.notna(r.line) else np.nan,
                         result=r.result))
    d = pd.DataFrame(rows)
    allowed = FORMS[league]
    med_all = float(d['buy'].median()) if len(d) and d['buy'].notna().any() else 2.0
    for k in range(len(BUCKETS)):
        x = d[d['b'] == k] if len(d) else d
        counts = {f: int((x['kind'] == f).sum()) if len(x) else 0 for f in allowed}
        rec = {}
        for f in allowed:
            y = x[x['kind'] == f] if len(x) else x
            rec[f] = [int((y['result'] == 'W').sum()) if len(y) else 0, int((y['result'] == 'L').sum()) if len(y) else 0]
        buys = x['buy'].dropna() if len(x) else pd.Series(dtype=float)
        buy = float(buys.median()) if len(buys) >= 4 else med_all
        out[str(k)] = dict(counts=counts, record=rec, buy=float(np.clip(buy, 0.5, 10.0)))
    return out


# your rules for HOW to play a side (from you, Oct 16 2026) - not learned, written down
ML_CAP = -250                       # never a leg priced worse than -250
KEYS = {'CFB': (7, 10, 14, 21),     # 7.5 -> 6.5, 10.5 -> 9.5, 14.5 -> 13.5, 21.5 -> 20.5
        'NFL': (3, 7)}              # your NFL habit: -3 to -4.5 -> -2.5, -7 / -7.5 -> -6.5
WIN_FIT = {'CFB': (0.0, -0.1157), 'NFL': (-0.0096, -0.1466)}   # P(win | spread), fitted on 2016-2025 games


def ml_price(line: float, league: str, vig: float = 0.025) -> int:
    """Estimated moneyline for a side laying/getting `line` (matches market prices within ~10-15 cents)."""
    a, b = WIN_FIT[league]
    p = 1 / (1 + np.exp(-(a + b * line))) + vig
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def key_buy(line: float, league: str) -> float | None:
    """The alt line you'd buy to get under/over a key number, or None if the line isn't sitting on one."""
    for k in KEYS[league]:
        if line < 0:
            lo = -(k + 1.5) if (league == 'NFL' and k == 3) else -(k + 0.5)   # NFL: -3 .. -4.5 -> -2.5
            if lo <= line <= -k:
                return -(k - 0.5)
        else:
            if k - 0.5 <= line <= k:                                       # +6.5 / +7 -> +7.5
                return k + 0.5
    return None


def how_you_play(model: dict | None, line: float, league: str, ml_odds: float | None = None) -> dict:
    """Suggested leg for a side at `line`, using YOUR rules:
      favorite, college, moneyline no worse than -250 -> moneyline
      line sitting on a key number                   -> buy through it (alt spread)
      otherwise                                      -> the spread
    NFL never uses moneylines (yours are a coin flip)."""
    price = ml_odds if ml_odds is not None and pd.notna(ml_odds) else ml_price(line, league)
    alt = key_buy(line, league)
    if league == 'CFB' and line < 0 and price >= ML_CAP:
        kind, leg_line = 'moneyline', None
    elif alt is not None:
        kind, leg_line = 'alt spread', alt
    else:
        kind, leg_line = 'spread', line
    rec = (0, 0)
    if model:
        ft = model['forms'].get(str(bucket_of(line)), {})
        rec = tuple(ft.get('record', {}).get(kind, [0, 0]))
    return dict(kind=kind, leg_line=leg_line, record=rec, bucket=BUCKETS[bucket_of(line)][2],
                price=price if kind == 'moneyline' else None)


def leg_text(team: str, leg: dict) -> str:
    if leg['kind'] == 'moneyline':
        return f"{team} ML (~{leg['price']:+d})" if leg.get('price') is not None else f'{team} ML'
    v = leg['leg_line']
    s = 'PK' if v == 0 else f'{v:+g}'
    return f'{team} {s}' + (' (alt)' if leg['kind'] == 'alt spread' else '')


def grade_leg(leg: dict, margin: float) -> str | None:
    """W/L/P of a leg given the side's final margin."""
    if margin is None or pd.isna(margin):
        return None
    v = margin if leg['kind'] == 'moneyline' else margin + leg['leg_line']
    return 'W' if v > 0 else 'L' if v < 0 else 'P'


# ---------------------------------------------------------------- this week
def suggest(model: dict, board: pd.DataFrame, games: pd.DataFrame, league: str, picks: pd.DataFrame | None = None,
            k: int = 3, feedback: list | None = None) -> pd.DataFrame:
    """Every board game's side you'd most likely bet, how you'd play it, ranked by style fit."""
    if model is None or board.empty:
        return pd.DataFrame()
    cur = board.set_index('id')['current_home'].to_dict()
    opens = board.set_index('id')['open_home'].to_dict() if 'open_home' in board else {}
    seasons = set(int(s) for s in board['season'].dropna().unique())
    g = games[games['season'].isin(seasons)]
    rows = side_features(g, league, picks=picks, current_home=cur, opens=opens, only_ids=board['id'])
    if rows.empty:
        return rows
    rows['raw'] = score(model, rows)
    rows['style'] = style_pct(model, rows['raw'].to_numpy())
    votes = {(str(f['id']), f['side']): f['vote'] for f in (feedback or []) if f.get('league') == league}
    # a side you turned down drops out; the other side of that game can still show up
    rows['_down'] = [votes.get((i, sd), 0) < 0 for i, sd in zip(rows['id'], rows['side'])]
    best = rows.sort_values(['_down', 'raw'], ascending=[True, False]).drop_duplicates('id')
    b = board.set_index('id')
    out = []
    for r in best.to_dict('records'):
        info = b.loc[r['id']]
        leg = how_you_play(model, r['line'], league)
        inj = (info.get('injuries') or {}).get(r['side']) or {}
        power = bool(r['own_power'] or r['opp_power'])
        row_d = {**info.to_dict(), 'id': r['id']}
        try:
            from mypicks import side_reasons, moved_against
            pro, con = side_reasons(r['side'], row_d, row_d.get('rule_flags') or {}, league)
            against = moved_against(r['side'], row_d)
        except Exception as exc:
            print(f'reasons failed: {type(exc).__name__}: {exc}')
            pro, con, against = [], [], 0.0
        out.append(dict(id=r['id'], kickoff=info['kickoff'], away=info['away'], home=info['home'], side=r['side'],
                        team=r['team'], opp=r['opp'], line=r['line'], style=int(r['style']), raw=r['raw'],
                        leg=leg_text(r['team'], leg), leg_kind=leg['kind'], leg_line=leg['leg_line'],
                        leg_record=leg['record'], bucket=leg['bucket'], why=explain(model, r),
                        key_injury=bool(inj.get('missing') or inj.get('qb_questionable')),
                        power=power, edge=info.get('edge'), season=int(r['season']), week=int(r['week']),
                        feats={f: float(r[f]) for f in model['feats']},
                        for_reasons=pro, against_reasons=con, proven_n=sum(x.startswith('⭐') for x in pro),
                        qualifies=bool(pro) and against < 3 and not any(x.startswith('⭐') for x in con),
                        moved_against=against,
                        vote=votes.get((str(r['id']), r['side']), 0)))
    o = pd.DataFrame(out).sort_values('raw', ascending=False).reset_index(drop=True)
    o['shortlist'] = False
    # your picks: fits your style AND has at least one real reason (proven spots first), no 3-pt move against
    elig = o[(~o['key_injury']) & (o['power'] if league == 'CFB' else True) & (o['vote'] >= 0) & o['qualifies']]
    ok = elig.sort_values(['proven_n', 'raw'], ascending=[False, False]).head(k).index
    o.loc[ok, 'shortlist'] = True
    return o


# ---------------------------------------------------------------- does your style win? (old seasons)
def replay(model: dict, games: pd.DataFrame, league: str, seasons, k: int = 3) -> dict:
    """Let the copy of you pick k legs a week on past seasons, play them your way, grade them.

    Also grades the same sides at the plain closing spread."""
    g = games[games['season'].isin(list(seasons)) & games['completed'] & games['spread_home'].notna()]
    if g.empty:
        return {}
    rows = side_features(g, league)
    rows = rows[rows['margin'].notna()]
    if league == 'CFB':
        rows = rows[(rows['own_power'] + rows['opp_power']) > 0]
    rows['raw'] = score(model, rows)
    top = rows.sort_values('raw', ascending=False).drop_duplicates('id').groupby(['season', 'week']).head(k)
    legs, ats = [], []
    by_kind = {}
    for r in top.to_dict('records'):
        leg = how_you_play(model, r['line'], league)
        res = grade_leg(leg, r['margin'])
        legs.append(res)
        by_kind.setdefault(leg['kind'], []).append(res)
        ats.append(grade_leg(dict(kind='spread', leg_line=r['line']), r['margin']))

    def rec(xs):
        return [sum(x == 'W' for x in xs), sum(x == 'L' for x in xs)]
    by_season = {}
    for s, d in top.groupby('season'):
        by_season[int(s)] = rec([grade_leg(dict(kind='spread', leg_line=x), m) for x, m in zip(d['line'], d['margin'])])
    return dict(legs=rec(legs), ats=rec(ats), by_kind={kk: rec(v) for kk, v in by_kind.items()}, by_season_ats=by_season,
                n_weeks=int(top.groupby(['season', 'week']).ngroups))


# ---------------------------------------------------------------- helpers for the app
HABITS = {'fav': 'Favorites', 'absl': 'Bigger lines (either way)', 'home': 'Home teams', 'road': 'Road teams',
          'rival': 'Rivalry / division games', 'own_look': 'Team itself looking ahead', 'own_letdown': 'Team itself in a letdown spot',
          'opp_look': 'Opponent looking ahead', 'opp_letdown': 'Opponent in a letdown spot',
          'trap_fav': 'Your trap rule: back the "worse" favorite', 'trap_dog': 'Underdog with the better record',
          'wpct_diff': 'Team with the better record', 'recent_team': "Teams you've bet this season",
          'own_rank': 'Ranked teams', 'opp_rank': 'Ranked opponents', 'own_power': 'Power-conference teams',
          'opp_power': 'Power-conference opponents', 'move_toward': 'Line moved toward them', 'night': 'Night games',
          'noon': 'Noon games', 'primetime': 'Primetime games', 'late': 'Late-afternoon games'}
for _k, (_, _, _nm) in enumerate(BUCKETS):
    HABITS[f'b{_k}'] = f'{_nm[0].upper()}{_nm[1:]} lines'


def habit_name(f: str) -> str:
    return HABITS.get(f, f)


def cfb_opens(extra: pd.DataFrame | None = None) -> dict:
    """Opening home spreads: bundled history (2016-2025) + this season's CFBD openers."""
    from pathlib import Path
    out = {}
    p = Path(__file__).resolve().parent / 'cfb_openers.csv.gz'
    if p.exists():
        o = pd.read_csv(p)
        out.update(zip(o['id'].astype(str), o['open_home']))
    if extra is not None and len(extra) and 'open_home' in extra:
        e = extra.dropna(subset=['open_home'])
        out.update(zip(e['id'].astype(str), e['open_home']))
    return out


FEEDBACK_KEY = 'style/feedback.json'


def load_feedback(store) -> list:
    return (store.get_json(FEEDBACK_KEY) or {}).get('votes', [])


def add_feedback(store, league: str, row: dict, vote: int):
    """Save a thumbs up (+1) / down (-1) on a suggested side; replaces an earlier vote on the same side."""
    votes = [v for v in load_feedback(store)
             if not (v.get('league') == league and str(v.get('id')) == str(row['id']) and v.get('side') == row['side'])]
    votes.append(dict(league=league, id=str(row['id']), side=row['side'], team=row['team'], leg=row.get('leg'),
                      season=int(row['season']), week=int(row['week']), vote=int(vote), feats=row['feats'],
                      at=pd.Timestamp.now(tz='UTC').isoformat()))
    store.put_json(FEEDBACK_KEY, dict(votes=votes))
