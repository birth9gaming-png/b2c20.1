"""Assemble the weekly board: one row per upcoming game with every signal."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

import config, odds as odds_mod, splits as splits_mod, spots as spots_mod, weather as weather_mod
from model import slate_features, predict_margin, predict_total, week_rank_fraction, cover_prob_model_side
from storage import read_frame


def _fmt_spread(x):
    if x is None or pd.isna(x):
        return '—'
    return 'PK' if abs(x) < 0.01 else f'{x:+g}'


def team_line(home_spread, side):
    """Team's own spread from the home spread."""
    if home_spread is None or pd.isna(home_spread):
        return np.nan
    return home_spread if side == 'home' else -home_spread


def verdict_for(side: str, row: dict, league: str) -> tuple[str, str]:
    """(verdict, explanation) for betting `side` in this game."""
    edge = row.get('edge')
    if edge is None or pd.isna(edge):
        return 'No model line', 'No current line or ratings for this game.'
    model_side = 'away' if edge > 0 else 'home'
    ae, frac = abs(edge), row.get('rank_frac', 1.0)
    pts = f"{ae:.1f} pts"
    if ae < config.LEAN_EDGE[league]:
        return 'No real edge either way', f'Model line is within {pts} of the market.'
    strong = frac < 0.05 and ae >= config.BIG_EDGE[league] * 0.5
    if side == model_side:
        return ('Model backs it (strong)' if strong else 'Model leans this way',
                f"Model is {pts} more bullish on this side than the market"
                + (' — one of the biggest disagreements this week.' if strong else '.'))
    return ('Model says stay off' if strong else 'Model leans against',
            f"Model is {pts} against this side" + (' — one of the biggest disagreements this week.' if strong else '.'))


def signals(side: str, row: dict) -> list[str]:
    """Market and situation signals pointing at `side` (the model edge is scored separately)."""
    other = 'home' if side == 'away' else 'away'
    hits = []
    mv = row.get('move_home')
    if mv is not None and pd.notna(mv) and abs(mv) >= 0.5:
        worse_for = 'home' if mv < 0 else 'away'          # side whose number got worse = market moving to it
        if worse_for == side:
            hits.append('Line moving its way')
    fl = row.get('split_flags') or {}
    if fl.get('sharp_side') == side:
        hits.append('Bigger-money side')
    if fl.get('rlm_side') == side:
        hits.append('Reverse line move')
    sp = (row.get('spots') or {}).get(other, {})
    for k in ('sandwich', 'lookahead', 'letdown'):
        if sp.get(k):
            hits.append(f'Opponent in {k} spot')
            break
    return hits


def confluence(side: str, row: dict, league: str) -> tuple[int, list[str]]:
    """Model (if it leans this side) + market/situation signals for `side`."""
    edge = row.get('edge')
    hits = []
    if edge is not None and pd.notna(edge) and abs(edge) >= config.LEAN_EDGE[league] and (('away' if edge > 0 else 'home') == side):
        hits.append('Model')
    hits += signals(side, row)
    return len(hits), hits


def pick_score(side: str, row: dict, league: str) -> tuple[float, list[str]]:
    """One 0-100 number for betting `side`.  Every piece is listed so you can see why.

      model edge     up to 50  (the Vegas-anchored model only moves off the line for proven spots:
                               CFB 1.5 pts = 50, NFL 1 pt = 50; 0 if the model is against this side)
      each signal    +12       (line moving its way, bigger-money side, reverse line move, opponent in a spot)
      each signal    -12       for the same signals on the OTHER side
      key injury     -25       a key player out (or QB questionable) on this side - the model can't fully see it
    """
    other = 'home' if side == 'away' else 'away'
    parts, score = [], 0.0
    edge = row.get('edge')
    if edge is not None and pd.notna(edge):
        toward = edge if side == 'away' else -edge
        pts = 50.0 * max(0.0, min(1.0, toward / config.BIG_EDGE[league]))
        score += pts
        parts.append(f'model edge {toward:+.1f} pts → +{pts:.0f}')
    for h in signals(side, row):
        score += 12; parts.append(f'{h} +12')
    for h in signals(other, row):
        score -= 12; parts.append(f'{h} (other side) −12')
    inj = (row.get('injuries') or {}).get(side) or {}
    if inj.get('missing') or inj.get('qb_questionable'):
        score -= 25; parts.append('key player out/questionable −25')
    return round(max(0.0, min(100.0, score)), 1), parts


def grade_from_score(score: float) -> str:
    return 'A' if score >= 70 else 'B' if score >= 55 else 'C' if score >= 40 else 'D'


def grade(n: int) -> str:            # kept for older saved picks
    return 'A' if n >= 4 else 'B' if n == 3 else 'C' if n == 2 else 'D'


def build_board(league: str, data: dict, model: dict, store, now=None, days_ahead: int = 8,
                with_weather: bool = True) -> pd.DataFrame:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    games, tg, extra = data['games'], data['tg'], data['extra']
    slate = games[(~games.completed) & (games.kickoff >= now - pd.Timedelta(hours=4)) &
                  (games.kickoff <= now + pd.Timedelta(days=days_ahead))].copy()
    if league == 'CFB':
        slate = slate[(slate.away != 'FCS') & (slate.home != 'FCS')]
    if slate.empty:
        return pd.DataFrame()
    if len(extra):
        slate = slate.merge(extra, on='id', how='left')
    hist = games[games.completed]
    feats, ratings, mratings = slate_features(hist, tg, slate, league, now=now)
    feats = feats.set_index('id').reindex(slate.id).reset_index()
    ok = feats[model['features']].notna().all(axis=1).values
    pred = np.where(ok, predict_margin(model, feats.fillna(0)), np.nan)
    slate['model_home_raw'] = pred                       # stats-only model, before injury adjustment
    # ---- injuries: who is missing for each team (model can't see this in team stats)
    import injuries as inj_mod
    try:
        inj_rows, inj_at, inj_src = inj_mod.load(store, league)
    except Exception as exc:
        print(f'injuries failed: {type(exc).__name__}: {exc}')
        inj_rows, inj_at, inj_src = pd.DataFrame(columns=['team', 'player', 'pos', 'status']), None, None
    try:
        inj_rows = inj_mod.resolve_teams(inj_rows, league, pd.concat([games['away'], games['home']]).unique())
    except Exception as exc:
        print(f'injury team matching failed: {type(exc).__name__}: {exc}')
    if league == 'NFL':
        keys = store.get_json('cache/nfl/key_players.json') or {}
    else:
        from data import current_season
        keys = store.get_json(f'cache/cfb/{current_season()}/key_players.json') or {}
    reports, adj = [], []
    for g in slate.itertuples():
        rep = {}
        for side, team in (('away', g.away), ('home', g.home)):
            rep[side] = inj_mod.team_report(team, inj_mod.match_team_rows(inj_rows, team, league), keys.get(team, []))
        a = 0.0
        q = config.QB_OUT_POINTS.get(league, 0.0)
        if rep['away']['qb_out']:
            a -= q                                        # away team worse -> away-minus-home margin down
        if rep['home']['qb_out']:
            a += q
        reports.append(rep); adj.append(a)
    slate['injuries'] = reports
    slate['injury_adjust'] = adj
    slate['injury_source'] = f'{inj_src} ({str(inj_at)[:16]} UTC)' if inj_src else None
    slate['stats_home'] = slate['model_home_raw'] + slate['injury_adjust']   # stats model (reference only)
    slate['model_home'] = slate['stats_home']
    slate['model_total'] = predict_total(model, feats)
    slate['away_games'] = feats['away_games'].values if 'away_games' in feats else np.nan
    slate['home_games'] = feats['home_games'].values if 'home_games' in feats else np.nan

    # ---- sportsbook lines
    q = odds_mod.load_snapshots(store, league, now=now)
    cb = odds_mod.current_board(q)
    slate = odds_mod.attach_events(slate, cb, league)
    hist_lines = odds_mod.line_history(q)
    cbi = cb.set_index('event_id') if len(cb) else pd.DataFrame()
    cur, opn, open_src, best_a, best_h, sharp, tot_cur, n_snap = [], [], [], [], [], [], [], []
    for g in slate.itertuples():
        e = cbi.loc[g.event_id] if g.event_id is not None and len(cbi) and g.event_id in cbi.index else None
        reversed_ = e is not None and odds_mod.match_team(e.home_team, [g.away, g.home]) == g.away
        flip = -1 if reversed_ else 1
        c = flip * e.spread_home if e is not None else g.spread_home
        cur.append(c)
        o_candidates = []
        if e is not None and pd.notna(e.open_home_seen):
            o_candidates.append((pd.Timestamp(e.open_seen_at), flip * e.open_home_seen, 'first saved snapshot'))
        cfbd_open = getattr(g, 'open_home', np.nan)
        if cfbd_open is not None and pd.notna(cfbd_open):
            o_candidates.append((pd.Timestamp.min.tz_localize('UTC'), cfbd_open, 'CFBD opener'))
        if o_candidates:
            o = min(o_candidates, key=lambda x: x[0]); opn.append(o[1]); open_src.append(o[2])
        else:
            opn.append(np.nan); open_src.append(None)
        if e is not None:
            a_side, h_side = ('home', 'away') if reversed_ else ('away', 'home')
            best_a.append((e[f'best_{a_side}_point'], e[f'best_{a_side}_price'], e[f'best_{a_side}_book']))
            best_h.append((e[f'best_{h_side}_point'], e[f'best_{h_side}_price'], e[f'best_{h_side}_book']))
            sharp.append(flip * e.sharp_home if pd.notna(e.sharp_home) else np.nan)
            tot_cur.append(e.total); n_snap.append(int(e.snapshots))
        else:
            best_a.append((np.nan, np.nan, None)); best_h.append((np.nan, np.nan, None))
            sharp.append(np.nan); tot_cur.append(g.total); n_snap.append(0)
    slate['current_home'] = cur
    slate['open_home'] = opn
    slate['open_source'] = open_src
    slate['sharp_home'] = sharp
    slate['current_total'] = tot_cur
    slate['snapshots'] = n_snap
    slate['best_away'] = best_a
    slate['best_home'] = best_h
    slate['move_home'] = slate.current_home - slate.open_home
    # ---- your rule flags (rankings, lookahead, letdown, trap, rivalry) and the Vegas-anchored model
    import rules as rules_mod, proven
    try:
        polls = rules_mod.load_polls(store.get_json(f'cache/cfb/{int(slate.season.max())}/polls.json')) if league == 'CFB' else {}
        seasons = set(int(x) for x in slate.season.dropna().unique())
        gs = games[games.season.isin(seasons)].copy()
        if 'spread_home' not in gs:
            gs['spread_home'] = np.nan
        flags = rules_mod.flag_games(gs, league, polls=polls,
                                     current_home=dict(zip(slate.id, slate.current_home))).set_index('id').to_dict('index')
    except Exception as exc:
        print(f'rule flags failed: {type(exc).__name__}: {exc}')
        flags = {}
    slate['rule_flags'] = [flags.get(i, {}) for i in slate.id]
    slate = proven.anchor(slate, flags, league)
    slate['edge'] = slate.model_home - slate.current_home         # >0: model likes away more than market
    slate['edge_open'] = slate.stats_home - slate.open_home       # the stats model only helps early, vs the opener
    slate['model_side'] = np.where(slate.edge > 0, 'away', np.where(slate.edge < 0, 'home', None))
    slate['rank_frac'] = week_rank_fraction(slate.edge.abs(), [slate.season, slate.week]).fillna(1.0)
    slate['p_model_side'] = [proven.side_prob(league, r, r['model_side']) if r['model_side'] else np.nan
                             for r in slate.to_dict('records')]
    if model['calibration'].get('rank_open'):
        fo = week_rank_fraction(slate.edge_open.abs(), [slate.season, slate.week]).fillna(1.0)
        slate['p_model_side_open'] = [cover_prob_model_side(model, f, at_open=True) if pd.notna(e) else np.nan
                                      for f, e in zip(fo, slate.edge_open)]
    else:
        slate['p_model_side_open'] = np.nan
    # line moved toward the model since open? (points of value the market already took)
    slate['moved_toward_model'] = np.sign(slate.edge_open) * slate.move_home

    # ---- spots
    try:
        sched = spots_mod.team_schedule(games)
        slate['spots'] = [spots_mod.spots_for_game(g, league, sched, mratings, g.current_home) for g in slate.itertuples()]
    except Exception as exc:                                   # never let one feature take the board down
        print(f'spots failed: {type(exc).__name__}: {exc}')
        slate['spots'] = [{'away': {}, 'home': {}} for _ in range(len(slate))]

    # ---- DraftKings splits
    try:
        latest = splits_mod.latest(store, league)
    except Exception as exc:
        print(f'splits failed: {type(exc).__name__}: {exc}')
        latest = {}
    sp_rows, fl_rows = [], []
    for g in slate.itertuples():
        spl = latest.get(g.id, (None, None))[0]
        sp_rows.append(spl)
        fl_rows.append(splits_mod.flags(spl, g.open_home, g.current_home))
    slate['splits'] = sp_rows
    slate['split_flags'] = fl_rows

    # ---- weather (cached ~3h per game)
    try:
        venues = read_frame(store, 'cache/cfb/venues.csv.gz') if league == 'CFB' else None
    except Exception:
        venues = None
    wx = []
    for g in slate.itertuples():
        w, indoor = None, False
        if with_weather:
            lat, lon, indoor = weather_mod.location(league, g, venues)
            key = f"weather/{league}/{g.id}_{now.strftime('%Y%m%d')}_{now.hour // 3}.json"
            w = store.get_json(key)
            if w is None and not indoor:
                try:
                    w = weather_mod.forecast(lat, lon, g.kickoff)
                    store.put_json(key, w or {})
                except Exception:
                    w = None
        wx.append(dict(data=w or None, indoor=indoor, flags=weather_mod.flags(w or None, indoor)))
    slate['weather'] = wx

    # ---- verdicts / confluence for the model side
    rows = slate.to_dict('records')
    for r in rows:
        side = r['model_side']
        r['confluence_n'], r['confluence'] = confluence(side, r, league) if side else (0, [])
        r['score'], r['score_parts'] = pick_score(side, r, league) if side else (0.0, [])
        r['grade'] = grade_from_score(r['score'])
        r['verdict_away'] = verdict_for('away', r, league)[0]
        r['verdict_home'] = verdict_for('home', r, league)[0]
    out = pd.DataFrame(rows)
    return out.sort_values('kickoff').reset_index(drop=True)


DEFAULT_RULES = dict(picks=3, min_edge_cfb=1.0, min_edge_nfl=0.75, max_spread_cfb=28.0, max_spread_nfl=14.0,
                     min_market_signals=0, require_spot=False, skip_if_moved_against=False, skip_if_value_gone=False,
                     skip_key_injury=True, min_grade='C')


RULES_VERSION = 20


def load_rules(store) -> dict:
    saved = store.get_json('settings/rules.json') or {}
    if saved.get('v', 0) < RULES_VERSION:          # model edges are now in Vegas-anchored points: reset the minimums
        saved = {k: v for k, v in saved.items() if not k.startswith('min_edge')}
    return {**DEFAULT_RULES, **saved, 'v': RULES_VERSION}


def save_rules(store, rules: dict):
    store.put_json('settings/rules.json', rules)


def passes_rules(r, league: str, rules: dict) -> tuple[bool, str]:
    """Your stipulations for a game to make the shortlist.  Returns (ok, reason if not)."""
    lg = league.lower()
    if r.get('edge') is None or pd.isna(r['edge']):
        return False, 'no model line'
    if abs(r['edge']) < rules[f'min_edge_{lg}']:
        return False, 'edge below your minimum'
    if pd.notna(r.get('current_home')) and abs(r['current_home']) > rules[f'max_spread_{lg}']:
        return False, 'spread bigger than your maximum'
    if not r.get('model_side'):
        return False, 'no model side'
    if len(signals(r['model_side'], r)) < rules['min_market_signals']:
        return False, 'not enough market/situation signals'
    inj = (r.get('injuries') or {}).get(r.get('model_side')) or {}
    if rules.get('skip_key_injury', True) and (inj.get('missing') or inj.get('qb_questionable')):
        return False, 'key player out on the model side'
    if 'ABCD'.index(r.get('grade', 'D')) > 'ABCD'.index(rules.get('min_grade', 'D')):
        return False, f"grade below {rules.get('min_grade')}"
    other = 'home' if r.get('model_side') == 'away' else 'away'
    sp = (r.get('spots') or {}).get(other, {})
    if rules['require_spot'] and not any(sp.get(k) for k in ('lookahead', 'letdown', 'sandwich')):
        return False, 'opponent not in a spot'
    mv = r.get('moved_toward_model')
    if rules['skip_if_moved_against'] and pd.notna(mv) and mv <= -0.5:
        return False, 'line moved against the model'
    if rules['skip_if_value_gone'] and pd.notna(mv) and mv >= 0.5 and abs(r['edge']) < rules[f'min_edge_{lg}'] + 1:
        return False, 'most of the value already moved'
    return True, ''


def leans(board: pd.DataFrame, league: str, rules: dict | None = None, n: int = 3) -> pd.DataFrame:
    """Best games that miss ONLY your grade cut (shown separately, never as picks)."""
    if board.empty:
        return board
    cut = {**DEFAULT_RULES, **(rules or {})}.get('min_grade', 'C')
    loose = {**DEFAULT_RULES, **(rules or {}), 'min_grade': 'D', 'picks': 50}
    sl = shortlist(board, league, rules=loose)
    return sl[sl.grade > cut].head(n) if len(sl) else sl


def shortlist(board: pd.DataFrame, league: str, n: int | None = None, rules: dict | None = None) -> pd.DataFrame:
    if board.empty:
        return board
    rules = {**DEFAULT_RULES, **(rules or {})}
    b = board[board.edge.notna()].copy()
    keep = [passes_rules(r, league, rules)[0] for r in b.to_dict('records')]
    b = b[keep]
    b['abs_edge'] = b.edge.abs()
    return b.sort_values(['score', 'abs_edge'], ascending=[False, False]).head(n or rules['picks'])


def side_label(row, side):
    team = row['away'] if side == 'away' else row['home']
    return f"{team} {_fmt_spread(team_line(row['current_home'], side))}"
