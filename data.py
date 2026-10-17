"""Glue: bundled history + freshly downloaded current-season data -> one dataset per league."""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd

from history import load_history, combine, cfb_games_frame, cfb_team_games, nfl_team_games, HIST
from storage import read_frame, write_frame
import nfl_data, cfb_data


def current_season(now=None) -> int:
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 3 else now.year - 1


# ---------------------------------------------------------------- refresh
def refresh_nfl(store, log=print):
    season = current_season()
    sched = nfl_data.fetch_schedule()
    write_frame(store, 'cache/nfl/schedule.csv.gz', sched[sched.season >= 2016])
    log(f'NFL schedule/scores/lines loaded ({len(sched[sched.season == season])} games this season)')
    try:
        import injuries
        pbp = nfl_data.fetch_pbp(season)
        summ = nfl_data.game_summaries(pbp)
        write_frame(store, f'cache/nfl/pbp_{season}.csv.gz', summ)
        store.put_json('cache/nfl/key_players.json', injuries.key_players_nfl(pbp))
        log(f'NFL play-by-play summarized ({summ["id"].nunique()} games)')
    except Exception as exc:
        log(f'NFL play-by-play not available yet ({exc}); ratings use bundled history + scores')
    store.put_json('cache/nfl/refreshed.json', {'at': datetime.now(timezone.utc).isoformat()})


def refresh_cfb(store, key, log=print):
    api = cfb_data.CFBD(key, store)
    cfb_data.refresh_season(api, store, current_season(), log=log)
    log(f'CFB season data loaded (CFBD calls this month: {api.usage_this_month()})')


# ---------------------------------------------------------------- load
def league_data(league: str, store) -> dict:
    """Return dict(games, tg, extra) where extra has per-game live-only columns."""
    g, tg = load_history(league)
    season = current_season()
    extra = pd.DataFrame(columns=['id'])
    if league == 'NFL':
        sched = read_frame(store, 'cache/nfl/schedule.csv.gz')
        if sched is not None:
            sched['id'] = sched['id'].astype(str)
            sched['kickoff'] = pd.to_datetime(sched['kickoff'], utc=True, format='ISO8601')
            new_g = sched[['id', 'league', 'season', 'week', 'season_type', 'kickoff', 'away', 'home', 'neutral',
                           'away_score', 'home_score', 'completed', 'spread_home', 'total']].copy()
            new_g['neutral'] = new_g['neutral'].astype(bool)
            new_g['completed'] = new_g['completed'].astype(bool)
            summ = read_frame(store, f'cache/nfl/pbp_{season}.csv.gz')
            new_tg = nfl_team_games(new_g[new_g.season == season], summ) if summ is not None and len(summ) else tg.iloc[0:0]
            g, tg = combine(g, tg, new_g, new_tg)
            extra = sched[['id', 'away_rest', 'home_rest', 'roof', 'div_game', 'away_qb', 'home_qb']]
    else:
        prefix = f'cache/cfb/{season}'
        raw_g = read_frame(store, f'{prefix}/games.csv.gz')
        if raw_g is not None and len(raw_g):
            raw_g['id'] = raw_g['id'].astype(str)
            lines = read_frame(store, f'{prefix}/lines.csv.gz')
            adv = read_frame(store, f'{prefix}/advanced.csv.gz')
            box = read_frame(store, f'{prefix}/box.csv.gz')
            new_g = cfb_games_frame(raw_g, lines)
            if adv is not None and box is not None and len(adv):
                new_tg = cfb_team_games(new_g, adv, box)
            else:
                new_tg = tg.iloc[0:0]
            g, tg = combine(g, tg, new_g, new_tg)
            ex = raw_g[['id', 'venue_id', 'venue', 'away_conf', 'home_conf']].copy()
            if lines is not None and len(lines):
                lines['id'] = lines['id'].astype(str)
                ex = ex.merge(lines[['id', 'open_home', 'total_open']], on='id', how='left')
            extra = ex
    return dict(games=g, tg=tg, extra=extra)


def refreshed_at(store, league: str):
    key = 'cache/nfl/refreshed.json' if league == 'NFL' else f'cache/cfb/{current_season()}/refreshed.json'
    v = store.get_json(key)
    return v.get('at') if v else None


# ---------------------------------------------------------------- one-call update (app + scheduled job)
def upcoming_rows(data: dict, league: str, days: int = 8) -> list[dict]:
    g = data['games']
    now = pd.Timestamp(datetime.now(timezone.utc))
    up = g[(~g.completed) & (g.kickoff >= now - pd.Timedelta(hours=4)) & (g.kickoff <= now + pd.Timedelta(days=days))]
    if league == 'CFB':
        up = up[(up.away != 'FCS') & (up.home != 'FCS')]
    return up[['id', 'away', 'home']].to_dict('records')


def age_hours(stamp) -> float:
    if not stamp:
        return 1e9
    try:
        t = pd.Timestamp(stamp)
        t = t.tz_localize('UTC') if t.tzinfo is None else t
        return (pd.Timestamp(datetime.now(timezone.utc)) - t).total_seconds() / 3600
    except (ValueError, TypeError):
        return 1e9


def last_lines(store, league: str):
    keys = store.list(f'odds/{league}/')
    if not keys:
        return None
    stamp = keys[-1].rsplit('/', 1)[-1].split('.json')[0]          # 2026-10-16T13-00-00
    d, t = stamp.split('T')
    return f"{d}T{t.replace('-', ':')}+00:00"


def _last_frame() -> str:
    """Where the last exception happened (file:line), for error reports."""
    import traceback, sys
    tb = traceback.extract_tb(sys.exc_info()[2])
    return f'{tb[-1].filename.rsplit("/", 1)[-1]}:{tb[-1].lineno} {tb[-1].line}' if tb else ''


def update_all(store, secrets: dict, league: str, log=print, stats=True, lines=True, splits=True,
               injuries=True) -> list[str]:
    """Refresh stats/scores, sportsbook lines, DraftKings splits and injury reports.  Returns problems."""
    from odds import OddsAPI
    import splits as splits_mod
    import injuries as inj_mod
    problems = []
    if injuries:
        try:
            inj_mod.refresh(store, league, log=log)
        except Exception as exc:
            problems.append(f'Injuries: {type(exc).__name__}: {exc} | ' + _last_frame())
    if stats:
        try:
            if league == 'NFL':
                refresh_nfl(store, log=log)
            else:
                refresh_cfb(store, secrets.get('CFBD_API_KEY', ''), log=log)
        except Exception as exc:
            problems.append(f'Stats: {type(exc).__name__}: {exc} | ' + _last_frame())
    if lines:
        try:
            api = OddsAPI(secrets.get('ODDS_API_KEY', ''), store)
            ev = api.fetch(league)
            log(f"Lines: {len(ev)} games saved (Odds API credits left: {api.last_headers.get('x-requests-remaining')})")
        except Exception as exc:
            problems.append(f'Lines: {type(exc).__name__}: {exc} | ' + _last_frame())
    if splits:
        try:
            rows = upcoming_rows(league_data(league, store), league)
            res = splits_mod.auto_refresh(store, league, rows, log=log)
            if not res['games']:
                problems.append('Splits: no games could be read from DraftKings or VSiN right now.')
        except Exception as exc:
            problems.append(f'Splits: {type(exc).__name__}: {exc} | ' + _last_frame())
    return problems


def stale(store, league: str) -> dict:
    """Which parts are due for an automatic update."""
    import splits as splits_mod
    inj = store.get_json(f'injuries/{league}.json') or {}
    return dict(stats=age_hours(refreshed_at(store, league)) > 12,
                lines=age_hours(last_lines(store, league)) > 12,
                splits=age_hours(splits_mod.last_capture(store, league)) > 1,
                injuries=age_hours(inj.get('at')) > 2)
