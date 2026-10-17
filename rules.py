"""Your handicapping rules, written as exact definitions so each one can be tested on past seasons.

Definitions (from you, Oct 16 2026):
  TRAP       CFB: an unranked team favored over a ranked team.
             NFL: the favorite has a clearly worse record than the underdog (1-3 Jets favored over 3-1 Browns).
             The trap is betting the "better" team; the rule backs the favorite Vegas chose.
  LOOKAHEAD  next game (within 8 days) is a ranked opponent or a rival, and today's opponent is neither.
             The rule fades the team looking ahead.
  LETDOWN    last game (within 9 days) was a WIN over a ranked opponent or a rival - their big game -
             and today's opponent is neither.  The rule fades the team with the hangover.
  RIVALRY    named historical CFB rivalries; NFL division games.
  LINE MOVE  the line moved toward a side between open and kickoff by 0.5-2.5 points.  Moves of 3+ points
             usually mean news (injury etc.) and are flagged instead.
  POWER      at least one power-conference team (or Notre Dame) in the game.
NFL has no rankings, so "ranked opponent" there means a division rival or a team at .600 or better (3+ games).
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

CFB_RIVALS = {frozenset(p) for p in (
    # Big Ten
    ('Ohio State', 'Michigan'), ('Michigan', 'Michigan State'), ('Indiana', 'Purdue'), ('Iowa', 'Minnesota'),
    ('Minnesota', 'Wisconsin'), ('Iowa', 'Wisconsin'), ('Iowa', 'Nebraska'), ('Illinois', 'Northwestern'),
    ('Michigan', 'Minnesota'), ('Penn State', 'Michigan State'), ('Illinois', 'Ohio State'), ('Purdue', 'Illinois'),
    ('Nebraska', 'Wisconsin'), ('Minnesota', 'Penn State'), ('USC', 'UCLA'), ('Oregon', 'Washington'),
    ('Ohio State', 'Penn State'), ('Maryland', 'Penn State'),
    # SEC
    ('Alabama', 'Auburn'), ('Alabama', 'Tennessee'), ('Alabama', 'LSU'), ('Georgia', 'Florida'), ('Georgia', 'Auburn'),
    ('Georgia', 'Georgia Tech'), ('Florida', 'Florida State'), ('Florida', 'Tennessee'), ('Florida', 'LSU'),
    ('LSU', 'Arkansas'), ('LSU', 'Ole Miss'), ('Ole Miss', 'Mississippi State'), ('Texas', 'Oklahoma'),
    ('Texas', 'Texas A&M'), ('Arkansas', 'Texas A&M'), ('Missouri', 'Arkansas'), ('Missouri', 'Kansas'),
    ('Kentucky', 'Louisville'), ('Kentucky', 'Tennessee'), ('Tennessee', 'Vanderbilt'), ('South Carolina', 'Clemson'),
    ('Oklahoma', 'Oklahoma State'), ('LSU', 'Texas A&M'), ('Georgia', 'South Carolina'), ('Auburn', 'LSU'),
    # ACC / Big 12 / independents
    ('Florida State', 'Miami'), ('North Carolina', 'NC State'), ('North Carolina', 'Duke'),
    ('Virginia', 'Virginia Tech'), ('Miami', 'Virginia Tech'), ('Pittsburgh', 'West Virginia'), ('Stanford', 'California'),
    ('Clemson', 'Florida State'), ('Boston College', 'Syracuse'), ('Kansas', 'Kansas State'), ('Iowa', 'Iowa State'),
    ('Utah', 'BYU'), ('Arizona', 'Arizona State'), ('TCU', 'Baylor'), ('TCU', 'SMU'), ('Baylor', 'Texas'),
    ('Texas Tech', 'TCU'), ('West Virginia', 'Virginia Tech'), ('Colorado', 'Colorado State'), ('Colorado', 'Nebraska'),
    ('Colorado', 'Utah'), ('Houston', 'Rice'), ('Cincinnati', 'Louisville'), ('Cincinnati', 'Miami (OH)'),
    ('UCF', 'South Florida'), ('Washington', 'Washington State'), ('Oregon', 'Oregon State'),
    ('Notre Dame', 'USC'), ('Notre Dame', 'Navy'), ('Notre Dame', 'Michigan'), ('Notre Dame', 'Michigan State'),
    ('Notre Dame', 'Purdue'), ('Notre Dame', 'Stanford'), ('Notre Dame', 'Boston College'),
    ('Army', 'Navy'), ('Army', 'Air Force'), ('Navy', 'Air Force'),
    # Group of 5
    ('Nevada', 'UNLV'), ('New Mexico', 'New Mexico State'), ('Utah State', 'BYU'), ('Marshall', 'Ohio'),
    ('Toledo', 'Bowling Green'), ('Western Michigan', 'Central Michigan'), ('Central Michigan', 'Eastern Michigan'),
    ('Ball State', 'Northern Illinois'), ('Akron', 'Kent State'), ('Miami (OH)', 'Ohio'), ('Louisiana Tech', 'Southern Miss'),
    ('Troy', 'South Alabama'), ('Georgia Southern', 'App State'), ('Louisiana', 'UL Monroe'), ('Tulane', 'Southern Miss'),
    ('Middle Tennessee', 'Western Kentucky'), ('UTEP', 'New Mexico State'), ('Wyoming', 'Colorado State'),
    ('Fresno State', 'San Diego State'), ('San José State', 'Fresno State'), ("Hawai'i", 'Fresno State'),
    ('Boise State', 'Fresno State'), ('Memphis', 'Tulane'), ('East Carolina', 'Marshall'))}

POWER = {'cfb:sec', 'cfb:big-ten', 'cfb:big-12', 'cfb:acc', 'cfb:pac-12'}
NFL_DIVISIONS = {
    'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'], 'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
    'AFC South': ['HOU', 'IND', 'JAX', 'TEN'], 'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
    'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'], 'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
    'NFC South': ['ATL', 'CAR', 'NO', 'TB'], 'NFC West': ['ARI', 'LA', 'SF', 'SEA']}
DIV_OF = {t: d for d, ts in NFL_DIVISIONS.items() for t in ts}


# ---------------------------------------------------------------- lookups
def load_polls(extra: list[dict] | None = None) -> dict:
    """(season, week) -> {team: rank} for AP polls."""
    rows = json.loads(gzip.decompress((HERE / 'cfb_polls.json.gz').read_bytes())) + list(extra or [])
    out = {}
    for p in rows:
        ranks = {k.split(':', 1)[-1]: v for k, v in (p.get('ranks') or {}).items()}
        out[(int(p['season']), int(p['week']))] = ranks
    return out


def polls_from_cfbd(rows: list[dict]) -> list[dict]:
    """CFBD /rankings -> our poll rows [{season, week, ranks: {team: rank}}] (AP poll, else the first poll)."""
    out = []
    for wk in rows or []:
        polls = wk.get('polls') or []
        ap = next((p for p in polls if 'ap' in str(p.get('poll', '')).lower()), polls[0] if polls else None)
        if not ap:
            continue
        ranks = {r.get('school'): int(r['rank']) for r in ap.get('ranks') or [] if r.get('school') and r.get('rank')}
        out.append(dict(season=int(wk['season']), week=int(wk['week']), ranks=ranks))
    return out


def rank_before(polls: dict, season: int, week: int) -> dict:
    weeks = [w for (s, w) in polls if s == season and w <= week]
    return polls[(season, max(weeks))] if weeks else {}


def load_conferences() -> dict:
    return json.loads((HERE / 'cfb_conferences.json').read_text())['seasons']


def is_power(conf: dict, team: str, season: int) -> bool:
    if team == 'Notre Dame':
        return True
    c = conf.get(str(season), {}).get(team) or conf.get(str(max(int(s) for s in conf)), {}).get(team)
    if c == 'cfb:pac-12' and season >= 2024:
        return False                       # two-team remnant
    return c in POWER


def rivals(league: str, a: str, b: str) -> bool:
    if league == 'NFL':
        return DIV_OF.get(a) is not None and DIV_OF.get(a) == DIV_OF.get(b)
    return frozenset((a, b)) in CFB_RIVALS


# ---------------------------------------------------------------- flags for every game
def flag_games(games: pd.DataFrame, league: str, polls: dict | None = None, conf: dict | None = None,
               current_home: dict | None = None) -> pd.DataFrame:
    """One row per game with every rule flag.  Uses only information known before that game.

    games       : full schedule (completed games with scores + upcoming games), our games table
    current_home: optional {game_id: home spread now} for upcoming games (else the table's spread_home)
    """
    polls = polls if polls is not None else (load_polls() if league == 'CFB' else {})
    conf = conf if conf is not None else (load_conferences() if league == 'CFB' else {})
    g = games.sort_values('kickoff').reset_index(drop=True)
    sched = {}
    for r in g.itertuples():
        for team in (r.away, r.home):
            sched.setdefault((r.season, team), []).append(r)
    out = []
    for r in g.itertuples():
        line = (current_home or {}).get(r.id, r.spread_home)
        ranks = rank_before(polls, r.season, r.week) if league == 'CFB' else {}
        rec = {}
        info = dict(id=r.id, rival=rivals(league, r.away, r.home), line=line)
        for side, team, opp in (('away', r.away, r.home), ('home', r.home, r.away)):
            hist = [x for x in sched.get((r.season, team), []) if x.kickoff < r.kickoff and x.completed]
            w = sum(1 for x in hist if (x.away_score - x.home_score) * (1 if x.away == team else -1) > 0)
            rec[side] = (w, len(hist))
            info[f'{side}_rank'] = ranks.get(team)
            info[f'{side}_wins'], info[f'{side}_games'] = w, len(hist)
        for side, team, opp in (('away', r.away, r.home), ('home', r.home, r.away)):
            fut = [x for x in sched.get((r.season, team), []) if x.kickoff > r.kickoff]
            past = [x for x in sched.get((r.season, team), []) if x.kickoff < r.kickoff and x.completed]

            def big_opp(o, ref_ranks):
                if rivals(league, team, o):
                    return True
                if league == 'CFB':
                    return o in ref_ranks
                ow = [x for x in sched.get((r.season, o), []) if x.kickoff < r.kickoff and x.completed]
                if len(ow) < 3:
                    return False
                wins = sum(1 for x in ow if (x.away_score - x.home_score) * (1 if x.away == o else -1) > 0)
                return wins / len(ow) >= 0.6
            today_big = big_opp(opp, ranks)
            nxt = fut[0] if fut and (fut[0].kickoff - r.kickoff).days <= 8 else None
            nopp = (nxt.home if nxt.away == team else nxt.away) if nxt is not None else None
            prv = past[-1] if past and (r.kickoff - past[-1].kickoff).days <= 9 else None
            popp = (prv.home if prv.away == team else prv.away) if prv is not None else None
            pranks = rank_before(polls, r.season, prv.week) if (prv is not None and league == 'CFB') else {}
            pwon = prv is not None and (prv.away_score - prv.home_score) * (1 if prv.away == team else -1) > 0
            info[f'{side}_lookahead'] = bool(nopp and big_opp(nopp, ranks) and not today_big)
            info[f'{side}_letdown'] = bool(prv is not None and pwon and big_opp(popp, pranks) and not today_big)
            info[f'{side}_next'] = nopp
            info[f'{side}_prev'] = popp if pwon else None
        # trap: favorite is the "worse" team on paper
        fav = 'home' if line is not None and pd.notna(line) and line < 0 else 'away' if line is not None and pd.notna(line) and line > 0 else None
        dog = {'home': 'away', 'away': 'home'}.get(fav)
        trap = False
        if fav:
            if league == 'CFB':
                trap = info[f'{fav}_rank'] is None and info[f'{dog}_rank'] is not None
            else:
                (fw, fg), (dw, dg) = rec[fav], rec[dog]
                trap = fg >= 3 and dg >= 3 and dw / dg - fw / fg >= 0.25
        info['fav'] = fav
        info['trap'] = trap
        if league == 'CFB':
            info['power_game'] = is_power(conf, r.away, r.season) or is_power(conf, r.home, r.season)
            info['both_power'] = is_power(conf, r.away, r.season) and is_power(conf, r.home, r.season)
        else:
            info['power_game'] = info['both_power'] = True
        out.append(info)
    return pd.DataFrame(out)
