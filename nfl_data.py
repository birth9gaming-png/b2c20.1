"""NFL data from nflverse (free, no key): schedule/scores/closing lines and play-by-play."""
from __future__ import annotations
import io
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

from history import nfl_code

SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
PBP_URL = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz'
ET = ZoneInfo('America/New_York')

PBP_COLUMNS = ['game_id', 'season_type', 'week', 'posteam', 'defteam', 'play_type', 'yards_gained', 'epa',
               'pass_attempt', 'interception', 'no_play', 'qb_kneel', 'qb_spike', 'qb_dropback', 'sack',
               'cpoe', 'fumbled_1_team', 'fumbled_2_team', 'fumble_recovery_1_team', 'fumble_recovery_2_team',
               'passer_player_name', 'rusher_player_name', 'receiver_player_name']

# Odds API full names -> nflverse codes
TEAM_NAMES = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
}
CODE_TO_NAME = {v: k for k, v in TEAM_NAMES.items()}

# Stadium coordinates for weather; roof 'dome'/'retractable' games skip wind flags.
STADIUMS = {
    'ARI': (33.5276, -112.2626, 'retractable'), 'ATL': (33.7554, -84.4008, 'retractable'),
    'BAL': (39.2780, -76.6227, 'outdoors'), 'BUF': (42.7738, -78.7870, 'outdoors'),
    'CAR': (35.2258, -80.8528, 'outdoors'), 'CHI': (41.8623, -87.6167, 'outdoors'),
    'CIN': (39.0955, -84.5161, 'outdoors'), 'CLE': (41.5061, -81.6995, 'outdoors'),
    'DAL': (32.7473, -97.0945, 'retractable'), 'DEN': (39.7439, -105.0201, 'outdoors'),
    'DET': (42.3400, -83.0456, 'dome'), 'GB': (44.5013, -88.0622, 'outdoors'),
    'HOU': (29.6847, -95.4107, 'retractable'), 'IND': (39.7601, -86.1639, 'retractable'),
    'JAX': (30.3239, -81.6373, 'outdoors'), 'KC': (39.0489, -94.4839, 'outdoors'),
    'LV': (36.0909, -115.1833, 'dome'), 'LAC': (33.9535, -118.3392, 'dome'),
    'LA': (33.9535, -118.3392, 'dome'), 'MIA': (25.9580, -80.2389, 'outdoors'),
    'MIN': (44.9737, -93.2581, 'dome'), 'NE': (42.0909, -71.2643, 'outdoors'),
    'NO': (29.9511, -90.0812, 'dome'), 'NYG': (40.8128, -74.0742, 'outdoors'),
    'NYJ': (40.8128, -74.0742, 'outdoors'), 'PHI': (39.9008, -75.1675, 'outdoors'),
    'PIT': (40.4468, -80.0158, 'outdoors'), 'SF': (37.4030, -121.9700, 'outdoors'),
    'SEA': (47.5952, -122.3316, 'outdoors'), 'TB': (27.9759, -82.5033, 'outdoors'),
    'TEN': (36.1665, -86.7713, 'outdoors'), 'WAS': (38.9076, -76.8645, 'outdoors'),
}


def _get(url, timeout=60) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def schedule_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """nflverse games.csv -> our games table (+ extra NFL columns)."""
    g = raw.copy()
    g = g[g['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])]
    out = pd.DataFrame({
        'id': g['game_id'].astype(str), 'league': 'NFL', 'season': g['season'].astype(int),
        'week': g['week'].astype(int),
        'season_type': np.where(g['game_type'] == 'REG', 'regular', 'postseason'),
        'away': g['away_team'].map(nfl_code), 'home': g['home_team'].map(nfl_code),
        'neutral': g['location'].astype(str).str.lower().eq('neutral'),
        'away_score': pd.to_numeric(g['away_score'], errors='coerce'),
        'home_score': pd.to_numeric(g['home_score'], errors='coerce'),
        'spread_home': -pd.to_numeric(g['spread_line'], errors='coerce'),   # spread_line > 0 = home favored
        'total': pd.to_numeric(g['total_line'], errors='coerce'),
        'away_rest': pd.to_numeric(g.get('away_rest'), errors='coerce'),
        'home_rest': pd.to_numeric(g.get('home_rest'), errors='coerce'),
        'roof': g.get('roof'), 'div_game': pd.to_numeric(g.get('div_game'), errors='coerce'),
        'away_qb': g.get('away_qb_name'), 'home_qb': g.get('home_qb_name'),
    })
    times = (g['gameday'].astype(str) + ' ' + g['gametime'].fillna('13:00').astype(str))
    kick = pd.to_datetime(times, errors='coerce')
    out['kickoff'] = kick.dt.tz_localize(ET, ambiguous='NaT', nonexistent='shift_forward').dt.tz_convert('UTC').values
    out['kickoff'] = pd.to_datetime(out['kickoff'], utc=True, format='ISO8601')
    out['completed'] = out.away_score.notna() & out.home_score.notna()
    return out.sort_values('kickoff').reset_index(drop=True)


def fetch_schedule() -> pd.DataFrame:
    return schedule_frame(pd.read_csv(io.BytesIO(_get(SCHEDULE_URL)), low_memory=False))


def game_summaries(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per team-game additive stats from play-by-play (same definitions as the history files)."""
    df = pbp[pbp['season_type'].isin(['REG', 'POST'])].copy()
    if df.empty:
        return pd.DataFrame()
    for c in ('posteam', 'defteam', 'fumbled_1_team', 'fumbled_2_team', 'fumble_recovery_1_team', 'fumble_recovery_2_team'):
        if c in df:
            df[c] = df[c].map(lambda v: nfl_code(v) if pd.notna(v) else None)
    for c in ('yards_gained', 'epa', 'pass_attempt', 'interception', 'sack', 'cpoe', 'qb_dropback'):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    ok = df['play_type'].isin(['run', 'pass']) & df['posteam'].notna() & df['defteam'].notna()
    for flag in ('no_play', 'qb_kneel', 'qb_spike'):
        if flag in df:
            ok &= pd.to_numeric(df[flag], errors='coerce').fillna(0).eq(0)
    p = df[ok].copy()
    passing = p['pass_attempt'].eq(1)
    dropback = p['qb_dropback'].eq(1) if 'qb_dropback' in p else passing
    rushing = p['play_type'].eq('run') & ~dropback
    p['plays'] = 1
    p['yards'] = p['yards_gained'].fillna(0)
    p['epa_sum'] = p['epa'].fillna(0)
    p['successes'] = p['epa'].gt(0).astype(int)
    p['pass_plays'] = passing.astype(int)
    p['explosive_passes'] = (passing & p['yards_gained'].ge(20)).astype(int)
    p['dropbacks'] = dropback.astype(int)
    p['pass_epa_sum'] = p['epa'].where(dropback, 0).fillna(0)
    p['rush_plays'] = rushing.astype(int)
    p['rush_epa_sum'] = p['epa'].where(rushing, 0).fillna(0)
    p['sacks'] = p['sack'].eq(1).astype(int) if 'sack' in p else 0
    p['invalid_core_plays'] = (p['yards_gained'].isna() | p['epa'].isna()).astype(int)
    cols = ['plays', 'yards', 'epa_sum', 'successes', 'pass_plays', 'explosive_passes', 'dropbacks',
            'pass_epa_sum', 'rush_plays', 'rush_epa_sum', 'sacks', 'invalid_core_plays']
    grouped = p.groupby(['game_id', 'posteam'])[cols].sum().reset_index().rename(columns={'game_id': 'id', 'posteam': 'team'})
    to = defaultdict(float)
    ints = df[df['interception'].eq(1)]
    for (gid, t), n in ints.groupby(['game_id', 'posteam']).size().items():
        to[(gid, t)] -= n
    for (gid, t), n in ints.groupby(['game_id', 'defteam']).size().items():
        to[(gid, t)] += n
    for i in (1, 2):
        a, b = f'fumbled_{i}_team', f'fumble_recovery_{i}_team'
        if a in df and b in df:
            lost = df[df[a].notna() & df[b].notna() & df[a].ne(df[b])]
            for (gid, t), n in lost.groupby(['game_id', a]).size().items():
                to[(gid, t)] -= n
            for (gid, t), n in lost.groupby(['game_id', b]).size().items():
                to[(gid, t)] += n
    grouped['net_turnovers'] = [to[(i, t)] for i, t in zip(grouped['id'], grouped['team'])]
    # invalid plays are rare; keep games but drop the bad plays' influence by zeroing the flag
    grouped['invalid_core_plays'] = 0
    return grouped


def fetch_pbp(season: int) -> pd.DataFrame:
    raw = _get(PBP_URL.format(season=season), timeout=180)
    return pd.read_csv(io.BytesIO(raw), compression='gzip', usecols=lambda c: c in PBP_COLUMNS, low_memory=False)


def fetch_pbp_summaries(season: int) -> pd.DataFrame:
    return game_summaries(fetch_pbp(season))
