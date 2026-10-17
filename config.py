"""Settings shared across the app.  Change values here, not inside modules."""

VERSION = '20.2'

LEAGUES = ('CFB', 'NFL')

ODDS_SPORT = {'CFB': 'americanfootball_ncaaf', 'NFL': 'americanfootball_nfl'}

# Up to 10 bookmakers in one request count as ONE region for Odds API credits.
# Keys checked against The Odds API bookmaker list (Oct 2026).  williamhill_us = Caesars,
# espnbet = theScore Bet (formerly ESPN Bet).  Pinnacle / LowVig / BetOnline are the sharper books.
BOOKMAKERS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'fanatics', 'espnbet', 'betrivers',
              'pinnacle', 'lowvig', 'betonlineag']
SHARP_BOOKS = ['pinnacle', 'lowvig', 'betonlineag']

# Odds API credits: markets x regions per call.  spreads + totals = 2 credits per league call.
ODDS_MARKETS = 'spreads,totals'

# Verdict bands for a side's model cover chance (after any manual adjustment)
VERDICT_BANDS = [
    (0.555, 'Model backs it (strong)'),
    (0.525, 'Model leans this way'),
    (0.475, 'No real edge either way'),
    (0.445, 'Model leans against'),
    (0.0, 'Model says stay off'),
]

# Points of edge shown as "big disagreement" (model line vs current line)
BIG_EDGE = {'CFB': 1.5, 'NFL': 1.0}     # Vegas-anchored model: a proven spot
LEAN_EDGE = {'CFB': 1.0, 'NFL': 0.75}

# DraftKings splits: handle% minus bets% on a side that signals bigger (sharper) tickets
SHARP_GAP = 10.0

# Shortlist size per league per week
SHORTLIST = 3

# Weather flags
WIND_FLAG_MPH = 15.0
COLD_FLAG_F = 32.0

# Key numbers for NFL spreads (crossing these matters most)
NFL_KEY_NUMBERS = (3, 7, 10, 6, 4, 14)

# Starting QB out: measured on our own data (NFL 2016-2026, 668 starts by a non-regular QB): those teams played
# 2.0 points worse than the stats model expected, while the closing line had already priced it (error -0.01).
# No equivalent CFB history is available, so CFB QB injuries are flagged but not given a point value.
QB_OUT_POINTS = {'NFL': 2.0, 'CFB': 0.0}
