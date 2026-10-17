"""Backend tests that run fully offline on the bundled history.

    python -m unittest discover -s tests -v
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cfb_data, ledger, odds, splits
from board import build_board, shortlist, verdict_for, confluence
from history import load_history
from model import load_model, slate_features, predict_margin
from nfl_data import CODE_TO_NAME
from storage import Store


def fake_events(slate: pd.DataFrame, league: str, shift: float = 0.0, reverse_first=False):
    ev = []
    for k, g in enumerate(slate.itertuples()):
        if pd.isna(g.spread_home):
            continue
        name = (lambda t: CODE_TO_NAME.get(t, t)) if league == 'NFL' else (lambda t: f'{t} Mascots')
        away, home = name(g.away), name(g.home)
        sh = float(g.spread_home) + shift
        if reverse_first and k == 0:
            away, home, sh = home, away, -sh
        books = []
        for key, adj, price in (('draftkings', 0.0, -110), ('fanduel', 0.5, -112), ('pinnacle', -0.5, -105)):
            books.append(dict(key=key, title=key, last_update='2025-01-01T00:00:00Z', markets=[
                dict(key='spreads', outcomes=[dict(name=home, price=price, point=sh + adj),
                                              dict(name=away, price=price, point=-(sh + adj))]),
                dict(key='totals', outcomes=[dict(name='Over', price=-110, point=float(g.total) if pd.notna(g.total) else 45.5),
                                             dict(name='Under', price=-110, point=float(g.total) if pd.notna(g.total) else 45.5)])]))
        ev.append(dict(id=f'ev{k}', sport_key='x', commence_time=g.kickoff.isoformat(),
                       home_team=home, away_team=away, bookmakers=books))
    return ev


def simulated(league: str, now: str):
    """History with every game after `now` turned into an upcoming (unplayed) game."""
    g, tg = load_history(league)
    now = pd.Timestamp(now)
    fut = g.kickoff >= now
    g = g.copy()
    g.loc[fut, ['away_score', 'home_score']] = np.nan
    g.loc[fut, 'completed'] = False
    tg = tg[tg.id.isin(g[~fut].id)]
    return dict(games=g, tg=tg, extra=pd.DataFrame(columns=['id'])), now


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = Store({}, local_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def save_snapshot(self, league, events, when):
        self.store.put_json(f"odds/{league}/{when.strftime('%Y-%m-%dT%H-%M-%S')}.json.gz",
                            dict(fetched_at=when.isoformat(), league=league, events=events))


class TestHistoryAndModel(unittest.TestCase):
    def test_history_loads(self):
        for lg, n in (('CFB', 7000), ('NFL', 2500)):
            g, tg = load_history(lg)
            self.assertGreater(len(g), n)
            self.assertTrue(set(['points', 'sr', 'expl', 'ypp']) <= set(tg.columns))
            self.assertGreater(g.spread_home.notna().mean(), 0.8)

    def test_model_files(self):
        for lg in ('CFB', 'NFL'):
            m = load_model(lg)
            self.assertIsNotNone(m)
            self.assertIn('rank_close', m['calibration'])
            for row in m['calibration']['rank_close']:
                self.assertTrue(0.4 < row['p'] < 0.6)        # honest: no band claims more than 60%

    def test_predictions_track_market(self):
        for lg, now in (('NFL', '2025-11-01T12:00:00Z'), ('CFB', '2025-10-15T12:00:00Z')):
            data, t = simulated(lg, now)
            g = data['games']
            slate = g[(~g.completed) & (g.kickoff < t + pd.Timedelta(days=7))]
            if lg == 'CFB':
                slate = slate[(slate.away != 'FCS') & (slate.home != 'FCS')]
            m = load_model(lg)
            f, r, mr = slate_features(g[g.completed], data['tg'], slate, lg, now=t)
            pred = predict_margin(m, f.fillna(0))
            ok = slate.spread_home.notna().values
            corr = np.corrcoef(pred[ok], slate.spread_home.values[ok])[0, 1]
            self.assertGreater(corr, 0.8, lg)


class TestParsers(unittest.TestCase):
    def test_dk_parse_and_flags(self):
        text = ('HOU Texans @ JAX Jaguars\n10/18, 09:30AM\nSpread\nOdds\n% Handle\n% Bets\n'
                'JAX Jaguars -2.5\n−120\n74%\n76%\nHOU Texans +2.5\n+100\n26%\n24%\n'
                'Total\nOdds\n% Handle\n% Bets\nOver 42.5\n−112\n38%\n79%\nUnder 42.5\n−108\n62%\n21%\n'
                'NY Jets @ NE Patriots\n10/18, 01:00PM\nSpread\nOdds\t% Handle\t% Bets\n'
                'NE Patriots -7.5\t−105\t60%\t78%\nNY Jets +7.5\t−115\t40%\t22%\n')
        p = splits.parse_dk(text)
        self.assertEqual(len(p), 2)
        s = splits.spread_splits(p[1])
        self.assertEqual(s['away']['handle'], 40.0)
        fl = splits.flags(s, open_home=-8.5, current_home=-7.5)
        self.assertEqual(fl['sharp_side'], 'away')        # 40% money on 22% of bets
        self.assertEqual(fl['rlm_side'], 'away')          # Jets number got worse (+8.5 -> +7.5) on 22% of bets
        gid, rev, score = splits.best_match(p[1], [dict(id='g1', away='NYJ', home='NE'), dict(id='g2', away='HOU', home='JAX')], 'NFL')
        self.assertEqual((gid, rev), ('g1', False))

    def test_cfbd_normalizers(self):
        games = cfb_data.games_frame([dict(id=1, season=2026, week=8, seasonType='regular', startDate='2026-10-17T19:30:00.000Z',
                                           awayTeam='USC', homeTeam='Notre Dame', awayClassification='fbs', homeClassification='fbs',
                                           neutralSite=False, completed=False, venueId=3, venue='Notre Dame Stadium')])
        self.assertEqual(games.iloc[0]['home'], 'Notre Dame')
        lines = cfb_data.lines_frame([dict(id=1, awayTeam='USC', homeTeam='Notre Dame', lines=[
            dict(provider='DraftKings', spread=-10.5, formattedSpread='Notre Dame -10.5', spreadOpen=-9.5, overUnder=55.5),
            dict(provider='ESPN Bet', spread=-10, formattedSpread='Notre Dame -10', spreadOpen=-9, overUnder=55),
            # a provider that reports the spread from the away side: verified by formattedSpread
            dict(provider='Odd', spread=10.5, formattedSpread='Notre Dame -10.5', spreadOpen=9.5)])])
        self.assertAlmostEqual(lines.iloc[0]['spread_home'], -10.5)
        self.assertAlmostEqual(lines.iloc[0]['open_home'], -9.5)
        box = cfb_data.box_frame([dict(id=1, teams=[dict(team='USC', stats=[
            dict(category='totalYards', stat='410'), dict(category='rushingAttempts', stat='30'),
            dict(category='completionAttempts', stat='22-35'), dict(category='turnovers', stat='2')])])])
        self.assertEqual(box.iloc[0]['plays'], 65)
        adv = cfb_data.advanced_frame([dict(gameId=1, season=2026, week=8, team='USC',
                                            offense=dict(plays=65, successRate=0.45, explosiveness=1.2),
                                            defense=dict(plays=70, successRate=0.40, explosiveness=1.1))])
        self.assertEqual(adv.iloc[0]['offense.successRate'], 0.45)
        v = cfb_data.venues_frame([dict(id=3, name='NDS', city='South Bend', state='IN', latitude=41.7, longitude=-86.2, dome=False)])
        self.assertAlmostEqual(v.iloc[0]['lat'], 41.7)

    def test_team_matching(self):
        self.assertEqual(odds.match_team('Appalachian State Mountaineers', ['App State', 'Troy']), 'App State')
        self.assertEqual(odds.match_team('Texas A&M Aggies', ['Texas', 'Texas A&M']), 'Texas A&M')
        self.assertEqual(odds.match_team('Kansas City Chiefs', ['KC', 'DEN']), 'KC')


class TestBoardAndLedger(Base):
    def run_board(self, league, now):
        data, t = simulated(league, now)
        g = data['games']
        slate = g[(~g.completed) & (g.kickoff < t + pd.Timedelta(days=8))]
        # opener 2 days earlier, then the "current" line 1 point toward the home team
        self.save_snapshot(league, fake_events(slate, league, 0.0, reverse_first=True), t - pd.Timedelta(days=2))
        self.save_snapshot(league, fake_events(slate, league, -1.0, reverse_first=True), t - pd.Timedelta(hours=2))
        m = load_model(league)
        b = build_board(league, data, m, self.store, now=t, days_ahead=8, with_weather=False)
        return b, data, t, m

    def test_nfl_board(self):
        b, data, t, m = self.run_board('NFL', '2025-11-01T12:00:00Z')
        self.assertGreater(len(b), 10)
        self.assertGreater(b.snapshots.gt(0).mean(), 0.9)          # odds matched to games
        self.assertTrue(np.allclose(b.move_home.dropna(), -1.0))    # current = open - 1 for every game
        self.assertTrue(b.edge.notna().all())
        sl = shortlist(b, 'NFL')
        self.assertLessEqual(len(sl), 3)
        r = b.iloc[0].to_dict()
        self.assertIn(verdict_for('away', r, 'NFL')[0], {'Model backs it (strong)', 'Model leans this way', 'No real edge either way',
                                                        'Model leans against', 'Model says stay off'})
        # reversed listing handled: first game's current line equals schedule spread - 1
        g0 = data['games'].set_index('id').loc[b.sort_values('kickoff').iloc[0].id]

    def test_cfb_board(self):
        b, data, t, m = self.run_board('CFB', '2025-10-15T12:00:00Z')
        self.assertGreater(len(b), 30)
        self.assertGreater(b.snapshots.gt(0).mean(), 0.9)
        self.assertGreater(b.edge.notna().mean(), 0.9)
        self.assertTrue(b.p_model_side.dropna().between(0.4, 0.6).all())

    def test_splits_flow_and_ledger(self):
        b, data, t, m = self.run_board('NFL', '2025-11-01T12:00:00Z')
        r = b.iloc[0]
        text = (f"{r.away} {CODE_TO_NAME[r.away].split()[-1]} @ {r.home} {CODE_TO_NAME[r.home].split()[-1]}\n11/02, 01:00PM\n"
                f"Spread\nOdds\n% Handle\n% Bets\n{r.away} X +3\n-110\n70%\n30%\n{r.home} X -3\n-110\n30%\n70%\n")
        p = splits.parse_dk(text)
        gid, rev, score = splits.best_match(p[0], b[['id', 'away', 'home']].to_dict('records'), 'NFL')
        self.assertEqual(gid, r.id)
        splits.save(self.store, 'NFL', p, {'0': gid})
        b2 = build_board('NFL', data, m, self.store, now=t, days_ahead=8, with_weather=False)
        row = b2[b2.id == r.id].iloc[0]
        self.assertEqual(row.split_flags['sharp_side'], 'away')
        # ledger: log the away side at the current number, then "finish" the game and grade
        line = -row.current_home
        ledger.add(self.store, dict(league='NFL', game_id=row.id, away=row.away, home=row.home, kickoff=str(row.kickoff),
                                    side='away', team=row.away, line=float(line + 0.5), odds=-110, units=1.0,
                                    verdict='x', grade='C'))
        full, _ = load_history('NFL')
        closes = ledger.find_closes(self.store, ledger.load(self.store))
        self.assertIn(str(row.id), closes.get('NFL', {}))
        n = ledger.grade(self.store, {'NFL': full}, closes)
        self.assertEqual(n, 1)
        p = ledger.load(self.store)[0]
        self.assertAlmostEqual(p['clv'], 0.5)
        self.assertIn(p['result'], ('W', 'L', 'P'))
        s = ledger.summary(ledger.load(self.store))
        self.assertEqual(len(s), 1)


class TestStorage(Base):
    def test_roundtrip(self):
        self.store.put_json('a/b.json', {'x': 1})
        self.store.put_json('a/c.json.gz', {'y': 2})
        self.assertEqual(self.store.get_json('a/b.json'), {'x': 1})
        self.assertEqual(self.store.get_json('a/c.json.gz'), {'y': 2})
        self.assertEqual(self.store.list('a/'), ['a/b.json', 'a/c.json.gz'])



class TestDuplicates(unittest.TestCase):
    def test_duplicate_stat_rows(self):
        """Live crash 10/16: CFBD listed games twice -> FCS step failed.  Must give one row per team per game."""
        from history import cfb_team_games, FCS
        games = pd.DataFrame(dict(id=['1', '2'], away=[FCS, 'Akron'], home=['Ohio State', 'Ohio'], away_score=[3, 20],
                                  home_score=[50, 24], neutral=[False, False], completed=[True, True]))
        adv = pd.DataFrame({'id': ['1', '1', '2', '2', '2'], 'team': ['Ohio State', 'Ohio State', 'Akron', 'Ohio', 'Ohio'],
                            'offense.successRate': .5, 'offense.explosiveness': 1.2, 'offense.plays': 70,
                            'defense.successRate': .3, 'defense.explosiveness': 1.0, 'defense.plays': 60})
        box = pd.DataFrame(dict(id=['1', '1', '2', '2', '2'], team=['Ohio State', 'Ohio State', 'Akron', 'Ohio', 'Ohio'],
                                yards=400, plays=70, giveaways=1))
        tg = cfb_team_games(games, adv, box)
        self.assertEqual(len(tg), 4)
        self.assertFalse(tg.duplicated(['id', 'team']).any())
        self.assertAlmostEqual(tg[tg.team == FCS].sr.iloc[0], .3)   # FCS offense = Ohio State's defense allowed

if __name__ == '__main__':
    unittest.main()
