"""Tests for the Pikkit import, your pick profile and the My Spots ranking (offline)."""
import shutil, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mypicks
from board import build_board
from model import load_model
from storage import Store
from test_backend import simulated, fake_events

CSV = b'''bet_id,sportsbook,type,status,odds,closing_line,ev,amount,profit,time_placed,time_settled,time_placed_iso,time_settled_iso,bet_info,tags,sports,leagues
1,Draftkings Sportsbook,parlay,SETTLED_LOSS,3.5,,,25,-25,x,x,2025-11-01T14:00:00.000Z,2025-11-02T03:00:00.000Z,Notre Dame -6.5 Spread Navy @ Notre Dame | KC Chiefs +2.5 Spread KC Chiefs @ BUF Bills,,American Football,NCAAFB | NFL
2,Fanduel Sportsbook,straight,SETTLED_WIN,1.91,,,25,22.7,x,x,2025-11-01T14:00:00.000Z,2025-11-02T03:00:00.000Z,Notre Dame -6.5 Spread Navy @ Notre Dame,,American Football,NCAAFB
3,Fanduel Sportsbook,straight,SETTLED_LOSS,1.91,,,25,-25,x,x,2025-11-01T14:00:00.000Z,2025-11-02T03:00:00.000Z,Navy +20.5 1st Half Spread Navy @ Notre Dame,,American Football,NCAAFB
4,Fanduel Sportsbook,straight,SETTLED_LOSS,5.0,,,10,-10,x,x,2025-11-01T14:00:00.000Z,2025-11-02T03:00:00.000Z,Jo Adell To Hit A Home Run Rays @ Guardians,,Baseball,MLB
'''


def tiny_games():
    t = pd.Timestamp('2025-11-01T20:00:00Z')
    cfb = pd.DataFrame([dict(id='g1', season=2025, week=10, away='Navy', home='Notre Dame', kickoff=t, neutral=False,
                             completed=True, away_score=10, home_score=31)])
    nfl = pd.DataFrame([dict(id='n1', season=2025, week=9, away='KC', home='BUF', kickoff=t + pd.Timedelta(days=1), neutral=False,
                             completed=True, away_score=20, home_score=24)])
    return {'CFB': cfb, 'NFL': nfl}


class TestImport(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(mypicks.bucket(-6.5), 'Favorite 4-10')
        self.assertEqual(mypicks.bucket(-10.5), 'Favorite 10.5-21')
        self.assertEqual(mypicks.bucket(-3.5), 'Favorite 0.5-3.5')
        self.assertEqual(mypicks.bucket(3.5), 'Underdog 0.5-3.5')
        self.assertEqual(mypicks.bucket(10.5), 'Underdog 10.5+')
        self.assertEqual(mypicks.bucket(-21), 'Favorite 21+')

    def test_parse_and_grade(self):
        self.assertIsNone(mypicks.parse_leg('Navy +20.5 1st Half Spread Navy @ Notre Dame'))
        p = mypicks.parse_leg('Notre Dame -6.5 Spread Navy @ Notre Dame')
        self.assertEqual((p['team'], p['line'], p['kind']), ('Notre Dame', -6.5, 'spread'))
        picks, summ = mypicks.import_pikkit(CSV, tiny_games())
        self.assertEqual(len(picks), 2)                      # duplicate ND pick counted once; 1st half + MLB skipped
        nd = picks[picks.team == 'Notre Dame'].iloc[0]
        self.assertEqual((nd.side, nd.result), ('home', 'W'))  # won by 21, laid 6.5
        kc = picks[picks.team == 'KC'].iloc[0]
        self.assertEqual(kc.result, 'L')                      # lost by 4 getting 2.5
        self.assertEqual(summ['bets'], 4)

    @unittest.skipUnless(mypicks.HISTORY_FILE.exists(), 'personal history file not present')
    def test_bundled_profile(self):
        h = mypicks.load_history()
        prof = mypicks.build_profile(h)
        self.assertGreater(prof['n'], 150)
        self.assertGreater(prof['overall']['rate'], 0.55)
        rate, why = mypicks.spot(prof, 'CFB', 'Notre Dame', -6.5, 'home')
        self.assertGreater(rate, prof['overall']['rate'])     # best bucket + team you know
        self.assertTrue(any('favorites of 4-10' in w for w in why))
        rate2, _ = mypicks.spot(prof, 'CFB', 'Some Team', -14, 'home')
        self.assertLess(rate2, rate)                          # your weak bucket


class TestRanking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.store = Store({}, local_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @unittest.skipUnless(mypicks.HISTORY_FILE.exists(), 'personal history file not present')
    def test_rank_spots(self):
        for lg, now in (('CFB', '2025-10-15T12:00:00Z'), ('NFL', '2025-11-01T12:00:00Z')):
            data, t = simulated(lg, now)
            g = data['games']
            slate = g[(~g.completed) & (g.kickoff < t + pd.Timedelta(days=8))]
            when = t - pd.Timedelta(hours=2)
            self.store.put_json(f"odds/{lg}/{when.strftime('%Y-%m-%dT%H-%M-%S')}.json.gz",
                                dict(fetched_at=when.isoformat(), league=lg, events=fake_events(slate, lg)))
            b = build_board(lg, data, load_model(lg), self.store, now=t, with_weather=False)
            prof = mypicks.build_profile(mypicks.load_history())
            r = mypicks.rank_spots(b, prof, lg)
            self.assertGreater(len(r), 5)
            self.assertIn('qualifies', r)
            q = r[r.qualifies]
            self.assertTrue(q.for_reasons.map(len).gt(0).all())        # every listed game has a real reason
            self.assertTrue((q.moved_against < 3).all())
            top = r.iloc[0]
            self.assertIn(top.side, ('away', 'home'))
            exp_line = -b.set_index('id').loc[top.id, 'current_home'] if top.side == 'away' else b.set_index('id').loc[top.id, 'current_home']
            self.assertAlmostEqual(top.line, exp_line)
            if lg == 'CFB':
                self.assertIn(True, set(r.power)); self.assertIn(False, set(r.power))


if __name__ == '__main__':
    unittest.main()
