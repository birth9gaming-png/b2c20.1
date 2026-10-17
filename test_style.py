"""Tests for the 'your style' copy-of-you model (offline, simulated games)."""
import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style
from test_backend import simulated


def fake_picks(games, league, n_weeks=8):
    """A bettor who always takes home favorites of 3-10: moneyline (CFB) or alt spread, else the spread."""
    done = games[games.completed & games.spread_home.notna()]
    weeks = sorted(set(zip(done.season, done.week)))[-n_weeks:]
    rows = []
    for s, w in weeks:
        g = done[(done.season == s) & (done.week == w)]
        for r in g.itertuples():
            if -10 <= r.spread_home <= -3:
                kind = 'moneyline' if league == 'CFB' else 'alt spread'
                line = np.nan if kind == 'moneyline' else r.spread_home + 2
                margin = r.home_score - r.away_score
                res = 'W' if (margin > 0 if kind == 'moneyline' else margin + line > 0) else 'L'
                rows.append(dict(placed=r.kickoff - pd.Timedelta(hours=3), league=league, season=s, week=w, game_id=r.id,
                                 away=r.away, home=r.home, team=r.home, side='home', kind=kind, line=line,
                                 neutral=False, bet_type='parlay', result=res))
    return pd.DataFrame(rows)


class TestStyle(unittest.TestCase):
    def test_learns_and_suggests(self):
        for lg, now in (('NFL', '2025-11-01T12:00:00Z'), ('CFB', '2025-10-15T12:00:00Z')):
            data, t = simulated(lg, now)
            g = data['games']
            picks = fake_picks(g, lg)
            self.assertGreater(len(picks), style.MIN_PICKS)
            m = style.train(picks, g, lg)
            self.assertIsNotNone(m)
            coef = dict(zip(m['feats'], m['coef']))
            self.assertGreater(coef['home'], 0)                      # learned "home teams"
            self.assertGreater(coef['b2'], 0)                        # learned "favorites of 3-7"
            leg = style.how_you_play(m, -5.0, lg)
            self.assertEqual(leg['kind'], 'moneyline' if lg == 'CFB' else 'spread')
            if lg == 'NFL':
                self.assertNotIn('moneyline', m['forms']['2']['counts'])
            # this week's board: upcoming games with a current line
            up = g[(~g.completed) & (g.kickoff > t) & (g.kickoff < t + pd.Timedelta(days=8))].copy()
            board = up.assign(current_home=up.spread_home, open_home=up.spread_home + 1, move_home=-1.0, edge=0.0,
                              injuries=[{} for _ in range(len(up))])
            sug = style.suggest(m, board, g, lg, picks=picks, k=3)
            self.assertEqual(int(sug.shortlist.sum()), min(3, int(sug.qualifies.sum())))
            self.assertGreater(int(sug.shortlist.sum()), 0)
            self.assertTrue(sug[sug.shortlist].qualifies.all())               # every pick has a real reason
            self.assertTrue(all(any('line moved toward them' in x for x in rs) for rs in sug[sug.shortlist].for_reasons))
            # no reasons anywhere -> no picks, even if the style fits
            flat = board.assign(open_home=board.current_home, move_home=0.0)
            self.assertEqual(int(style.suggest(m, flat, g, lg, picks=picks, k=3).shortlist.sum()), 0)
            self.assertTrue(sug['raw'].is_monotonic_decreasing)
            top = sug[sug.shortlist].iloc[0]
            self.assertTrue(0 <= top['style'] <= 100)
            rep = style.replay(m, g, lg, sorted(g.season.unique())[:-1], k=3)
            self.assertGreater(sum(rep['legs']), 0)
            fb = [dict(league=lg, id=str(top['id']), side=top['side'], team=top['team'], vote=-1, feats=top['feats'],
                       season=int(top['season']), week=int(top['week']))]
            sug2 = style.suggest(m, board, g, lg, picks=picks, k=3, feedback=fb)
            self.assertNotIn((str(top['id']), top['side']), set(zip(sug2[sug2.shortlist].id.astype(str), sug2[sug2.shortlist].side)))
            m2 = style.train(picks, g, lg, feedback=fb)
            self.assertEqual(m2['n_feedback'], 1)
            self.assertLess(style.score(m2, pd.DataFrame([top['feats']]))[0], style.score(m, pd.DataFrame([top['feats']]))[0])

    def test_your_leg_rules(self):
        leg = lambda l, lg='CFB': style.leg_text('T', style.how_you_play(None, l, lg))
        self.assertEqual(leg(-7.5), 'T -6.5 (alt)')          # 7.5 -> 6.5
        self.assertEqual(leg(-10.5), 'T -9.5 (alt)')         # 10.5 -> 9.5
        self.assertEqual(leg(-14.5), 'T -13.5 (alt)')        # 14.5 -> 13.5
        self.assertEqual(leg(-21.5), 'T -20.5 (alt)')        # 21.5 -> 20.5
        self.assertEqual(leg(-9.5), 'T -9.5')                # not on a key number, and ML would be ~-345
        self.assertTrue(leg(-3).startswith('T ML'))          # small college favorite: moneyline, about -157
        for l in (-7, -9.5, -14, -24):                       # never a leg worse than -250
            self.assertNotEqual(style.how_you_play(None, l, 'CFB')['kind'], 'moneyline')
        self.assertEqual(leg(-3.5, 'NFL'), 'T -2.5 (alt)')
        self.assertEqual(leg(-7, 'NFL'), 'T -6.5 (alt)')
        self.assertEqual(style.how_you_play(None, -1.5, 'NFL')['kind'], 'spread')   # no NFL moneylines
        self.assertLess(abs(style.ml_price(-7, 'CFB') + 255), 20)

    def test_leg_text_and_grading(self):
        self.assertEqual(style.leg_text('Alabama', dict(kind='moneyline', leg_line=None)), 'Alabama ML')
        self.assertEqual(style.leg_text('KC', dict(kind='alt spread', leg_line=-2.5)), 'KC -2.5 (alt)')
        self.assertEqual(style.grade_leg(dict(kind='alt spread', leg_line=-2.5), 3), 'W')
        self.assertEqual(style.grade_leg(dict(kind='spread', leg_line=-7.5), 7), 'L')
        self.assertEqual(style.grade_leg(dict(kind='moneyline', leg_line=None), 1), 'W')
        self.assertEqual(style.bucket_name(-5), 'favorite 3-7')
        self.assertEqual(style.bucket_name(10), 'underdog 7+')


if __name__ == '__main__':
    unittest.main()
