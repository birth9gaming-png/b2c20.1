# Born to Cover 19

A spread reference board for **college football and the NFL**. For every game it shows:

- the model's **fair spread** and how far it is from the sportsbook line (the edge)
- the **opening line, current line and movement**, plus the best number available
- **DraftKings splits** (% of bets vs % of money), downloaded automatically, with bigger-money and reverse-line-move flags
- **lookahead / letdown / sandwich spots**, rest days and kickoff **weather**
- a **verdict for any side you like** ("Model backs it", "No real edge", "Model says stay off")
- a **pick log** that grades results and your **closing-line value** automatically

Nothing needs to be typed in. Set your stipulations once in **My rules** (sidebar sliders: picks per week, minimum edge, biggest spread you'll bet, how many signals must agree, only-when-the-opponent-is-in-a-spot, and more). The shortlist only shows games that pass them. To check one of your own plays, pick the game and side in *Check my pick* and press **Log this pick**.

## What changed from v17.1

v17.1 had 125 files of stacked patches. Its 72% records came from choosing the best of 120+ versions using the same seasons it was graded on. Its "60–65%" estimates were mostly totals, inflated by hand-typed point values. v18 is about 20 code files plus its data files, all in one folder. The model's numbers come from tests where each season was predicted by a model that never saw that season.

## My spots (v19), built from your own betting history

The app learns your spots from your own Pikkit history. It grades every spread pick you made, including parlay legs, each counted once.

**First-time setup:** in *My picks*, upload your Pikkit export (transactions.csv) and press **Import this file**. It's saved to your private Cloudflare storage, not to GitHub (your repo is public). Press **Update now** first so this season's college games are loaded for grading.

**The My spots tab, each week:**
- **Your spots:** about 4 games per league ranked by your own record in similar spots: line size (you're strongest as a 0.5-10 point favorite and a 4-10 point underdog), teams you know (teams you've bet 3+ times) and home/road. College spots only show games with a power-conference team, since those are the games you watch. Teams with a key player out are left off.
- **Each card shows** the model's opinion, the line now vs. the open, injuries, DraftKings splits, lookahead/letdown flags, rivalry and weather. The model is a second opinion and never overrides you.
- **Other games worth a look:** the model's biggest disagreements outside your spots. In college these lean toward Group of 5 games, where the model has done best.
- **Log $25 buttons** record each pick as a $25 straight bet. *My picks* then grades it and tracks closing-line value.

The percentage on each card is **your historical hit rate in spots like that one**, pulled toward your overall record. It is not a guarantee, and most spots have only 10-40 past picks behind them.

**Keep feeding it.** In *My picks*, upload a new Pikkit export any time (transactions.csv). The app re-grades everything and relearns your spots. Picks you log in the app are added to your profile automatically.

## How picks are chosen (v18.1)

Every game gets a **Pick Score (0-100)** for the side the model likes:

| Piece | Points |
|---|---|
| Model edge (model line vs market) | up to 50 (CFB 12+ pts or NFL 6+ pts of edge = 50) |
| Each signal agreeing: line moving that way, bigger DraftKings money, reverse line move, opponent in a lookahead/letdown/sandwich spot | +12 each |
| Same signals on the other team | -12 each |
| Key player out (main QB, top RB, top two receivers) or QB questionable, on that side | -25 |

Grades: A 70+, B 55-69, C 40-54, D under 40. The shortlist is the highest scores that pass **My rules**, and only C or better by default. Lower games show as "leans", never as picks. The weights are our own rules, not fitted to results. The pick log tracks each grade's record so we can tune them.

**Injuries are automatic.** NFL reports come from ESPN and college reports from Covers.com. Key players come from this season's stats.
- **NFL starting QB out:** the model line moves 2.0 points against that team. That figure comes from our own data: across 668 starts by a non-regular QB from 2016 to 2026, those teams played 2.0 points worse than the stats model expected, while the closing line had already priced it in.
- **Other key players out, or a college QB out:** flagged, and by default kept off the shortlist. There's no history to put a point value on them yet.

## Honest performance (walk-forward 2018–2025)

| | vs closing line | vs opening line (CFB) |
|---|---|---|
| Every game | ~50% | ~50% |
| Top 2 disagreements per week, CFB | 52% | **58% (117-85), line moved +0.9 pts toward the model** |
| Top 2 disagreements per week, NFL | 54% | no opener history yet |

**The model's information gets priced in during the week.** Its best use is betting early, Sunday night and Monday, when it disagrees with the opener. At game-day lines, use it as a fair-number check and a veto. The NFL "model is against this side by 4.5+ points" spots covered only 45.7%.

## Setup

Every file lives in one folder, so GitHub's normal upload works.

1. **Upload.** In your repo, go to *Add file → Upload files*, drag in **every file from this zip**, and commit. Same names replace the old ones. If the repo has a leftover `btc` folder or an `__init__.py`, it does no harm.
2. **Streamlit.** Main file is `app.py`. Keep your Secrets (`CFBD_API_KEY`, `ODDS_API_KEY`, `R2_*`); the format is in `secrets.example.toml`. If the app was already deployed, choose *Manage app → Reboot app*.
3. **Optional: automatic line snapshots.** These save openers and closing lines 9 times a week even when the app is closed (about 155 of your 500 monthly Odds API credits).
   - In GitHub, go to *Add file → Create new file* and type the name `.github/workflows/snapshots.yml`. Paste the contents of `snapshots.yml` from this zip and commit.
   - Then go to *Settings → Secrets and variables → Actions* and add the same six keys as repository secrets.
4. Open the app. It updates stats, lines and DraftKings splits by itself when they're stale. **Update now** forces an update.

## Weekly rhythm

- **Sunday night / Monday:** check *This week*. The top cards are the biggest disagreements with the openers, which is where the model has historically been worth the most.
- **Game day:** use *Check my pick* on your own plays. If the model says "stay off", get a better number or pass.
- **After the games:** *My picks* grades everything and shows average CLV. Positive CLV over 100+ bets is the clearest proof that your picks have an edge.

## Research behind it

- **Opponent-adjusted efficiency ratings**, the approach behind Massey-Peabody, which reported about 56% ATS over 500+ public picks.
- **Market-implied team ratings** taken from earlier closing lines.
- **Line timing.** Our model's top picks beat openers far more often than closing lines, and lines tend to overreact (Simon, *Management Science* 2024).
- **Tested and shown with their real records:** lookahead and letdown spots, power conference vs Group of 5, and home underdogs. None beat the closing line on their own. Reverse line movement and splits can't be tested yet because there's no free history of splits. The app saves every download, so your own pick log will answer it.

## Data sources and limits

- **NFL:** nflverse schedule, scores, closing lines and play-by-play. Free, no key.
- **CFB:** CollegeFootballData API, roughly 4 calls per update (free tier: 1,000 per month). Opening lines come from CFBD.
- **Lines:** The Odds API, 2 credits per league per update. One request covers DraftKings, FanDuel, BetMGM, Caesars, Fanatics, theScore (ESPN Bet), BetRivers, Pinnacle, LowVig and BetOnline.
- **Splits:** the DraftKings Network splits page, with VSiN's copy of the same DraftKings data as a fallback. Both sites' robots.txt allow it, and the app makes only a few requests per update. If DraftKings changes its page layout, the *DraftKings splits* tab has a hidden paste box as a backup.
- **Weather:** Open-Meteo, free.

## Retraining

```bash
python train.py                  # rebuilds model_CFB.json and model_NFL.json
python -m unittest test_backend
```

Retrain in the off-season after adding the finished season to the history files.
