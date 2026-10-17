"""Born to Cover 19 — your spots + spread reference board for CFB and NFL.

Run:  streamlit run app.py
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import config, ledger, mypicks, splits as splits_mod, spots as spots_mod, injuries as inj_mod
import style
from board import (build_board, shortlist, verdict_for, confluence, grade, team_line, load_rules, save_rules,
                   passes_rules, pick_score, grade_from_score, leans)
from data import league_data, refreshed_at, current_season, update_all, stale, last_lines
from model import load_model, slate_features, predict_margin
from odds import OddsAPI, load_snapshots, line_history
from ratings import team_table, fit_market_ratings
from storage import Store

ET = ZoneInfo('America/New_York')
st.set_page_config(page_title='Born to Cover 19', page_icon='🏈', layout='wide')


# ------------------------------------------------------------------ setup
def secrets() -> dict:
    try:
        return {k: st.secrets[k] for k in st.secrets}
    except Exception:
        return {}


@st.cache_resource
def get_store():
    return Store(secrets())


SEC = secrets()
store = get_store()


def fav_text(away, home, home_spread):
    if home_spread is None or pd.isna(home_spread):
        return '—'
    if abs(home_spread) < 0.01:
        return 'PK'
    return f'{home} {home_spread:g}' if home_spread < 0 else f'{away} {-home_spread:g}'


def et(ts):
    return pd.Timestamp(ts).tz_convert(ET).strftime('%a %m/%d %I:%M %p')


def pct(x):
    return '—' if x is None or pd.isna(x) else f'{100 * x:.1f}%'


def data_stamp(league):
    snaps = store.list(f'odds/{league}/')
    spl = store.list(f'splits/{league}/')
    return f"{refreshed_at(store, league)}|{len(snaps)}|{snaps[-1] if snaps else ''}|{len(spl)}"


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def cached_data(league, stamp):
    return league_data(league, store)


@st.cache_data(show_spinner=False, ttl=3 * 3600)
def cached_board(league, stamp, days, with_weather):
    model = load_model(league, store)
    data = cached_data(league, stamp)
    return build_board(league, data, model, store, days_ahead=days, with_weather=with_weather)


# ------------------------------------------------------------------ sidebar
st.sidebar.title('🏈 Born to Cover 20')
league = st.sidebar.radio('League', config.LEAGUES, horizontal=True)
days = st.sidebar.slider('Days ahead', 2, 10, 6)
with_weather = st.sidebar.checkbox('Kickoff weather forecast', value=True)
model = load_model(league, store)

auto = st.sidebar.checkbox('Update automatically', value=True,
                            help='On open: stats if older than 12 h, DraftKings splits if older than 1 h, injury reports if older '
                                 'than 2 h, sportsbook lines if older than 12 h (2 Odds API credits each time).')


def run_update(**parts):
    with st.spinner('Updating stats, lines, DraftKings splits and injury reports…'):
        problems = update_all(store, SEC, league, log=lambda m: None, **parts)
    st.cache_data.clear()
    if problems:
        with st.sidebar.expander(f'⚠️ {len(problems)} update problem(s)', expanded=True):
            for p in problems:
                st.code(p)


due = stale(store, league)
if auto and any(due.values()) and not st.session_state.get(f'auto_done_{league}'):
    st.session_state[f'auto_done_{league}'] = True
    run_update(**due)
if st.sidebar.button('Update now', type='primary', help='Stats & scores, sportsbook lines (2 credits) and DraftKings splits.'):
    run_update()

rules = load_rules(store)
with st.sidebar.expander('My rules (which games make the shortlist)'):
    lg_ = league.lower()
    new = dict(rules)
    new['picks'] = st.slider('Picks per week', 1, 5, int(rules['picks']))
    new[f'min_edge_{lg_}'] = st.slider('Minimum model edge (pts)', 0.0, 3.0, float(rules[f'min_edge_{lg_}']), 0.25)
    new[f'max_spread_{lg_}'] = st.slider('Skip spreads bigger than', 3.0, 40.0, float(rules[f'max_spread_{lg_}']), 0.5)
    new['min_grade'] = st.select_slider('Lowest grade allowed', options=['A', 'B', 'C', 'D'], value=rules.get('min_grade', 'D'))
    new['min_market_signals'] = st.slider('Minimum market/spot signals agreeing (line move, money, RLM, spot)', 0, 4,
                                          int(rules.get('min_market_signals', 0)))
    new['skip_key_injury'] = st.checkbox('Skip if the team the model likes has a key player out',
                                         value=bool(rules.get('skip_key_injury', True)))
    new['require_spot'] = st.checkbox('Only when the opponent is in a lookahead/letdown spot', value=bool(rules['require_spot']))
    new['skip_if_moved_against'] = st.checkbox('Skip if the line moved against the model', value=bool(rules['skip_if_moved_against']))
    new['skip_if_value_gone'] = st.checkbox('Skip if the line already moved to the model', value=bool(rules['skip_if_value_gone']))
    if new != rules:
        save_rules(store, new)
        rules = new

usage = store.get_json('usage/odds_api.json') or {}
month = datetime.now(timezone.utc).strftime('%Y-%m')
cfbd_calls = (store.get_json(f'usage/cfbd_{month}.json') or {}).get('calls', 0)
st.sidebar.caption(
    f"Stats updated: {(refreshed_at(store, league) or 'never — using bundled history')[:16]}  \n"
    f"Lines updated: {(last_lines(store, league) or 'never')[:16]} ({len(store.list(f'odds/{league}/'))} snapshots)  \n"
    f"Splits updated: {(splits_mod.last_capture(store, league) or 'never')[:16]}  \n"
    f"Odds API credits left: {usage.get('x-requests-remaining', '?')} · CFBD calls this month: {cfbd_calls}/1000  \n"
    f"Storage: {store.describe()}  \n"
    f"Model trained {model['trained_at'][:10] if model else '—'} on {model['games'] if model else 0} games (UTC times)")

if not model:
    st.error('No model file found. Upload model_CFB.json and model_NFL.json to the repo (same folder as app.py).')
    st.stop()

stamp = data_stamp(league)
try:
    with st.spinner('Building the board…'):
        board = cached_board(league, stamp, days, with_weather)
except Exception as exc:
    import traceback
    st.error('Could not build the board. Open "Error details" below and send a screenshot.')
    with st.expander('Error details'):
        st.code(f'{type(exc).__name__}: {exc}\n\n' + traceback.format_exc())
    board = pd.DataFrame()

tabs = st.tabs(['My picks', 'All games', 'Check my pick', 'DraftKings splits', 'My history', 'Ratings', 'How it works'])

# ------------------------------------------------------------------ My spots
def my_profile():
    hist = mypicks.load_history(store)
    app_picks = mypicks.ledger_as_history(ledger.load(store))
    allp = pd.concat([hist, app_picks], ignore_index=True) if len(app_picks) else hist
    return mypicks.build_profile(allp) if len(allp) else {}, allp


def log_straight(r, team, side, line, note=''):
    best = r.get('best_away') if side == 'away' else r.get('best_home')
    ledger.add(store, dict(league=league, game_id=r['id'], season=int(r['season']), week=int(r['week']), away=r['away'],
                           home=r['home'], kickoff=str(r['kickoff']), side=side, team=team, line=float(line), odds=-110.0,
                           units=1.0, stake=mypicks.UNIT, book=(best[2] if best is not None and len(best) > 2 else None),
                           model_line_home=float(r['model_home']) if pd.notna(r.get('model_home')) else None,
                           market_home_at_pick=float(r['current_home']) if pd.notna(r.get('current_home')) else None,
                           verdict='my spot', grade=r.get('grade'), score=r.get('score'), notes=note))


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_style(league, stamp, hist_key):
    """The copy of you for one league, retrained whenever your history, your votes or the data change."""
    _, allp = my_profile()
    data = cached_data(league, stamp)
    opens = style.cfb_opens(data.get('extra')) if league == 'CFB' else None
    return style.train(allp, data['games'], league, opens=opens, feedback=style.load_feedback(store))


def style_section(all_picks):
    feedback = style.load_feedback(store)
    hist_key = f"{len(all_picks)}|{all_picks['placed'].max() if len(all_picks) else ''}|{len(feedback)}|{feedback[-1]['at'] if feedback else ''}"
    try:
        with st.spinner('Learning how you pick…'):
            sm = cached_style(league, data_stamp(league), hist_key)
    except Exception as exc:
        import traceback
        st.error(f'Could not build your style model ({type(exc).__name__}).')
        with st.expander('Error details'):
            st.code(traceback.format_exc())
        return
    st.subheader(f'Your style — 3 {league} picks')
    if sm is None:
        st.info(f'Not enough {league} picks in your history to learn your style yet ({style.MIN_PICKS} needed).')
        return
    games = cached_data(league, data_stamp(league))['games']
    try:
        sug = style.suggest(sm, board, games, league, picks=all_picks, k=3, feedback=feedback)
    except Exception as exc:
        import traceback
        st.error(f'Could not rank this week ({type(exc).__name__}).')
        with st.expander('Error details'):
            st.code(traceback.format_exc())
        return
    rep = store.get_json(f'style/{league}_replay.json') or {}
    test_txt = STYLE_TEST.get(league, '')
    if rep.get('n_picks') == sm['n_picks']:
        lw, ll = rep['legs']; aw, al = rep['ats']
        test_txt = (f"Tested on {rep['seasons']} (before you bet): its 3 legs a week went {lw}-{ll} "
                    f"({100 * lw / max(1, lw + ll):.0f}%); the same sides at the plain spread {aw}-{al} "
                    f"({100 * aw / max(1, aw + al):.1f}%, 52.4% breaks even).")
    st.caption(f"Picked ONLY the way you pick: learned from your {sm['n_picks']} {league} picks "
               f"({'spread, alt spread and moneyline legs' if league == 'CFB' else 'spread and alt spread legs; your NFL moneylines are left out'}), "
               f"compared with the {sm['n_sides']} sides you passed on in those weeks. No stats model or betting-market "
               f"data decides anything here. Legs follow your rules: college moneylines only up to about -250, otherwise "
               f"the spread, bought a point through {', '.join(str(k) for k in style.KEYS[league])} when the line sits on one. "
               f"A pick also needs at least one real reason (line moved toward them, opponent in a lookahead or letdown "
               f"spot, your trap rule, bigger DraftKings money / reverse line move, or a ⭐ proven spot), and no 3-point "
               f"move against it; ⭐ proven spots go first. Each reason shows its record against the closing line since 2016. "
               f"Style score = how much a side looks like your picks (100 = most like you). "
               f"👍 / 👎 teach it: every vote retrains it ({sm.get('n_feedback', 0)} votes so far). "
               f"Teams with a key player out are left off. {test_txt}")
    if league == 'NFL':
        st.warning('Heads up: your NFL picks are hard to copy from data — what you see watching games isn\'t in the numbers, '
                   'and on past seasons the copy hit about 51% of legs. Use these to compare with your own reads, not to bet blindly.')
    top = sug[sug.shortlist] if len(sug) else sug
    if top.empty:
        st.write('No game fits your style AND has a real reason yet. Lines, splits and injuries update through the day — check back closer to kickoff.')
        return
    bmap = board.set_index('id').to_dict('index')
    cols = st.columns(len(top))
    for col, (_, s_) in zip(cols, top.iterrows()):
        r = bmap[s_.id]; r['id'] = s_.id
        with col:
            st.markdown(f"### {s_.leg}")
            st.caption(f"{s_.away} @ {s_.home} · {et(s_.kickoff)} · game line {s_.team} {s_.line:+g}")
            st.metric('Style score', f'{s_.style}')
            for why in s_.for_reasons:
                st.markdown('✅ ' + why)
            for why in s_.against_reasons:
                st.markdown('❌ ' + why)
            w_, l_ = s_.leg_record
            if w_ + l_:
                st.caption(f"You're {w_}-{l_} on {s_.leg_kind} legs on {s_.bucket} lines")
            for why in s_.why:
                st.caption('• ' + why)
            edge = s_.edge
            if edge is not None and pd.notna(edge):
                toward = edge if s_.side == 'away' else -edge
                st.caption(f"Model (compare only): {'agrees' if toward >= 0.5 else 'disagrees' if toward <= -0.5 else 'neutral'} "
                           f"({toward:+.1f} pts)")
            b1, b2 = st.columns(2)
            if s_.vote > 0:
                st.caption("✅ You said you'd bet this")
            elif b1.button("👍 I'd bet this", key=f'up_{s_.id}'):
                style.add_feedback(store, league, s_.to_dict(), +1)
                st.rerun()
            if b2.button('👎 Not for me', key=f'down_{s_.id}'):
                style.add_feedback(store, league, s_.to_dict(), -1)
                st.rerun()
            if st.button(f'Log ${mypicks.UNIT:.0f} on {s_.leg}', key=f'logs_{s_.id}'):
                log_straight(r, s_.team, s_.side, s_.line, note=f'style: {s_.leg}')
                st.success('Logged.')
    with st.expander('Every game: the side you would most likely play'):
        show = sug.assign(Kickoff=sug.kickoff.map(et), Matchup=sug.away + ' @ ' + sug.home,
                          **{'Your leg': sug.leg, 'Style score': sug['style'], 'Why': sug.why.map(lambda x: '; '.join(x)),
                             'Key injury': sug.key_injury.map({True: 'yes', False: ''}),
                             'Shortlist': sug.shortlist.map({True: 'yes', False: ''})})
        st.dataframe(show[['Kickoff', 'Matchup', 'Your leg', 'Style score', 'Shortlist', 'Why', 'Key injury']],
                     hide_index=True, width='stretch')
    with st.expander('What it learned about how you bet'):
        rec = sm.get('recognise')
        st.caption(f"Recognises your picks on a season it hasn't seen: {rec:.2f} (0.5 = guessing, 1.0 = perfect)." if rec else '')
        st.dataframe(pd.DataFrame([dict(Habit=style.habit_name(f), Weight=round(c, 2)) for f, c in
                                   sorted(zip(sm['feats'], sm['coef']), key=lambda x: -abs(x[1]))]),
                     hide_index=True, width='stretch')
        fr = []
        for k_, (_, _, nm) in enumerate(style.BUCKETS):
            f_ = sm['forms'][str(k_)]
            fr.append({'Line': nm, **{f'Your {kk} record': f"{v[0]}-{v[1]}" for kk, v in f_['record'].items()}})
        st.caption('Your record by line and leg type (from your history):')
        st.dataframe(pd.DataFrame(fr), hide_index=True, width='stretch')
        if feedback:
            fbl = [v for v in feedback if v.get('league') == league]
            st.caption(f"Your votes: {sum(v['vote'] > 0 for v in fbl)} 👍, {sum(v['vote'] < 0 for v in fbl)} 👎 — "
                       + ', '.join(f"{'👍' if v['vote'] > 0 else '👎'} {v.get('leg') or v['team']}" for v in fbl[-10:]))
        if st.button('Test my style on past seasons (about a minute)', key=f'replay_{league}'):
            with st.spinner('Letting the copy of you bet 2016-2022…'):
                res = style.replay(sm, games, league, range(2016, 2023), k=3)
            if res:
                res.update(n_picks=sm['n_picks'], seasons='2016-2022')
                store.put_json(f'style/{league}_replay.json', res)
                st.rerun()


STYLE_TEST = {'CFB': 'Tested on 2016-2022 (before you bet), played with your leg rules: its 3 legs a week went 181-117 (61%; '
                     'moneylines 85-33). The same sides at the plain spread went 164-130 (55.8%, 52.4% breaks even), winning in 6 of 7 seasons.',
              'NFL': 'Tested on 2016-2022 with your leg rules: legs 186-175 (51.5%), plain spread 168-186 (47.5%).'}


with tabs[0]:
    profile, all_picks = my_profile()
    if not board.empty and len(all_picks):
        style_section(all_picks)
        st.divider()
    if board.empty:
        st.info('No upcoming games in the window yet. Press **Update now** in the sidebar.')
    elif not profile:
        st.info('Upload your Pikkit export (transactions.csv) in the **My history** tab. It imports automatically, then your picks show here.')
    else:
        ranked = mypicks.rank_spots(board, profile, league, cached_data(league, data_stamp(league))['games'],
                                    store.get_json(f'cache/cfb/{current_season()}/polls.json') if league == 'CFB' else None)
        bmap = board.set_index('id').to_dict('index')
        n_show = 4
        mine = ranked[(ranked.power) & (~ranked.key_injury)] if league == 'CFB' else ranked[~ranked.key_injury]
        top = mine[mine.qualifies].head(n_show)
        st.subheader(f'Spots with a real reason — {league}')
        st.caption('⭐ = proven against the closing line since 2016; other reasons show their record too. '
                   'A game only makes this list when it fits your spots AND has at least one real reason: the line '
                   'moved toward your side, the opponent is in a lookahead or letdown spot, your trap rule, or bigger '
                   'DraftKings money / a reverse line move on your side. Games where the line moved 3+ points against '
                   'you, or a key player is out, are left off. Order: most net reasons first, then your record '
                   f"({profile['overall']['w']}-{profile['overall']['l']} on {profile['n']} graded spread picks). "
                   'The model is a note only.')
        cols = st.columns(max(1, len(top)))
        for col, (_, s_) in zip(cols, top.iterrows()):
            r = bmap[s_.id]; r['id'] = s_.id
            with col:
                st.markdown(f"### {s_.team} {s_.line:+g}")
                st.caption(f"{s_.away} @ {s_.home} · {et(s_.kickoff)}")
                for why in s_.for_reasons:
                    st.markdown('✅ ' + why)
                for why in s_.against_reasons:
                    st.markdown('❌ ' + why)
                st.metric('Your record in spots like this', f'{100 * s_.spot_rate:.0f}%')
                for why in s_.reasons:
                    st.caption('• ' + why)
                st.caption(f"Model (note only): {s_.model} · market {fav_text(r['away'], r['home'], r['current_home'])}, "
                         f"opened {fav_text(r['away'], r['home'], r['open_home'])}")
                extra = []
                for side_ in ('away', 'home'):
                    extra += spots_mod.describe((r.get('spots') or {}).get(side_, {}), r[side_])
                    extra += inj_mod.describe((r.get('injuries') or {}).get(side_, {}), r[side_])
                if s_.rival:
                    extra.append('Rivalry game')
                if (r.get('weather') or {}).get('flags'):
                    extra.append(' · '.join(r['weather']['flags']))
                fl = r.get('split_flags') or {}
                for k_, lab in (('sharp_side', 'bigger DraftKings money'), ('rlm_side', 'reverse line move'), ('public_side', 'public side')):
                    if fl.get(k_):
                        extra.append(f"{lab}: {r[fl[k_]]}")
                for e in extra:
                    st.caption('⚠️ ' + e)
                if st.button(f'Log ${mypicks.UNIT:.0f} on {s_.team} {s_.line:+g}', key=f'log_{s_.id}'):
                    log_straight(r, s_.team, s_.side, s_.line)
                    st.success('Logged as a straight bet.')
        if top.empty:
            st.write('No game has a real reason behind it yet. Lines, splits and injuries update through the day, so check back closer to kickoff.')

        st.subheader('Where the model is off Vegas')
        st.caption('The model now sits on the Vegas line for almost every game. It only moves off it for spots that '
                   'have beaten the closing line since 2016: '
                   + ('the favorite in a ranked-vs-ranked game (57.3%, 240-179), 1.5 pts.' if league == 'CFB' else
                      'an underdog against a favorite that is looking ahead (53.8%, 370-318), 1 pt.')
                   + ' These are this week\'s games where it does.')
        others = board.copy()
        others = others.assign(abs_edge=others.edge.abs()).sort_values(['abs_edge', 'score'], ascending=[False, False])
        others = others[others.edge.abs() >= config.LEAN_EDGE[league]].head(4)
        if others.empty:
            st.write('None this week — the model agrees with Vegas on every game.')
        for _, r in others.iterrows():
            side = r.model_side
            team = r.away if side == 'away' else r.home
            ln_ = team_line(r.current_home, side)
            c1, c2 = st.columns([4, 1])
            why_ = '; '.join(p['label'] for p in (r.proven or {}).get(side, [])) if isinstance(r.proven, dict) else ''
            c1.write(f"**{team} {ln_:+g}** ({r.away} @ {r.home}, {et(r.kickoff)}) — model line "
                     f"{fav_text(r.away, r.home, r.model_home)}, {abs(r.edge):.1f} pts off Vegas" + (f" ({why_})" if why_ else ''))
            if c2.button(f'Log ${mypicks.UNIT:.0f}', key=f'logm_{r.id}'):
                log_straight(r.to_dict(), team, side, ln_, note='model pick')
                st.success('Logged.')

        with st.expander('Every game ranked by your spots'):
            show = ranked.assign(Pick=ranked.team + ' ' + ranked.line.map(lambda x: f'{x:+g}'),
                                 Kickoff=ranked.kickoff.map(et), Matchup=ranked.away + ' @ ' + ranked.home,
                                 **{'Your record here': (100 * ranked.spot_rate).round(0).astype(int).astype(str) + '%',
                                    'Reasons for': ranked.for_reasons.map(lambda x: '; '.join(x)),
                                    'Against': ranked.against_reasons.map(lambda x: '; '.join(x)),
                                    'Makes list': ranked.qualifies.map({True: 'yes', False: ''}),
                                    'Why': ranked.reasons.map(lambda x: '; '.join(x)), 'Model': ranked.model,
                                    'Key injury': ranked.key_injury.map({True: 'yes', False: ''})})
            st.dataframe(show[['Kickoff', 'Matchup', 'Pick', 'Makes list', 'Reasons for', 'Against', 'Your record here', 'Model', 'Key injury']],
                         hide_index=True, width='stretch')
        with st.expander('Your profile (what the app learned from your picks)'):
            rows_ = []
            for key, v in sorted(profile['bucket'].items()):
                lg_, b_ = key.split('|')
                if lg_ in (league, 'ALL'):
                    rows_.append({'League': 'Both' if lg_ == 'ALL' else lg_, 'Spot': b_, 'Record': f"{v['w']}-{v['l']}",
                                  'Win %': round(100 * v['w'] / max(1, v['w'] + v['l']), 1)})
            st.dataframe(pd.DataFrame(rows_), hide_index=True, width='stretch')
            f_ = profile['followed']
            st.caption(f"Teams you know (3+ past picks): {f_['followed']['w']}-{f_['followed']['l']} · other teams: "
                       f"{f_['other']['w']}-{f_['other']['l']}. Teams you know: " + ', '.join(list(profile['teams'])[:25]))

# ------------------------------------------------------------------ This week
with tabs[1]:
    if board.empty:
        st.info('No upcoming games in the window. Press **Update now** in the sidebar (it also runs automatically when the app opens).')
    else:
        no_lines = int(board.snapshots.eq(0).sum())
        if no_lines:
            st.caption(f'{no_lines} games have no saved sportsbook snapshot yet — their line comes from the '
                       f'{"CFBD" if league == "CFB" else "nflverse"} schedule. Press **Update now** to save one.')
        st.subheader(f'Your shortlist — {league}')
        with st.expander('How the picks are chosen'):
            st.markdown(f"""
Every game gets a **Pick Score (0-100)** for the side the model likes. The shortlist is simply the highest scores that pass **My rules**, so the picks are always the top grades.

| Piece | Points |
|---|---|
| Model edge (Vegas-anchored model vs market) | up to **50** — the model sits on the Vegas line and only moves {'1.5 pts for a ranked-vs-ranked favorite' if league == 'CFB' else '1 pt for an underdog vs a favorite looking ahead'} (proven since 2016), which earns the full 50 |
| Each signal agreeing: line moving that way, bigger money on that side (DraftKings), reverse line move, opponent in a lookahead/letdown/sandwich spot | **+12** each |
| The same signals pointing at the other team | **−12** each |
| A key player (main QB, top RB, top two receivers) out, or the QB questionable, on that side | **−25** |

Grades: **A** 70+, **B** 55-69, **C** 40-54, **D** under 40. By default only **C or better** can be a pick (change it in My rules).

The old stats model disagreed with Vegas by 7+ points in about 1 of 5 college games, and those gaps came true only ~3% of the time against the closing line, so the model is now anchored to Vegas. The +12 / −12 / −25 weights are **our own rules**, not fitted to results — the
*My picks* tab tracks every grade so we can see which ones actually win and adjust them.
Injuries come from {'ESPN' if league == 'NFL' else 'Covers.com'} automatically. By game day the Vegas line already prices injuries, so they don't move the model; a key player out on your side costs 25 points of Pick Score and keeps the game off your picks.
""")
        sl = shortlist(board, league, rules=rules)
        band = model['calibration']['rank_close'][0]
        band_open = (model['calibration'].get('rank_open') or [None])[0]
        if sl.empty:
            st.info(f"No {league} game grades {rules.get('min_grade', 'C')} or better under your rules right now. "
                    'That is the model saying pass (or bet small) — not every week has value. Check again after the next '
                    'line/splits update.')
            ln = leans(board, league, rules)
            if len(ln):
                st.caption('Closest leans (below your grade cut): ' + ' · '.join(
                    f"{(x.away if x.model_side == 'away' else x.home)} {team_line(x.current_home, x.model_side):+g} "
                    f"(grade {x.grade}, score {x.score:.0f})" for x in ln.itertuples()))
        cols = st.columns(max(1, len(sl)))
        for col, (_, r) in zip(cols, sl.iterrows()):
            side = r.model_side
            team = r.away if side == 'away' else r.home
            line_now = team_line(r.current_home, side)
            with col:
                st.markdown(f"### {team} {line_now:+g}" if pd.notna(line_now) else f'### {team}')
                st.caption(f"{r.away} @ {r.home} · {et(r.kickoff)}")
                st.metric('Model line', fav_text(r.away, r.home, r.model_home),
                          delta=f"{abs(r.edge):.1f} pts toward {team}", delta_color='off')
                st.write(f"Market {fav_text(r.away, r.home, r.current_home)} · "
                         f"opened {fav_text(r.away, r.home, r.open_home)}")
                mv = r.moved_toward_model
                if pd.notna(mv) and mv >= 0.5:
                    st.warning(f'Line already moved {mv:.1f} toward the model since open — some value is gone.')
                elif pd.notna(mv) and mv <= -0.5:
                    st.success(f'Line moved {abs(mv):.1f} AWAY from the model since open — better number now.')
                st.write(f"**Grade {r.grade} · score {r.score:.0f}**")
                st.caption(' · '.join(r.score_parts))
                for s_ in ('away', 'home'):
                    for msg in inj_mod.describe(r.injuries.get(s_, {}), r.away if s_ == 'away' else r.home):
                        st.caption('🩹 ' + msg)
                for s in ('away', 'home'):
                    for msg in spots_mod.describe(r.spots.get(s, {}), r.away if s == 'away' else r.home):
                        st.caption('⚠️ ' + msg)
                if r.weather.get('flags'):
                    st.caption('🌬️ ' + ' · '.join(r.weather['flags']))
        msg = (f"History (walk-forward 2018-2025): the top 5% of weekly disagreements covered "
               f"{band['w']}-{band['l']} at the closing line")
        if band_open:
            msg += f" and {band_open['w']}-{band_open['l']} against the opening line"
        st.caption(msg + '. The edge is biggest early in the week; by kickoff the market has usually moved toward the model.')

        st.subheader('Every game')
        board = board.sort_values('score', ascending=False).reset_index(drop=True)
        view = pd.DataFrame({
            'Kickoff (ET)': board.kickoff.map(et),
            'Matchup': board.away + ' @ ' + board.home,
            'Market': [fav_text(a, h, x) for a, h, x in zip(board.away, board.home, board.current_home)],
            'Open': [fav_text(a, h, x) for a, h, x in zip(board.away, board.home, board.open_home)],
            'Move': board.move_home.round(1),
            'Model': [fav_text(a, h, x) for a, h, x in zip(board.away, board.home, board.model_home)],
            'Model likes': [(r.away if r.model_side == 'away' else r.home) if pd.notna(r.edge) else '—' for r in board.itertuples()],
            'Edge (pts)': board.edge.abs().round(1),
            'Grade': board.grade,
            'Score': board.score.round(0),
            'Injuries': [' · '.join(inj_mod.describe(i.get('away', {}), r.away) + inj_mod.describe(i.get('home', {}), r.home))
                         for i, r in zip(board.injuries, board.itertuples())],
            'My rules': [('✓' if ok else reason) for ok, reason in (passes_rules(r, league, rules) for r in board.to_dict('records'))],
            'Signals': [', '.join(x) for x in board.confluence],
            'Why (score parts)': [' · '.join(x) for x in board.score_parts],
            'Spots': [' / '.join(sum((spots_mod.describe(s.get(k, {}), (r.away if k == 'away' else r.home)) for k in ('away', 'home')), []))
                      for s, r in zip(board.spots, board.itertuples())],
            'Weather': [' · '.join(w.get('flags') or []) for w in board.weather],
            'Total (mkt / model)': [f"{t:g} / {m:.1f}" if pd.notna(t) and pd.notna(m) else '—'
                                    for t, m in zip(board.current_total, board.model_total)],
        })
        st.dataframe(view, hide_index=True, width='stretch',
                     column_config={'Edge (pts)': st.column_config.ProgressColumn(
                         'Edge (pts)', min_value=0, max_value=float(max(1.0, board.edge.abs().max() or 1)), format='%.1f')})
        st.caption('Sorted by Pick Score. Market = median of saved sportsbook quotes. Move = change in the home spread since '
                   'the first saved line (negative = moved toward the home team). Model = stats model after injury adjustment. '
                   + (f"Injury reports: {board.injury_source.iloc[0]}." if board.injury_source.notna().any() else 'Injury reports not loaded yet — press Update now.'))

# ------------------------------------------------------------------ Check my pick
with tabs[2]:
    if board.empty:
        st.info('No games loaded yet — press **Update now** in the sidebar.')
    else:
        labels = [f"{r.away} @ {r.home} — {et(r.kickoff)}" for r in board.itertuples()]
        i = st.selectbox('Game', range(len(board)), format_func=lambda k: labels[k])
        r = board.iloc[i].to_dict()
        side = st.radio('Your side', ['away', 'home'], horizontal=True,
                        format_func=lambda s: f"{r[s]} {team_line(r['current_home'], s):+g}" if pd.notna(r['current_home']) else r[s])
        team = r[side]
        c1, c2, c3 = st.columns(3)
        default_line = float(team_line(r['current_home'], side)) if pd.notna(r['current_home']) else 0.0
        my_line = c1.number_input(f'Your number for {team}', value=default_line, step=0.5, format='%.1f')
        my_odds = c2.number_input('Price', value=-110, step=5)
        adj = c3.number_input(f'Your news/injury adjustment (pts toward {team})', value=0.0, step=0.5, format='%.1f',
                              help='Positive = you think the news helps your side by this many points. It shifts the model line.')
        # recompute edge with the user's number and adjustment
        sgn = 1 if side == 'away' else -1
        model_home = r['model_home'] + (adj * sgn if pd.notna(r['model_home']) else 0)
        my_home = my_line if side == 'home' else -my_line
        rr = dict(r)
        rr['edge'] = model_home - my_home if pd.notna(model_home) else np.nan
        verdict, why = verdict_for(side, rr, league)
        n, hits = confluence(side, rr, league)
        sc, sc_parts = pick_score(side, rr, league)
        color = {'Model backs it (strong)': 'green', 'Model leans this way': 'green', 'No real edge either way': 'gray',
                 'Model leans against': 'orange', 'Model says stay off': 'red'}.get(verdict, 'gray')
        st.markdown(f"## :{color}[{verdict}]")
        st.write(why)
        a, b, c, d = st.columns(4)
        a.metric('Model line', fav_text(r['away'], r['home'], model_home))
        b.metric('Your number', f"{team} {my_line:+g}")
        c.metric('Pick score for your side', f"{sc:.0f} · grade {grade_from_score(sc)}")
        best = r['best_away'] if side == 'away' else r['best_home']
        d.metric('Best number saved', f"{best[0]:+g} ({best[1]:+.0f}, {best[2]})" if best and pd.notna(best[0]) else '—')
        st.caption('Score parts: ' + ' · '.join(sc_parts))
        for s_ in ('away', 'home'):
            rep_ = (r.get('injuries') or {}).get(s_, {})
            lines_ = inj_mod.describe(rep_, r[s_])
            if lines_:
                st.warning('🩹 ' + ' · '.join(lines_) + ('  (model already moved 2 pts for the QB)' if rep_.get('qb_out') and league == 'NFL' else ''))
            elif rep_.get('others'):
                st.caption(f"{r[s_]} also listed out: " + ', '.join(rep_['others']))
            if not rep_.get('known'):
                st.caption(f"{r[s_]}: key players not identified yet (needs a stats update) — injuries can't be checked.")
        st.write('**Signals pointing your way:** ' + (', '.join(hits) if hits else 'none'))
        other = 'home' if side == 'away' else 'away'
        against = [h for h in confluence(other, rr, league)[1]]
        st.write('**Signals pointing the other way:** ' + (', '.join(against) if against else 'none'))
        with st.expander('Market & splits', expanded=True):
            st.write(f"Opened {fav_text(r['away'], r['home'], r['open_home'])} ({r.get('open_source') or 'no opener saved'}) → "
                     f"now {fav_text(r['away'], r['home'], r['current_home'])}. "
                     f"Sharp books: {fav_text(r['away'], r['home'], r['sharp_home'])}.")
            spl = r.get('splits')
            if spl:
                st.write(f"DraftKings — {r['away']}: {spl['away']['bets']:.0f}% bets / {spl['away']['handle']:.0f}% money · "
                         f"{r['home']}: {spl['home']['bets']:.0f}% bets / {spl['home']['handle']:.0f}% money")
                for note in (r.get('split_flags') or {}).get('notes', []):
                    st.caption('• ' + note)
            else:
                st.caption('No DraftKings splits pasted for this game yet (DraftKings splits tab).')
        with st.expander('Spots, rest & weather', expanded=True):
            recs = model.get('spot_records', {})
            for s in ('away', 'home'):
                info = r['spots'].get(s, {})
                for msg in spots_mod.describe(info, r[s]):
                    k = 'sandwich' if info.get('sandwich') else 'lookahead' if info.get('lookahead') else 'letdown'
                    rec = recs.get(k)
                    st.write('⚠️ ' + msg + (f" — fading this spot went {rec['w']}-{rec['l']} ({100 * rec['pct']:.1f}%) at closing lines 2016-2025" if rec else ''))
                if info.get('rest_days') is not None:
                    st.caption(f"{r[s]}: {info['rest_days']} days since last game")
            w = r.get('weather') or {}
            if w.get('indoor'):
                st.caption('Indoors.')
            elif w.get('data'):
                wd = w['data']
                st.caption(f"Kickoff forecast: {wd.get('temp_f')}°F, wind {wd.get('wind_mph')} mph (gusts {wd.get('gust_mph')}), precip {wd.get('precip_pct')}%")
        units = st.number_input(f'Units (${mypicks.UNIT:.0f} each)', value=1.0, step=0.5)
        note = st.text_input('Notes (why you like it)')
        if st.button('Log this pick', type='primary'):
            ledger.add(store, dict(league=league, game_id=r['id'], season=int(r['season']), week=int(r['week']),
                                   away=r['away'], home=r['home'], kickoff=str(r['kickoff']), side=side, team=team,
                                   line=float(my_line), odds=float(my_odds), units=float(units),
                                   book=(best[2] if best else None), model_line_home=float(model_home) if pd.notna(model_home) else None,
                                   market_home_at_pick=float(r['current_home']) if pd.notna(r['current_home']) else None,
                                   verdict=verdict, signals=hits, grade=grade_from_score(sc), score=sc, adjustment=float(adj), notes=note))
            st.success('Logged. It will be graded with the closing line and final score after the game.')

# ------------------------------------------------------------------ Splits
with tabs[3]:
    keys = store.list(f'splits/{league}/')
    last = store.get_json(keys[-1]) if keys else None
    c1, c2 = st.columns([3, 1])
    c1.write('DraftKings % of bets and % of money are downloaded automatically (DraftKings Network splits page, '
             "or VSiN's copy of the same DraftKings numbers if that page can't be read).")
    if c2.button('Refresh splits now'):
        with st.spinner('Reading DraftKings splits…'):
            rows_up = board[['id', 'away', 'home']].to_dict('records') if not board.empty else []
            res = splits_mod.auto_refresh(store, league, rows_up, log=lambda m: None)
        st.cache_data.clear()
        st.rerun()
    if last:
        st.caption(f"Last download: {str(last.get('captured_at'))[:16]} UTC from {last.get('source') or 'manual paste'} — "
                   f"{len(last.get('matches') or {})} of {len(last.get('games') or [])} games matched to the schedule.")
    lt = splits_mod.latest(store, league)
    if lt and not board.empty:
        rows = []
        for r in board.itertuples():
            if r.id in lt:
                spl, at = lt[r.id]
                fl = r.split_flags or {}
                rows.append({'Matchup': f'{r.away} @ {r.home}', 'Saved (UTC)': str(at)[:16],
                             'Away bets / money': f"{spl['away']['bets']:.0f}% / {spl['away']['handle']:.0f}%",
                             'Home bets / money': f"{spl['home']['bets']:.0f}% / {spl['home']['handle']:.0f}%",
                             'Bigger money': {'away': r.away, 'home': r.home}.get(fl.get('sharp_side'), ''),
                             'Public side': {'away': r.away, 'home': r.home}.get(fl.get('public_side'), ''),
                             'Reverse move': {'away': r.away, 'home': r.home}.get(fl.get('rlm_side'), '')})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    else:
        st.info('No splits saved yet — they download with **Update now** (or automatically when the app opens).')
    st.caption('"Bigger money" = the side has 10+ points more of the money than of the bets (fewer, larger tickets). '
               '"Reverse move" = the line moved against the side most bettors are on. Research note: these are signals to '
               'weigh, not proven edges — a study of college football totals (2005-2016) found following reverse line '
               'movement alone was not profitable. Your pick log will show whether they help your picks.')
    with st.expander('Backup: paste splits by hand (only if the automatic download fails)'):
        text = st.text_area('Paste the DraftKings splits page text', height=160)
        if text:
            parsed = splits_mod.parse_dk(text)
            slate_rows = board[['id', 'away', 'home']].to_dict('records') if not board.empty else []
            matches = {}
            for k, g in enumerate(parsed):
                gid, rev, score = splits_mod.best_match(g, slate_rows, league)
                if gid:
                    g['_reversed'] = bool(rev)
                    matches[str(k)] = gid
            st.write(f'Read {len(parsed)} games, matched {len(matches)}.')
            if st.button('Save pasted splits'):
                splits_mod.save(store, league, parsed, matches)
                st.cache_data.clear()
                st.success('Saved.')

# ------------------------------------------------------------------ My picks
with tabs[4]:
    st.subheader('Your betting history (Pikkit)')
    hist = mypicks.load_history(store)
    up = st.file_uploader('Upload a new Pikkit export (transactions.csv) any time — the app re-grades it and relearns your spots',
                          type=['csv'])
    if st.session_state.get('import_msg'):
        st.success(st.session_state.pop('import_msg'))
    file_id = f'{up.name}:{up.size}' if up is not None else None
    if up is not None and st.session_state.get('imported_file') != file_id:
        with st.spinner('Grading every spread pick against final scores…'):
            try:
                gb = {lg: cached_data(lg, data_stamp(lg))['games'] for lg in config.LEAGUES}
                new_picks, summ = mypicks.import_pikkit(up.getvalue(), gb)
            except Exception as exc:
                import traceback
                st.error(f'Import failed ({type(exc).__name__}). Details below.')
                st.code(traceback.format_exc())
                new_picks = pd.DataFrame()
        st.session_state['imported_file'] = file_id
        if len(new_picks):
            mypicks.save_history(store, new_picks, summ)
            st.session_state['import_msg'] = f'Imported {len(new_picks)} football picks. My spots is ready.'
            st.rerun()                                    # redraw so My spots uses the new history
        else:
            st.error('No football spread/moneyline picks could be matched in that file.')
    if len(hist):
        sp = hist[(hist.kind == 'spread') & hist.result.isin(['W', 'L', 'P'])]
        w, l, pu = (sp.result == 'W').sum(), (sp.result == 'L').sum(), (sp.result == 'P').sum()
        as_straight = mypicks.UNIT * (w * 100 / 110 - l)
        c1, c2, c3 = st.columns(3)
        c1.metric('Your spread picks (each counted once)', f'{w}-{l}' + (f'-{pu}' if pu else ''),
                  delta=f'{100 * w / max(1, w + l):.1f}%', delta_color='off')
        c2.metric(f'If each was a ${mypicks.UNIT:.0f} straight bet', f'${as_straight:,.0f}')
        summ = (store.get_json('mypicks/history.json') or {}).get('summary') or mypicks.bundled_summary()
        if summ:
            c3.metric('What your account actually made (all bets)', f"${summ['profit']:,.0f}",
                      delta=f"on ${summ['staked']:,.0f} wagered", delta_color='off')
        st.caption('Most of your spread picks were parlay legs. Your picks were right far more often than your parlays paid — '
                   'tracking them as straight bets here shows what the picking itself is worth.')
        by = sp.assign(Bucket=sp.line.map(mypicks.bucket)).groupby(['league', 'Bucket']).result.agg(
            lambda x: f"{(x == 'W').sum()}-{(x == 'L').sum()}")
        with st.expander('Your history by league and spot'):
            st.dataframe(by.rename('Record').reset_index(), hide_index=True, width='stretch')
    st.divider()
    st.subheader(f'Picks logged in this app (${mypicks.UNIT:.0f} per pick)')
    picks = ledger.load(store)
    if picks:
        games_by = {}
        for lg in config.LEAGUES:
            try:
                games_by[lg] = cached_data(lg, data_stamp(lg))['games']
            except Exception:
                pass
        closes = ledger.find_closes(store, picks)
        changed = ledger.grade(store, games_by, closes)
        picks = ledger.load(store)
        if changed:
            st.success(f'Graded {changed} picks.')
        st.subheader('Results')
        s = ledger.summary(picks)
        if not s.empty:
            s.insert(3, 'Dollars', (s['Units'] * mypicks.UNIT).round(0))
            st.dataframe(s, hide_index=True, width='stretch')
            for by, title in (('league', 'By league'), ('verdict', 'By what the model said'), ('grade', 'By grade')):
                t = ledger.summary(picks, by)
                if not t.empty:
                    st.write(f'**{title}**')
                    st.dataframe(t, hide_index=True, width='stretch')
        st.caption('Avg CLV = points better than the closing number. Positive CLV over 100+ bets is the clearest sign '
                   'your picks have a real edge, long before the win-loss record settles down.')
        st.subheader('All picks')
        df = pd.DataFrame(picks)
        show = [c for c in ['logged_at', 'league', 'team', 'line', 'odds', 'units', 'verdict', 'grade', 'closing_line',
                            'clv', 'result', 'profit', 'final', 'notes', 'pick_id'] if c in df]
        st.dataframe(df[show].sort_values('logged_at', ascending=False), hide_index=True, width='stretch')
        rm = st.selectbox('Delete a pick', [None] + [p['pick_id'] for p in picks],
                          format_func=lambda x: '—' if x is None else next(f"{p['team']} {p['line']:+g} ({p['logged_at'][:10]})" for p in picks if p['pick_id'] == x))
        if rm and st.button('Delete'):
            ledger.remove(store, rm)
            st.rerun()
        st.download_button('Download picks (CSV)', df.to_csv(index=False), 'picks.csv')
    else:
        st.info('No picks logged yet. Use the **Log $25** buttons in **My picks**, or **Check my pick**.')

# ------------------------------------------------------------------ Ratings
with tabs[5]:
    try:
        d = cached_data(league, stamp)
        g = d['games']
        now = pd.Timestamp(datetime.now(timezone.utc))
        from ratings import fit_ratings
        r = fit_ratings(g[g.completed], d['tg'], now, current_season(), league)
        mr = fit_market_ratings(g[g.completed], now, current_season())
        t = team_table(r, league)
        if mr:
            t['market_rating'] = t.team.map(mr['R'])
        if league == 'CFB':
            t = t[t.team != 'FCS']
        t = t[t.games > 0] if (t.games > 0).any() else t
        num = t.select_dtypes('number').columns
        t[num] = t[num].round(3)
        st.write('Opponent-adjusted ratings (positive = better than average; defense columns are flipped so positive = better defense). '
                 '`points` = points per game, `sr` = success rate, `expl` = explosiveness (CFB) / EPA per play (NFL), '
                 '`ypp` = yards per play. `market_rating` = team strength implied by recent closing lines.')
        st.dataframe(t.sort_values('market_rating' if 'market_rating' in t else 'off_points', ascending=False),
                     hide_index=True, width='stretch')
    except Exception as exc:
        st.info(f'Ratings unavailable: {exc}')

# ------------------------------------------------------------------ How it works
with tabs[6]:
    b = model['backtest']
    st.markdown(f"""
### What the model does
For every game it builds a **fair spread** from opponent-adjusted team ratings (points, success rate, explosiveness/EPA,
yards per play — offense and defense, home field, recent games weighted more) plus the team strengths implied by
recent closing lines. The gap between that number and the sportsbook line is the **edge**.

### Honest track record (walk-forward: every season predicted by a model that never saw it)
* Seasons {b['seasons']}, {b['games']} games. Average miss vs final margin: model {b['model_mae']} pts, closing line {b['market_mae']} pts.
* Every game vs the closing line: {b['every_game']['w']}-{b['every_game']['l']} ({100 * b['every_game']['pct']:.1f}%).
* Top 2 disagreements per week vs the closing line: {b['top2_per_week']['w']}-{b['top2_per_week']['l']} ({100 * b['top2_per_week']['pct']:.1f}%).
""")
    if 'vs_open' in b:
        v = b['vs_open']
        st.markdown(f"* **Top 2 per week vs the OPENING line: {v['top2_per_week']['w']}-{v['top2_per_week']['l']} "
                    f"({100 * v['top2_per_week']['pct']:.1f}%)**, and the line moved toward the model by "
                    f"{v['top2_per_week']['avg_clv']:+.2f} pts on average. The model's information gets priced in during "
                    "the week — its best use is betting early, before the market moves.")
    st.markdown("""
### What this means for you
* A spread model that honestly shows 65% at game-day lines does not exist in the data. If any tool shows that, it's overfit.
* Use the model as a **fair-number check**: when it strongly disagrees with your side, that's a reason to pass or shop for a better number.
* The fastest proof of your own edge is **closing-line value** in *My picks*. Sharp bettors beat the close consistently.

### Research notes
* Massey-Peabody (Wharton), a model built the same way, reported ~56% ATS on 500+ public picks — that's what elite looks like.
* Home underdogs covered 53% from 1980-2003 (Levitt and others), then 47.8% from 2004-2012 once the edge became known.
* Lines tend to over-react (Simon, *Management Science* 2024), so a big early move can partly reverse.
* Power-conference teams vs Group of 5 (a published 2002-2011 edge): 52.1% ATS in 2016-2025, below the 52.4% break-even.
* A 25-7 run is common luck: a 50% bettor making 250 picks a season has about a 5% chance of one; a 55% bettor about 19%.
""")
    recs = model.get('spot_records', {})
    if recs:
        st.markdown('**Fading situational spots at closing lines (2016-2025):** ' + ' · '.join(
            f"{k} {v['w']}-{v['l']} ({100 * v['pct']:.1f}%)" for k, v in recs.items()))
    with st.expander('Full backtest numbers'):
        st.json(b)
    with st.expander('Model coefficients'):
        st.json(dict(margin=model['margin'], total=model.get('total'), calibration=model['calibration'],
                     rating_settings=model.get('rating_settings')))
