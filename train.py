"""Rebuild model_CFB.json and model_NFL.json from the bundled history files.

    python train.py            # both leagues
    python train.py CFB
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from history import load_history, HIST
from model import history_features, train, save_model
from ratings import weekly_market_ratings
from spots import history_records


def openers():
    p = HIST / 'cfb_openers.csv.gz'
    if not p.exists():
        return None
    o = pd.read_csv(p, dtype={'id': str})
    return o.set_index('id')['open_home']


def main(leagues):
    for lg in leagues:
        t = time.time()
        g, tg = load_history(lg)
        df = history_features(g, tg, lg)
        model = train(df, lg, openers=openers() if lg == 'CFB' else None)
        model['spot_records'] = history_records(g, lg, weekly_market_ratings(g))
        path = save_model(model)
        print(f'{lg}: trained on {model["games"]} games in {time.time() - t:.0f}s -> {path.name}')
        print(json.dumps(model['backtest'], indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['CFB', 'NFL'])
