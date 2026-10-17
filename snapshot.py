"""Scheduled job: save sportsbook lines + DraftKings splits (and refresh stats) to storage.

Runs from GitHub Actions on a schedule so openers, line moves and closing lines
are captured even when nobody has the app open.  Reads the same keys as the app
from environment variables: CFBD_API_KEY, ODDS_API_KEY, R2_ACCOUNT_ID,
R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.

    python snapshot.py            # both leagues: stats, lines, splits
    python snapshot.py --no-stats # lines + splits only
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from storage import Store
from data import update_all

KEYS = ('CFBD_API_KEY', 'ODDS_API_KEY', 'R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET', 'MODEL_DATA_DIR')


def main():
    secrets = {k: os.environ[k] for k in KEYS if os.environ.get(k)}
    store = Store(secrets)
    print('Storage:', store.describe())
    if store.kind != 'r2':
        print('WARNING: R2 secrets missing - snapshots are saved inside the runner and will be lost.')
    stats = '--no-stats' not in sys.argv
    failed = False
    for league in ('CFB', 'NFL'):
        problems = update_all(store, secrets, league, log=lambda m: print(f'[{league}] {m}'), stats=stats)
        for p in problems:
            print(f'[{league}] PROBLEM: {p}')
            failed = failed or p.startswith('Lines')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
