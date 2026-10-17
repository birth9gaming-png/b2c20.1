"""Kickoff-hour forecast from Open-Meteo (free, no key).  Only for games within 7 days."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests

import config
from nfl_data import STADIUMS

URL = 'https://api.open-meteo.com/v1/forecast'


def forecast(lat: float, lon: float, kickoff: pd.Timestamp) -> dict | None:
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    now = datetime.now(timezone.utc)
    if not (now - timedelta(hours=6) <= kickoff.to_pydatetime() <= now + timedelta(days=7)):
        return None
    r = requests.get(URL, timeout=20, params=dict(
        latitude=lat, longitude=lon, hourly='temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability',
        temperature_unit='fahrenheit', wind_speed_unit='mph', timezone='UTC',
        start_date=kickoff.strftime('%Y-%m-%d'), end_date=(kickoff + timedelta(hours=4)).strftime('%Y-%m-%d')))
    r.raise_for_status()
    h = r.json().get('hourly', {})
    times = pd.to_datetime(h.get('time', []), utc=True, format='ISO8601')
    if not len(times):
        return None
    i = int(abs(times - kickoff.floor('h')).argmin())
    pick = lambda k: (h.get(k) or [None] * len(times))[i]
    return dict(temp_f=pick('temperature_2m'), wind_mph=pick('wind_speed_10m'),
                gust_mph=pick('wind_gusts_10m'), precip_pct=pick('precipitation_probability'))


def flags(w: dict | None, indoor: bool) -> list[str]:
    if indoor:
        return ['Indoors / roof']
    if not w:
        return []
    out = []
    if w.get('wind_mph') is not None and w['wind_mph'] >= config.WIND_FLAG_MPH:
        out.append(f"Wind {w['wind_mph']:.0f} mph")
    if w.get('temp_f') is not None and w['temp_f'] <= config.COLD_FLAG_F:
        out.append(f"{w['temp_f']:.0f}°F")
    if w.get('precip_pct') is not None and w['precip_pct'] >= 60:
        out.append(f"Rain/snow {w['precip_pct']:.0f}%")
    return out


def location(league: str, game, venues: pd.DataFrame | None):
    """(lat, lon, indoor) for a game."""
    if league == 'NFL':
        lat, lon, roof = STADIUMS.get(game.home, (None, None, 'outdoors'))
        return lat, lon, roof in ('dome',)
    if venues is not None and getattr(game, 'venue_id', None) is not None and pd.notna(game.venue_id):
        v = venues[venues.venue_id == game.venue_id]
        if len(v):
            v = v.iloc[0]
            return v.lat, v.lon, bool(v.dome)
    return None, None, False
