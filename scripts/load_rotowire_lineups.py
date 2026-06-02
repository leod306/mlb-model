"""
load_rotowire_lineups.py
------------------------
Scrapes RotoWire's free projected lineup page and stores batting orders
in the lineups table with source='rotowire'.

Official MLB lineups (source='mlb') always win — never overwritten.
Falls back to league avg OPS if no lineups available.

Player ID resolution order:
  1. lineups table (source='mlb') — best source, exact IDs from MLB feed
  2. game_probables table — covers SPs
  3. MLB People search API — fallback for unresolved names
  4. None — saved as NULL, BvP will skip but lineup OPS still uses league avg

Run order: BEFORE mlb_engine_daily.py
"""
from __future__ import annotations

import os
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if os.getenv("DYNO") is None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass

from app.db import engine


# =============================================================================
# CONFIG
# =============================================================================

ROTOWIRE_URL    = "https://www.rotowire.com/baseball/daily-lineups.php"
REQUEST_TIMEOUT = 20
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

MLB_PEOPLE_SEARCH = "https://statsapi.mlb.com/api/v1/people/search"

TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",   "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",      "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",           "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",        "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",       "Detroit Tigers": "DET",
    "Houston Astros": "HOU",         "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",     "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",          "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",        "New York Mets": "NYM",
    "New York Yankees": "NYY",       "Oakland Athletics": "ATH",
    "Athletics": "ATH",              "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",     "San Diego Padres": "SD",
    "San Francisco Giants": "SF",    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",          "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    "D-backs": "ARI",   "Diamondbacks": "ARI",
    "Braves": "ATL",    "Orioles": "BAL",
    "Red Sox": "BOS",   "Cubs": "CHC",
    "White Sox": "CWS", "Reds": "CIN",
    "Guardians": "CLE", "Rockies": "COL",
    "Tigers": "DET",    "Astros": "HOU",
    "Royals": "KC",     "Angels": "LAA",
    "Dodgers": "LAD",   "Marlins": "MIA",
    "Brewers": "MIL",   "Twins": "MIN",
    "Mets": "NYM",      "Yankees": "NYY",
    "Phillies": "PHI",  "Pirates": "PIT",
    "Padres": "SD",     "Giants": "SF",
    "Mariners": "SEA",  "Cardinals": "STL",
    "Rays": "TB",       "Rangers": "TEX",
    "Blue Jays": "TOR", "Nationals": "WSH",
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC":  "KC",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "ATH",
    "ATH": "ATH", "PHI": "PHI", "PIT": "PIT", "SD":  "SD",
    "SF":  "SF",  "SEA": "SEA", "STL": "STL", "TB":  "TB",
    "TEX": "TEX", "TOR": "TOR", "WSH": "WSH",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove punctuation for fuzzy matching."""
    if not name:
        return ""
    # Normalize unicode (é → e, etc.)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase, remove punctuation except spaces
    clean = re.sub(r"[^a-z\s]", "", ascii_name.lower())
    return " ".join(clean.split())


def should_skip(raw: str) -> bool:
    t = raw.strip().replace("\xa0", " ").replace("\u00a0", " ")
    if not t or len(t) < 2:
        return True
    if "ERA" in t:
        return True
    if t in ("Expected Lineup", "Confirmed Lineup", "Projected Lineup"):
        return True
    if re.match(r"^\$[\d,]+$", t):
        return True
    return False


def clean_name(raw: str) -> str:
    t = raw.strip().replace("\xa0", " ").replace("\u00a0", " ")
    t = re.sub(r"\s*[LRS]$", "", t).strip()
    return t


# =============================================================================
# PLAYER ID CACHE — built from existing DB data, no API needed
# =============================================================================

def build_player_id_cache() -> tuple[dict[str, int], dict[str, list]]:
    """
    Build two lookup structures from existing DB data:
    1. full_cache: normalized full name -> player_id
    2. last_cache: normalized last name -> [(full_name, player_id), ...]
       Used to match RotoWire abbreviated names like "T. Bazzana"
    """
    full_cache: dict[str, int] = {}
    last_cache: dict[str, list] = {}

    def add(name: str, pid: int, overwrite: bool = True) -> None:
        key = normalize_name(name)
        if not key:
            return
        if overwrite or key not in full_cache:
            full_cache[key] = pid
        parts = key.split()
        if parts:
            last = parts[-1]
            last_cache.setdefault(last, [])
            if not any(p[1] == pid for p in last_cache[last]):
                last_cache[last].append((key, pid))

    try:
        with engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT player_name, player_id
                FROM lineups
                WHERE source = 'mlb'
                  AND player_id IS NOT NULL
                  AND player_name IS NOT NULL
            """)).fetchall()
            for name, pid in rows:
                add(name, int(pid), overwrite=True)

            rows2 = conn.execute(text("""
                SELECT DISTINCT home_sp_name, home_sp_id
                FROM game_probables
                WHERE home_sp_id IS NOT NULL AND home_sp_name IS NOT NULL
                UNION
                SELECT DISTINCT away_sp_name, away_sp_id
                FROM game_probables
                WHERE away_sp_id IS NOT NULL AND away_sp_name IS NOT NULL
            """)).fetchall()
            for name, pid in rows2:
                add(name, int(pid), overwrite=False)

    except Exception as e:
        log(f"  ⚠️  player_id cache build warning: {e}")

    return full_cache, last_cache


def lookup_player_id_api(name: str) -> Optional[int]:
    """
    Last-resort MLB People search API lookup.
    Only called for names not in the DB cache — usually <5% of players.
    """
    try:
        r = requests.get(
            MLB_PEOPLE_SEARCH,
            params={"names": name, "sportId": 1},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        people = r.json().get("people", [])
        if not people:
            return None
        # Take the first active MLB player match
        for p in people:
            if p.get("active") and p.get("primarySport", {}).get("id") == 1:
                return int(p["id"])
        return int(people[0]["id"]) if people else None
    except Exception:
        return None


def resolve_player_id(
    name: str,
    full_cache: dict[str, int],
    last_cache: dict[str, list],
    api_cache: dict[str, Optional[int]],
) -> Optional[int]:
    """
    Resolve a player name to MLB ID using three strategies:
    1. Exact normalized full name match  ("tyler bazzana" -> ID)
    2. Abbreviated name match ("t bazzana" -> last name lookup -> confirm first initial)
    3. MLB People API fallback
    """
    key = normalize_name(name)
    if not key:
        return None

    # 1. Exact full name match
    if key in full_cache:
        return full_cache[key]

    # 2. Abbreviated name match — handle "T. Bazzana" -> "t bazzana"
    parts = key.split()
    if len(parts) >= 2:
        first_part = parts[0]   # e.g. "t" (from "T. Bazzana")
        last_part  = parts[-1]  # e.g. "bazzana"
        candidates = last_cache.get(last_part, [])
        if len(first_part) == 1:
            # Single initial — match against first letter of first name
            matches = [
                pid for full_key, pid in candidates
                if full_key.split()[0].startswith(first_part)
            ]
            if len(matches) == 1:
                full_cache[key] = matches[0]  # cache for next time
                return matches[0]
            # Multiple matches with same initial+last — can't resolve safely
        elif candidates:
            # Full last name but partial first — try prefix match
            matches = [
                pid for full_key, pid in candidates
                if full_key.split()[0].startswith(first_part)
            ]
            if len(matches) == 1:
                full_cache[key] = matches[0]
                return matches[0]

    # 3. DB cache (full_cache) re-check after abbreviation attempt
    if key in full_cache:
        return full_cache[key]

    # 2. API lookup (cached so each unique name only hits API once)
    if key not in api_cache:
        pid = lookup_player_id_api(name)
        api_cache[key] = pid
        if pid:
            full_cache[key] = pid  # warm the cache for next time

    return api_cache.get(key)


# =============================================================================
# DB HELPERS
# =============================================================================

def ensure_db_ready() -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE lineups
            ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'mlb'
        """))
        conn.execute(text("""
            ALTER TABLE lineups
            ALTER COLUMN player_id DROP NOT NULL
        """))
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS lineups_game_team_order_idx
            ON lineups (game_pk, team, batting_order)
        """))


def get_official_game_pks(target_date: date) -> set:
    try:
        with engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT game_pk FROM lineups
                WHERE official_date = :d AND source = 'mlb'
            """), {"d": target_date}).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def get_game_pk(home_team: str, away_team: str, target_date: date) -> Optional[int]:
    try:
        with engine.begin() as conn:
            row = conn.execute(text("""
                SELECT game_pk FROM games
                WHERE official_date = :d
                  AND home_team = :home
                  AND away_team = :away
                  AND season = :season
                LIMIT 1
            """), {
                "d": target_date,
                "home": home_team,
                "away": away_team,
                "season": target_date.year,
            }).fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def upsert_rows(rows: list) -> int:
    if not rows:
        return 0

    sql = text("""
        INSERT INTO lineups (
            game_pk, official_date, team, side,
            batting_order, player_id, player_name, position, source
        ) VALUES (
            :game_pk, :official_date, :team, :side,
            :batting_order, :player_id, :player_name, :position, 'rotowire'
        )
        ON CONFLICT (game_pk, team, batting_order) DO UPDATE SET
            player_name = EXCLUDED.player_name,
            player_id   = COALESCE(EXCLUDED.player_id, lineups.player_id),
            position    = EXCLUDED.position,
            source      = CASE
                            WHEN lineups.source = 'mlb' THEN 'mlb'
                            ELSE 'rotowire'
                          END,
            updated_at  = NOW()
        WHERE lineups.source != 'mlb'
    """)

    by_game: dict = {}
    for r in rows:
        by_game.setdefault(r["game_pk"], []).append(r)

    inserted = 0
    for game_pk, game_rows in by_game.items():
        try:
            with engine.begin() as conn:
                conn.execute(sql, game_rows)
            inserted += len(game_rows)
            resolved = sum(1 for r in game_rows if r["player_id"] is not None)
            teams = f"{game_rows[0]['team']}/{game_rows[-1]['team']}"
            log(f"  ✅ {teams} ({game_pk}): {len(game_rows)} rows | {resolved} IDs resolved")
        except Exception as e:
            log(f"  ⚠️  game {game_pk} failed: {e}")

    return inserted


# =============================================================================
# SCRAPER
# =============================================================================

def fetch_page() -> Optional[str]:
    try:
        resp = requests.get(ROTOWIRE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        log(f"❌ Failed to fetch RotoWire: {e}")
        return None


def norm_team(name: str) -> Optional[str]:
    name = name.strip()
    if name in TEAM_MAP:
        return TEAM_MAP[name]
    for k, v in TEAM_MAP.items():
        if len(k) > 3 and k.lower() in name.lower():
            return v
    return None


def parse_lineups(
    html: str,
    target_date: date,
    full_cache: dict[str, int],
    last_cache: dict[str, list],
    api_cache: dict[str, Optional[int]],
) -> list:
    soup = BeautifulSoup(html, "html.parser")
    boxes = soup.select(".lineup__box")
    log(f"  Found {len(boxes)} lineup boxes")

    all_rows = []
    for box in boxes:
        try:
            rows = _parse_box(box, target_date, full_cache, last_cache, api_cache)
            all_rows.extend(rows)
        except Exception as e:
            log(f"  ⚠️  Box parse error: {e}")
    return all_rows


def _parse_box(
    box,
    target_date: date,
    full_cache: dict[str, int],
    last_cache: dict[str, list],
    api_cache: dict[str, Optional[int]],
) -> list:
    rows = []

    team_els = box.select(".lineup__team")
    if len(team_els) < 2:
        return []

    away_raw  = team_els[0].get_text(strip=True)
    home_raw  = team_els[1].get_text(strip=True)
    away_abbr = norm_team(away_raw)
    home_abbr = norm_team(home_raw)

    if not away_abbr or not home_abbr:
        log(f"  ⚠️  Could not map teams: {away_raw} @ {home_raw}")
        return []

    game_pk = get_game_pk(home_abbr, away_abbr, target_date)
    if not game_pk:
        log(f"  ⚠️  No game_pk: {away_abbr} @ {home_abbr}")
        return []

    player_lists = box.select(".lineup__list")
    if len(player_lists) < 2:
        return []

    for side_idx, (side, team_abbr) in enumerate([("away", away_abbr), ("home", home_abbr)]):
        if side_idx >= len(player_lists):
            continue

        batter_els    = player_lists[side_idx].select("li")
        batting_order = 0

        for el in batter_els:
            raw = el.get_text(strip=True)
            if should_skip(raw):
                continue

            batting_order += 1
            if batting_order > 9:
                break

            position = None
            pos_el   = el.select_one(".lineup__pos")
            if pos_el:
                position    = pos_el.get_text(strip=True)
                player_name = clean_name(raw.replace(pos_el.get_text(strip=True), ""))
            else:
                player_name = clean_name(raw)

            if not player_name or len(player_name) < 2:
                batting_order -= 1
                continue

            # ── Resolve player ID from cache ──────────────────────────
            player_id = resolve_player_id(player_name, full_cache, last_cache, api_cache)

            rows.append({
                "game_pk":       game_pk,
                "official_date": target_date,
                "team":          team_abbr,
                "side":          side,
                "batting_order": batting_order,
                "player_id":     player_id,
                "player_name":   player_name,
                "position":      position,
            })

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    today = date.today()
    log("=" * 55)
    log(f"  load_rotowire_lineups.py  |  {today}")
    log("=" * 55)

    log("\nEnsuring DB is ready...")
    try:
        ensure_db_ready()
    except Exception as e:
        log(f"  ⚠️  DB setup warning (may already exist): {e}")

    # Build player ID cache from existing DB data — no API calls needed
    log("\nBuilding player ID cache from DB...")
    full_cache, last_cache = build_player_id_cache()
    api_cache: dict[str, Optional[int]] = {}
    log(f"  Cached {len(full_cache)} player name\u2192ID mappings")

    log("\nFetching RotoWire lineup page...")
    html = fetch_page()
    if not html:
        log("❌ Could not fetch RotoWire. Skipping.")
        return

    official_pks = get_official_game_pks(today)
    if official_pks:
        log(f"  Official MLB lineups exist for {len(official_pks)} games — will be preserved")

    log("\nParsing lineups...")
    rows = parse_lineups(html, today, full_cache, last_cache, api_cache)

    if not rows:
        log("⚠️  No lineup rows parsed.")
        return

    if official_pks:
        before = len(rows)
        rows   = [r for r in rows if r["game_pk"] not in official_pks]
        if before - len(rows):
            log(f"  Skipped {before - len(rows)} rows (official lineups exist)")

    resolved   = sum(1 for r in rows if r["player_id"] is not None)
    unresolved = len(rows) - resolved
    games      = {r["game_pk"] for r in rows}
    log(f"  Parsed {len(rows)} batter rows across {len(games)} games")
    log(f"  Player IDs resolved: {resolved}/{len(rows)} ({100*resolved//max(len(rows),1)}%)")
    if unresolved:
        missing_names = [r["player_name"] for r in rows if r["player_id"] is None]
        log(f"  Unresolved names: {missing_names[:10]}{'...' if len(missing_names)>10 else ''}")

    inserted = upsert_rows(rows)
    log(f"\n✅ Upserted {inserted} RotoWire lineup rows for {today}")


if __name__ == "__main__":
    main()