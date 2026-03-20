from __future__ import annotations

import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")
PROB_TABLE  = os.getenv("MLB_PROBABLES_TABLE", "game_probables")
PRED_TABLE  = os.getenv("MLB_PREDICTIONS_TABLE", "predictions")

SEASON             = int(os.getenv("MLB_SEASON", "2026"))
GAME_TYPES         = os.getenv("MLB_GAME_TYPES", "S,R")
WINDOW_DAYS        = int(os.getenv("WINDOW_DAYS", "30"))
DEFAULT_TOTAL_LINE = float(os.getenv("DEFAULT_TOTAL_LINE", "8.5"))

SLEEP        = float(os.getenv("REQUEST_SLEEP_SECONDS", "0.05"))
HTTP_TIMEOUT = 25

ML_DIR             = os.path.join(PROJECT_ROOT, "ml")
WIN_MODEL_PATH     = os.getenv("MLB_WIN_MODEL_PATH",   os.path.join(ML_DIR, "win_model.pkl"))
RUN_MODEL_PATH     = os.getenv("MLB_RUN_MODEL_PATH",   os.path.join(ML_DIR, "run_model.pkl"))
TOTAL_MODEL_PATH   = os.getenv("MLB_TOTAL_MODEL_PATH", os.path.join(ML_DIR, "total_model.pkl"))
FALLBACK_MODEL_PATH = os.getenv("MLB_MODEL_PATH",      os.path.join(ML_DIR, "mlb_model.pkl"))

MLB_BASE = "https://statsapi.mlb.com/api/v1"
MLB_FEED = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)


def get_json(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def utc_dt(iso_z: str) -> datetime:
    return datetime.fromisoformat(iso_z.replace("Z", "+00:00")).astimezone(timezone.utc)


def _safe_float(val) -> Optional[float]:
    """Return a JSON-safe float or None. Handles nan, inf, None, numpy scalars."""
    try:
        if val is None:
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def implied_moneyline(p: float) -> Optional[int]:
    if p is None:
        return None
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p <= 0.0 or p >= 1.0:
        return None
    if p >= 0.5:
        return int(round(-100 * (p / (1 - p))))
    return int(round(100 * ((1 - p) / p)))


# ---------------------------------------------------------------------------
# Table setup
# ---------------------------------------------------------------------------

def ensure_tables(cur):
    cur.execute(f"CREATE TABLE IF NOT EXISTS {GAMES_TABLE} (game_pk BIGINT PRIMARY KEY);")
    cur.execute(f"CREATE TABLE IF NOT EXISTS {PROB_TABLE}  (game_pk BIGINT PRIMARY KEY);")
    cur.execute(f"CREATE TABLE IF NOT EXISTS {PRED_TABLE}  (game_pk BIGINT PRIMARY KEY);")

    for col, typedef in [
        ("official_date",   "DATE"),
        ("game_date_utc",   "TIMESTAMPTZ"),
        ("season",          "INT"),
        ("game_type",       "TEXT"),
        ("status",          "TEXT"),
        ("home_team",       "TEXT"),
        ("away_team",       "TEXT"),
        ("home_team_id",    "INT"),
        ("away_team_id",    "INT"),
    ]:
        cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS {col} {typedef};")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{GAMES_TABLE}_official_date ON {GAMES_TABLE}(official_date);")

    for col, typedef in [
        ("official_date",  "DATE"),
        ("home_sp_id",     "INT"),
        ("away_sp_id",     "INT"),
        ("home_sp_name",   "TEXT"),
        ("away_sp_name",   "TEXT"),
        ("status",         "TEXT"),
    ]:
        cur.execute(f"ALTER TABLE {PROB_TABLE} ADD COLUMN IF NOT EXISTS {col} {typedef};")
    cur.execute(f"ALTER TABLE {PROB_TABLE} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{PROB_TABLE}_official_date ON {PROB_TABLE}(official_date);")

    for col, typedef in [
        ("official_date",    "DATE"),
        ("home_team",        "TEXT"),
        ("away_team",        "TEXT"),
        ("prediction",       "INT"),
        ("win_probability",  "DOUBLE PRECISION"),
        ("home_win_prob",    "DOUBLE PRECISION"),
        ("away_win_prob",    "DOUBLE PRECISION"),
        ("home_ml_implied",  "INT"),
        ("away_ml_implied",  "INT"),
        ("run_diff_pred",    "DOUBLE PRECISION"),
        ("total_runs_pred",  "DOUBLE PRECISION"),
        ("ml_pick",          "TEXT"),
        ("runline_pick",     "TEXT"),
        ("ou_pick",          "TEXT"),
    ]:
        cur.execute(f"ALTER TABLE {PRED_TABLE} ADD COLUMN IF NOT EXISTS {col} {typedef};")
    cur.execute(f"ALTER TABLE {PRED_TABLE} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{PRED_TABLE}_official_date ON {PRED_TABLE}(official_date);")


# ---------------------------------------------------------------------------
# Schedule + probables upsert
# ---------------------------------------------------------------------------

def team_map(season: int) -> Dict[int, str]:
    data = get_json(f"{MLB_BASE}/teams", {"sportId": 1, "season": season})
    out: Dict[int, str] = {}
    for t in data.get("teams", []) or []:
        tid  = t.get("id")
        abbr = t.get("abbreviation") or t.get("teamCode") or t.get("fileCode") or t.get("name")
        if tid and abbr:
            out[int(tid)] = str(abbr)
    return out


def upsert_schedule(cur, start: date, end: date, tmap: Dict[int, str]) -> int:
    data = get_json(
        f"{MLB_BASE}/schedule",
        {
            "sportId":   1,
            "season":    SEASON,
            "gameTypes": GAME_TYPES,
            "startDate": start.isoformat(),
            "endDate":   end.isoformat(),
            "hydrate":   "team,venue",
        },
    )

    rows = []
    for d in data.get("dates", []) or []:
        for g in d.get("games", []) or []:
            home = (((g.get("teams") or {}).get("home") or {}).get("team") or {})
            away = (((g.get("teams") or {}).get("away") or {}).get("team") or {})
            if not home.get("id") or not away.get("id"):
                continue
            rows.append((
                int(g["gamePk"]),
                (g.get("officialDate") or d.get("date")),
                utc_dt(g["gameDate"]),
                SEASON,
                str(g.get("gameType") or ""),
                str((g.get("status") or {}).get("detailedState") or ""),
                int(home["id"]),
                int(away["id"]),
                str(home.get("abbreviation") or tmap.get(int(home["id"])) or home.get("name") or home["id"]),
                str(away.get("abbreviation") or tmap.get(int(away["id"])) or away.get("name") or away["id"]),
                datetime.now(timezone.utc),
            ))

    if not rows:
        return 0

    sql = f"""
    INSERT INTO {GAMES_TABLE} (
      game_pk, official_date, game_date_utc, season, game_type, status,
      home_team_id, away_team_id, home_team, away_team, updated_at
    )
    VALUES %s
    ON CONFLICT (game_pk) DO UPDATE SET
      official_date=EXCLUDED.official_date,
      game_date_utc=EXCLUDED.game_date_utc,
      season=EXCLUDED.season,
      game_type=EXCLUDED.game_type,
      status=EXCLUDED.status,
      home_team_id=EXCLUDED.home_team_id,
      away_team_id=EXCLUDED.away_team_id,
      home_team=EXCLUDED.home_team,
      away_team=EXCLUDED.away_team,
      updated_at=NOW();
    """
    execute_values(cur, sql, rows, page_size=1000)
    return len(rows)


def parse_probables(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str], Optional[str]]:
    gd     = payload.get("gameData") or {}
    prob   = gd.get("probablePitchers") or {}
    players = gd.get("players") or {}
    status = (gd.get("status") or {}).get("detailedState") or None

    def one(side: str) -> Tuple[Optional[int], Optional[str]]:
        p   = prob.get(side) or {}
        pid = p.get("id")
        if not pid:
            return None, None
        pid = int(pid)
        pl  = players.get(f"ID{pid}") or {}
        name = pl.get("fullName") or p.get("fullName")
        return pid, name

    hid, hname = one("home")
    aid, aname = one("away")
    return hid, hname, aid, aname, status


def upsert_probables(cur, start: date, end: date) -> int:
    cur.execute(
        f"""
        SELECT game_pk, official_date
        FROM {GAMES_TABLE}
        WHERE official_date BETWEEN %s AND %s
        ORDER BY official_date, game_pk
        """,
        (start, end),
    )
    games = cur.fetchall()

    rows = []
    now  = datetime.now(timezone.utc)

    for game_pk, off_date in games:
        try:
            payload = get_json(MLB_FEED.format(gamePk=int(game_pk)))
        except Exception:
            continue

        hid, hname, aid, aname, status = parse_probables(payload)
        rows.append((int(game_pk), off_date, hid, aid, hname, aname, status, now))

        if SLEEP:
            time.sleep(SLEEP)

    if not rows:
        return 0

    sql = f"""
    INSERT INTO {PROB_TABLE} (
      game_pk, official_date,
      home_sp_id, away_sp_id,
      home_sp_name, away_sp_name,
      status, updated_at
    )
    VALUES %s
    ON CONFLICT (game_pk) DO UPDATE SET
      official_date=EXCLUDED.official_date,
      home_sp_id=EXCLUDED.home_sp_id,
      away_sp_id=EXCLUDED.away_sp_id,
      home_sp_name=EXCLUDED.home_sp_name,
      away_sp_name=EXCLUDED.away_sp_name,
      status=EXCLUDED.status,
      updated_at=NOW();
    """
    execute_values(cur, sql, rows, page_size=1000)
    return len(rows)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models() -> Tuple[Any, Any, Any]:
    win = joblib.load(WIN_MODEL_PATH) if os.path.exists(WIN_MODEL_PATH) else joblib.load(FALLBACK_MODEL_PATH)
    if not os.path.exists(RUN_MODEL_PATH):
        raise FileNotFoundError(f"Missing run model: {RUN_MODEL_PATH}")
    if not os.path.exists(TOTAL_MODEL_PATH):
        raise FileNotFoundError(f"Missing total model: {TOTAL_MODEL_PATH}")
    run   = joblib.load(RUN_MODEL_PATH)
    total = joblib.load(TOTAL_MODEL_PATH)
    return win, run, total


def align_to_model(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Select and order only the columns the model was trained on.
    Missing columns are filled with 0. Extra columns are dropped.
    """
    feat_cols = getattr(model, "feature_names_in_", None)
    drop = {
        "game_pk", "official_date", "home_team", "away_team",
        "home_team_id", "away_team_id", "home_sp_id", "away_sp_id",
        "home_sp_name", "away_sp_name",
    }
    if feat_cols is None:
        return df[[c for c in df.columns if c not in drop]].copy()

    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0
    return X


# ---------------------------------------------------------------------------
# Feature builder  ← THIS IS THE KEY FIX
# ---------------------------------------------------------------------------

def build_features_for_date(cur, target: date) -> pd.DataFrame:
    """
    Build the 8 model features for every game on `target` date.

    Features (matching train_model.py):
        era_diff, whip_diff,
        home_last10_runs_scored, away_last10_runs_scored,
        home_last10_runs_allowed, away_last10_runs_allowed,
        home_last10_run_diff, away_last10_run_diff
    """
    cur.execute(
        f"""
        SELECT
            g.game_pk,
            g.official_date,
            g.home_team,
            g.away_team,
            g.home_team_id,
            g.away_team_id,
            p.home_sp_id,
            p.away_sp_id,
            p.home_sp_name,
            p.away_sp_name,
            -- rolling team stats (populated by build_dataset / rolling update)
            g.home_last10_runs_scored,
            g.away_last10_runs_scored,
            g.home_last10_runs_allowed,
            g.away_last10_runs_allowed,
            g.home_last10_run_diff,
            g.away_last10_run_diff,
            -- pitcher ERA/WHIP joined from pitchers table
            hp.era   AS home_era,
            hp.whip  AS home_whip,
            ap.era   AS away_era,
            ap.whip  AS away_whip
        FROM {GAMES_TABLE} g
        LEFT JOIN {PROB_TABLE} p
            ON p.game_pk = g.game_pk
        LEFT JOIN pitchers hp
            ON LOWER(TRIM(hp.pitcher_name)) = LOWER(TRIM(p.home_sp_name))
            AND hp.season = %s
        LEFT JOIN pitchers ap
            ON LOWER(TRIM(ap.pitcher_name)) = LOWER(TRIM(p.away_sp_name))
            AND ap.season = %s
        WHERE g.official_date = %s
        ORDER BY g.game_pk
        """,
        (SEASON, SEASON, target),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "game_pk", "official_date", "home_team", "away_team",
        "home_team_id", "away_team_id",
        "home_sp_id", "away_sp_id", "home_sp_name", "away_sp_name",
        "home_last10_runs_scored", "away_last10_runs_scored",
        "home_last10_runs_allowed", "away_last10_runs_allowed",
        "home_last10_run_diff", "away_last10_run_diff",
        "home_era", "home_whip", "away_era", "away_whip",
    ])

    # League-average ERA/WHIP fallback for unknown pitchers
    cur.execute(
        "SELECT AVG(era), AVG(whip) FROM pitchers WHERE season = %s",
        (SEASON,),
    )
    row = cur.fetchone()
    league_era  = float(row[0]) if row and row[0] is not None else 4.20
    league_whip = float(row[1]) if row and row[1] is not None else 1.30

    df["home_era"]  = pd.to_numeric(df["home_era"],  errors="coerce").fillna(league_era)
    df["away_era"]  = pd.to_numeric(df["away_era"],  errors="coerce").fillna(league_era)
    df["home_whip"] = pd.to_numeric(df["home_whip"], errors="coerce").fillna(league_whip)
    df["away_whip"] = pd.to_numeric(df["away_whip"], errors="coerce").fillna(league_whip)

    df["era_diff"]  = df["home_era"]  - df["away_era"]
    df["whip_diff"] = df["home_whip"] - df["away_whip"]

    # Rolling stats: fill with 0.0 when not yet available (spring training / early season)
    rolling_cols = [
        "home_last10_runs_scored", "away_last10_runs_scored",
        "home_last10_runs_allowed", "away_last10_runs_allowed",
        "home_last10_run_diff",    "away_last10_run_diff",
    ]
    for col in rolling_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["has_home_sp"] = df["home_sp_id"].notna().astype(int)
    df["has_away_sp"] = df["away_sp_id"].notna().astype(int)

    return df


# ---------------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------------

def save_predictions(cur, out: pd.DataFrame) -> int:
    if out.empty:
        return 0

    rows = list(out[[
        "game_pk", "official_date", "away_team", "home_team",
        "prediction", "win_probability",
        "home_win_prob", "away_win_prob",
        "home_ml_implied", "away_ml_implied",
        "run_diff_pred", "total_runs_pred",
        "ml_pick", "runline_pick", "ou_pick",
    ]].itertuples(index=False, name=None))

    rows2 = [r + (datetime.now(timezone.utc),) for r in rows]

    sql = f"""
    INSERT INTO {PRED_TABLE} (
      game_pk, official_date, away_team, home_team,
      prediction, win_probability,
      home_win_prob, away_win_prob,
      home_ml_implied, away_ml_implied,
      run_diff_pred, total_runs_pred,
      ml_pick, runline_pick, ou_pick,
      updated_at
    )
    VALUES %s
    ON CONFLICT (game_pk) DO UPDATE SET
      official_date=EXCLUDED.official_date,
      away_team=EXCLUDED.away_team,
      home_team=EXCLUDED.home_team,
      prediction=EXCLUDED.prediction,
      win_probability=EXCLUDED.win_probability,
      home_win_prob=EXCLUDED.home_win_prob,
      away_win_prob=EXCLUDED.away_win_prob,
      home_ml_implied=EXCLUDED.home_ml_implied,
      away_ml_implied=EXCLUDED.away_ml_implied,
      run_diff_pred=EXCLUDED.run_diff_pred,
      total_runs_pred=EXCLUDED.total_runs_pred,
      ml_pick=EXCLUDED.ml_pick,
      runline_pick=EXCLUDED.runline_pick,
      ou_pick=EXCLUDED.ou_pick,
      updated_at=NOW();
    """
    execute_values(cur, sql, rows2, page_size=1000)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("GAMES_TABLE  =", GAMES_TABLE)
    print("PROB_TABLE   =", PROB_TABLE)
    print("PRED_TABLE   =", PRED_TABLE)

    win_model, run_model, total_model = load_models()

    today = datetime.now(timezone.utc).date()
    start = today
    end   = today + timedelta(days=WINDOW_DAYS)

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_tables(cur)

            tmap    = team_map(SEASON)
            n_sched = upsert_schedule(cur, start, end, tmap)
            n_prob  = upsert_probables(cur, start, end)

            cur.execute(
                f"SELECT MIN(official_date) FROM {GAMES_TABLE} WHERE official_date >= %s",
                (today,),
            )
            slate = cur.fetchone()[0]
            if not slate:
                c.commit()
                print("No games found in DB for today or future window.")
                return

            base = build_features_for_date(cur, slate)
            if base.empty:
                c.commit()
                print(f"No games found for slate date {slate}.")
                return

            X_win = align_to_model(base, win_model)
            X_run = align_to_model(base, run_model)
            X_tot = align_to_model(base, total_model)

            if hasattr(win_model, "predict_proba"):
                p = win_model.predict_proba(X_win)
                home_prob_raw = p[:, 1] if p.shape[1] > 1 else p[:, 0]
            else:
                home_prob_raw = win_model.predict(X_win)

            # Sanitize all model outputs — nan/inf → safe fallback
            home_prob  = [_safe_float(x) or 0.5  for x in home_prob_raw]
            away_prob  = [round(1.0 - hp, 6)      for hp in home_prob]
            prediction = [1 if hp >= 0.5 else 0   for hp in home_prob]
            run_diff   = [_safe_float(x) or 0.0   for x in run_model.predict(X_run)]
            total_runs = [_safe_float(x) or 0.0   for x in total_model.predict(X_tot)]

            ml_pick       = ["HOME" if hp >= 0.5  else "AWAY"      for hp in home_prob]
            runline_pick  = ["HOME -1.5" if rd >= 1.5 else "AWAY +1.5" for rd in run_diff]
            ou_pick       = ["OVER" if tr >= DEFAULT_TOTAL_LINE else "UNDER" for tr in total_runs]

            out = pd.DataFrame({
                "game_pk":        base["game_pk"].astype(int),
                "official_date":  pd.to_datetime(base["official_date"]).dt.date,
                "away_team":      base["away_team"].astype(str),
                "home_team":      base["home_team"].astype(str),
                "prediction":     prediction,
                "win_probability": home_prob,
                "home_win_prob":  home_prob,
                "away_win_prob":  away_prob,
                "home_ml_implied": [implied_moneyline(x) for x in home_prob],
                "away_ml_implied": [implied_moneyline(x) for x in away_prob],
                "run_diff_pred":  run_diff,
                "total_runs_pred": total_runs,
                "ml_pick":        ml_pick,
                "runline_pick":   runline_pick,
                "ou_pick":        ou_pick,
            })

            n_saved = save_predictions(cur, out)
            c.commit()

            print(f"Schedule upserted:   {n_sched}")
            print(f"Probables upserted:  {n_prob}")
            print(f"Slate date:          {slate}")
            print(f"Predictions saved:   {n_saved}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()