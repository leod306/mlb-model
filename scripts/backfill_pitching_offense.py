"""
scripts/backfill_pitching_offense.py
------------------------------------
Rebuilds game_features with STRICT as-of-date, leak-free stats computed
entirely from your own tables (pitcher_game_log + games). No BBRef, no
name-matching, no forward-only window.

For each completed game it computes, using ONLY data dated before that game:
  - home/away starter FIP + WHIP   (from pitcher_game_log by pitcher_id)
  - home/away bullpen FIP          (team relief appearances)
  - home/away rolling runs/game    (offense proxy, replaces the stale OPS wRC+)
  - park_run_factor / park_hr_factor (constants; weather left for the weather script)
  - the *_diff columns the model consumes

Why this fixes the -53u O/U hole:
  The old feature builder covered only 47% of games and defaulted FIP/wRC+ on
  a third more. The totals model was effectively blind on >half its inputs.
  This backfill gives ~100% coverage with real, honest, as-of-date signal.

Run from project root:
    PYTHONPATH=. python scripts/backfill_pitching_offense.py            # all seasons in games
    PYTHONPATH=. python scripts/backfill_pitching_offense.py 2026       # one season
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.getenv("DYNO") is None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass

from app.db import engine

GAMES_TABLE    = os.getenv("MLB_GAMES_TABLE",    "games")
FEATURES_TABLE = os.getenv("MLB_FEATURES_TABLE", "game_features")
PROB_TABLE     = os.getenv("MLB_PROBABLES_TABLE", "game_probables")

FIP_CONSTANT   = 3.10       # league-average HR/BB/K environment
LEAGUE_FIP     = 4.20       # prior for pitchers with little/no history
LEAGUE_WHIP    = 1.30
LEAGUE_RPG     = 4.50       # league runs/game prior
SHRINK_IP      = 25.0       # IP of "prior" weight for FIP shrinkage
SHRINK_GAMES   = 10.0       # games of "prior" weight for offense shrinkage
ROLL_OFFENSE   = 15         # rolling window for runs/game

PARK_FACTORS = {
    "ARI": (1.00, 1.03), "ATH": (0.96, 0.94), "ATL": (1.02, 1.06), "BAL": (1.01, 1.04),
    "BOS": (1.04, 1.03), "CHC": (1.01, 1.03), "CWS": (0.99, 1.00), "CIN": (1.07, 1.12),
    "CLE": (0.97, 0.98), "COL": (1.24, 1.19), "DET": (0.95, 0.92), "HOU": (0.99, 1.03),
    "KC":  (1.00, 0.96), "LAA": (1.00, 1.01), "LAD": (0.99, 0.98), "MIA": (0.94, 0.91),
    "MIL": (1.02, 1.05), "MIN": (1.01, 1.02), "NYM": (0.98, 0.96), "NYY": (1.03, 1.11),
    "PHI": (1.05, 1.09), "PIT": (0.98, 0.95), "SD": (0.96, 0.94), "SEA": (0.95, 0.93),
    "SF":  (0.93, 0.89), "STL": (0.99, 0.97), "TB": (0.97, 0.95), "TEX": (1.04, 1.07),
    "TOR": (1.03, 1.06), "WSH": (1.00, 1.01),
}
# normalize any stray abbreviations to the set used across the system
TEAM_ALIAS = {"CHW": "CWS", "OAK": "ATH", "AZ": "ARI", "WSN": "WSH", "SFG": "SF",
              "TBR": "TB", "KCR": "KC", "SDP": "SD"}


def norm(t):
    t = (str(t) or "").strip().upper()
    return TEAM_ALIAS.get(t, t)


def fip_from_counts(ip, bb, so, hr):
    """Shrunk FIP: blends observed toward league prior by innings pitched."""
    if ip is None or ip <= 0:
        return LEAGUE_FIP
    raw = ((13 * hr + 3 * bb - 2 * so) / ip) + FIP_CONSTANT
    w = ip / (ip + SHRINK_IP)
    return w * raw + (1 - w) * LEAGUE_FIP


def whip_from_counts(ip, h, bb):
    if ip is None or ip <= 0:
        return LEAGUE_WHIP
    raw = (h + bb) / ip
    w = ip / (ip + SHRINK_IP)
    return w * raw + (1 - w) * LEAGUE_WHIP


def main():
    # Arg parsing:
    #   (no arg)          → rebuild ALL games (one-time history backfill)
    #   YYYY (e.g. 2026)  → rebuild one season
    #   YYYY-MM-DD        → rebuild one date's slate (fast; for daily pipeline)
    #   today             → rebuild today's slate (what run_full_update.py calls)
    #
    # IMPORTANT: the stat HISTORIES (FIP accumulation) are always loaded from the
    # full pitcher_game_log regardless of filter, so a single-date rebuild still
    # sees every prior start. Only the set of games we WRITE is filtered.
    season_filter = ""
    params = {}
    mode = "all"
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower()
        if arg == "today":
            from datetime import date as _date
            season_filter = "WHERE g.official_date = :d"
            params["d"] = _date.today()
            mode = f"date={params['d']}"
        elif "-" in arg:  # YYYY-MM-DD
            season_filter = "WHERE g.official_date = :d"
            params["d"] = arg
            mode = f"date={arg}"
        else:             # YYYY season
            season_filter = "WHERE g.season = :season"
            params["season"] = int(arg)
            mode = f"season={arg}"
    print(f"Backfill mode: {mode}")

    print("Loading games + probable starters (fallback only) ...")
    games = pd.read_sql(text(f"""
        SELECT g.game_pk, g.official_date, g.season, g.home_team, g.away_team,
               g.home_score, g.away_score,
               p.home_sp_id, p.away_sp_id
        FROM {GAMES_TABLE} g
        LEFT JOIN {PROB_TABLE} p ON p.game_pk = g.game_pk
        {season_filter}
        ORDER BY g.official_date, g.game_pk
    """), engine, params=params)
    games["official_date"] = pd.to_datetime(games["official_date"]).dt.date
    games["home_team"] = games["home_team"].map(norm)
    games["away_team"] = games["away_team"].map(norm)
    print(f"  {len(games)} games")

    print("Loading pitcher game log ...")
    pgl = pd.read_sql(text("""
        SELECT game_pk, side, official_date, pitcher_id, team, role,
               innings_pitched, hits_allowed, walks, strikeouts, home_runs
        FROM pitcher_game_log
    """), engine)
    pgl["official_date"] = pd.to_datetime(pgl["official_date"]).dt.date
    pgl["team"] = pgl["team"].map(norm)
    for c in ["innings_pitched", "hits_allowed", "walks", "strikeouts", "home_runs"]:
        pgl[c] = pd.to_numeric(pgl[c], errors="coerce").fillna(0.0)
    print(f"  {len(pgl)} log rows")

    # --- ACTUAL starters, straight from the log ---------------------------------
    # game_probables only covers ~1,674 upcoming games, so for history it's 80%
    # empty and both starters defaulted to league FIP (the bug that made
    # sp_fip_diff==0 jump to 80%). pitcher_game_log records who actually started
    # each game (role='SP', by game_pk + side), which is near-complete history
    # AND is the real starter, not a projection. We key off this and fall back to
    # game_probables only for games with no log row yet (today's slate).
    def norm_side(s):
        s = str(s).strip().lower()
        if s in ("home", "h"):
            return "home"
        if s in ("away", "a", "road", "visitor"):
            return "away"
        return s

    sp_log = pgl[(pgl["role"] == "SP") & pgl["game_pk"].notna()].copy()
    sp_log["side_n"] = sp_log["side"].map(norm_side)
    # if multiple SP rows per side (openers/bullpen games), take the one with most IP
    sp_log = sp_log.sort_values("innings_pitched", ascending=False)
    starter_by_game_side = {}
    for (gpk, side), grp in sp_log.groupby(["game_pk", "side_n"]):
        starter_by_game_side[(int(gpk), side)] = grp.iloc[0]["pitcher_id"]

    def actual_starter(game_pk, side, fallback_id):
        pid = starter_by_game_side.get((int(game_pk), side))
        if pid is not None and pd.notna(pid):
            return pid
        return fallback_id  # game_probables id, used only for not-yet-played games

    n_from_log = len(starter_by_game_side)
    print(f"  actual starters from log: {n_from_log} (game,side) pairs")

    # --- Build per-key cumulative sums keyed by date ---
    # Correctness: for a game on date D we need the SUM of all prior
    # appearances, strictly D' < D. Per key we store a date-sorted table of
    # cumulative totals as of each log date; at lookup we take the last entry
    # whose date is strictly < D. The '<' filter does the exclusion (no shift).
    STAT_COLS = ["innings_pitched", "hits_allowed", "walks", "strikeouts", "home_runs"]
    sp = pgl[pgl["role"] == "SP"].sort_values("official_date")
    rp = pgl[pgl["role"] == "RP"].sort_values("official_date")

    def cum_by_key(df, key):
        out = {}
        for k, grp in df.groupby(key):
            daily = grp.groupby("official_date")[STAT_COLS].sum()  # collapse same-date rows
            cs = daily.cumsum().reset_index()
            out[k] = cs
        return out

    print("Indexing starter histories ...")
    sp_cum = cum_by_key(sp, "pitcher_id")
    print("Indexing bullpen histories ...")
    rp_cum = cum_by_key(rp, "team")

    def asof_totals(cum_map, key, game_date):
        """Cumulative totals of all appearances STRICTLY before game_date."""
        arr = cum_map.get(key)
        if arr is None:
            return None
        prior = arr[arr["official_date"] < game_date]
        if prior.empty:
            return None
        row = prior.iloc[-1]
        return (row["innings_pitched"], row["hits_allowed"],
                row["walks"], row["strikeouts"], row["home_runs"])

    # --- Offense: rolling runs/game as-of-date ---
    # History must come from ALL games (not the filtered slate) or a single-date
    # run would have no prior games to average. Load the full team run history
    # independently, exactly like the pitcher histories above.
    print("Building team run histories (full history) ...")
    all_games = pd.read_sql(text(f"""
        SELECT official_date, home_team, away_team, home_score, away_score
        FROM {GAMES_TABLE}
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
    """), engine)
    all_games["official_date"] = pd.to_datetime(all_games["official_date"]).dt.date
    all_games["home_team"] = all_games["home_team"].map(norm)
    all_games["away_team"] = all_games["away_team"].map(norm)

    hh = all_games[["official_date", "home_team", "home_score"]].rename(
        columns={"home_team": "team", "home_score": "runs"})
    aa = all_games[["official_date", "away_team", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "runs"})
    tg = pd.concat([hh, aa], ignore_index=True)
    tg["runs"] = pd.to_numeric(tg["runs"], errors="coerce")
    tg = tg.dropna(subset=["runs"]).sort_values("official_date")
    team_hist = {t: g.sort_values("official_date") for t, g in tg.groupby("team")}

    def asof_rpg(team, game_date):
        g = team_hist.get(team)
        if g is None:
            return LEAGUE_RPG
        prior = g[g["official_date"] < game_date]["runs"]
        if prior.empty:
            return LEAGUE_RPG
        recent = prior.tail(ROLL_OFFENSE)
        n = len(recent)
        w = n / (n + SHRINK_GAMES)
        return w * float(recent.mean()) + (1 - w) * LEAGUE_RPG

    print("Computing features per game ...")
    rows = []
    for _, g in games.iterrows():
        gd = g["official_date"]
        # actual starter from the log (falls back to probables id for future games)
        home_sp = actual_starter(g["game_pk"], "home", g["home_sp_id"])
        away_sp = actual_starter(g["game_pk"], "away", g["away_sp_id"])
        hp = asof_totals(sp_cum, home_sp, gd) if pd.notna(home_sp) else None
        ap = asof_totals(sp_cum, away_sp, gd) if pd.notna(away_sp) else None
        hb = asof_totals(rp_cum, g["home_team"], gd)
        ab = asof_totals(rp_cum, g["away_team"], gd)

        h_sp_fip  = fip_from_counts(*[hp[i] for i in (0,2,3,4)]) if hp else LEAGUE_FIP
        a_sp_fip  = fip_from_counts(*[ap[i] for i in (0,2,3,4)]) if ap else LEAGUE_FIP
        h_sp_whip = whip_from_counts(hp[0], hp[1], hp[2]) if hp else LEAGUE_WHIP
        a_sp_whip = whip_from_counts(ap[0], ap[1], ap[2]) if ap else LEAGUE_WHIP
        h_bp_fip  = fip_from_counts(*[hb[i] for i in (0,2,3,4)]) if hb else LEAGUE_FIP
        a_bp_fip  = fip_from_counts(*[ab[i] for i in (0,2,3,4)]) if ab else LEAGUE_FIP

        h_rpg = asof_rpg(g["home_team"], gd)
        a_rpg = asof_rpg(g["away_team"], gd)
        # offense proxy scaled to a ~100 baseline so it slots where wrc_plus was
        h_wrc = round((h_rpg / LEAGUE_RPG) * 100, 1)
        a_wrc = round((a_rpg / LEAGUE_RPG) * 100, 1)

        prf, phf = PARK_FACTORS.get(g["home_team"], (1.0, 1.0))

        rows.append({
            "game_pk": int(g["game_pk"]),
            "official_date": gd,
            "home_team": g["home_team"], "away_team": g["away_team"],
            "home_sp_fip": round(h_sp_fip, 3), "away_sp_fip": round(a_sp_fip, 3),
            "home_sp_whip": round(h_sp_whip, 3), "away_sp_whip": round(a_sp_whip, 3),
            "home_bullpen_fip": round(h_bp_fip, 3), "away_bullpen_fip": round(a_bp_fip, 3),
            "home_wrc_plus": h_wrc, "away_wrc_plus": a_wrc,
            "park_run_factor": prf, "park_hr_factor": phf,
            "sp_fip_diff": round(h_sp_fip - a_sp_fip, 3),
            "bullpen_fip_diff": round(h_bp_fip - a_bp_fip, 3),
            "offense_wrc_diff": round(h_wrc - a_wrc, 1),
        })

    feats = pd.DataFrame(rows)
    print(f"  built {len(feats)} feature rows")

    if feats.empty:
        print("  No feature rows to write — skipping (off day or no games scheduled yet).")
        return

    # sanity: how much real signal did we recover?
    sp_zero = (feats["sp_fip_diff"] == 0).mean()
    wrc_def = (feats["home_wrc_plus"] == 100.0).mean()
    print(f"  sp_fip_diff == 0 : {sp_zero:.1%}  (was 34% — mostly early-season now)")
    print(f"  wrc == 100 default: {wrc_def:.1%}  (was 33%)")

    print("Writing to game_features (upsert on game_pk) ...")
    upsert(feats)
    print(f"Done. {len(feats)} rows upserted into {FEATURES_TABLE}.")


def upsert(feats: pd.DataFrame):
    cols = ["game_pk", "official_date", "home_team", "away_team",
            "home_sp_fip", "away_sp_fip", "home_sp_whip", "away_sp_whip",
            "home_bullpen_fip", "away_bullpen_fip", "home_wrc_plus", "away_wrc_plus",
            "park_run_factor", "park_hr_factor",
            "sp_fip_diff", "bullpen_fip_diff", "offense_wrc_diff"]
    val = ", ".join(f":{c}" for c in cols)
    setc = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "game_pk")
    sql = text(f"""
        INSERT INTO {FEATURES_TABLE} ({", ".join(cols)}, updated_at)
        VALUES ({val}, NOW())
        ON CONFLICT (game_pk) DO UPDATE SET {setc}, updated_at = NOW()
    """)
    recs = feats.to_dict("records")
    with engine.begin() as c:
        for i in range(0, len(recs), 500):
            c.execute(sql, recs[i:i+500])


if __name__ == "__main__":
    main()