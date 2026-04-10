from __future__ import annotations

import math
from datetime import date
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text

from app.db import engine

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


# ---------------------------------------------------------------------------
# JSON-safe type helpers
# ---------------------------------------------------------------------------

def safe_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def safe_int(val) -> Optional[int]:
    try:
        if val is None:
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else int(f)
    except (TypeError, ValueError):
        return None


def safe_str(val) -> Optional[str]:
    if val is None:
        return None
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    return str(val)


def team_logo_by_id(team_id) -> str:
    tid = safe_int(team_id)
    if tid is None:
        return ""
    return f"https://www.mlbstatic.com/team-logos/team-cap-on-dark/{tid}.svg"


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
def homepage(request: Request, game_date: Optional[str] = Query(default=None)):
    if not game_date:
        game_date = date.today().isoformat()
    return templates.TemplateResponse("index.html", {"request": request, "game_date": game_date})


@router.get("/matchups", response_class=HTMLResponse)
def matchups_page(request: Request, game_date: Optional[str] = Query(default=None)):
    if not game_date:
        game_date = date.today().isoformat()
    return templates.TemplateResponse("matchups.html", {"request": request, "game_date": game_date})


@router.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request, game_date: Optional[str] = Query(default=None)):
    if not game_date:
        game_date = date.today().isoformat()
    return templates.TemplateResponse("predictions.html", {"request": request, "game_date": game_date})


# ---------------------------------------------------------------------------
# API: games
# ---------------------------------------------------------------------------

@router.get("/api/games")
def api_games(game_date: str = Query(..., description="YYYY-MM-DD")):
    sql = text(
        """
        SELECT
            g.game_pk,
            g.official_date,
            g.game_date_utc,
            g.home_team,
            g.away_team,
            COALESCE(gp.home_sp_name, g.home_starting_pitcher) AS home_starting_pitcher,
            COALESCE(gp.away_sp_name, g.away_starting_pitcher) AS away_starting_pitcher,
            COALESCE(gp.home_sp_id, NULL) AS home_sp_id,
            COALESCE(gp.away_sp_id, NULL) AS away_sp_id,
            g.home_team_id,
            g.away_team_id
        FROM games g
        LEFT JOIN game_probables gp ON gp.game_pk = g.game_pk
        WHERE CAST(g.official_date AS DATE) = :d
        ORDER BY g.game_date_utc ASC NULLS LAST, g.game_pk
        """
    )

    df = pd.read_sql(sql, engine, params={"d": game_date})

    games: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        games.append({
            "game_pk":               safe_int(r.get("game_pk")),
            "game_date":             safe_str(r.get("official_date")),
            "game_date_utc":         safe_str(r.get("game_date_utc")),
            "home_team":             safe_str(r.get("home_team")),
            "away_team":             safe_str(r.get("away_team")),
            "home_logo":             team_logo_by_id(r.get("home_team_id")),
            "away_logo":             team_logo_by_id(r.get("away_team_id")),
            "home_starting_pitcher": safe_str(r.get("home_starting_pitcher")),
            "away_starting_pitcher": safe_str(r.get("away_starting_pitcher")),
            "home_sp_id":            safe_int(r.get("home_sp_id")),
            "away_sp_id":            safe_int(r.get("away_sp_id")),
            "home_team_id":          safe_int(r.get("home_team_id")),
            "away_team_id":          safe_int(r.get("away_team_id")),
        })

    return {"ok": True, "count": len(games), "games": games}


# ---------------------------------------------------------------------------
# API: predictions
# ---------------------------------------------------------------------------

@router.get("/api/predict/today")
def api_predict_today(game_date: str = Query(..., description="YYYY-MM-DD")):
    sql = text(
        """
        SELECT
            g.game_pk,
            g.official_date,
            g.game_date_utc,
            g.home_team,
            g.away_team,

            COALESCE(gp.home_sp_name, g.home_starting_pitcher) AS home_starting_pitcher,
            COALESCE(gp.away_sp_name, g.away_starting_pitcher) AS away_starting_pitcher,
            COALESCE(gp.home_sp_id, NULL) AS home_sp_id,
            COALESCE(gp.away_sp_id, NULL) AS away_sp_id,

            g.home_team_id,
            g.away_team_id,

            -- MODEL
            p.home_win_prob,
            p.away_win_prob,
            p.home_win_prob_lo,
            p.home_win_prob_hi,
            p.home_win_prob_std,

            p.home_ml_implied,
            p.away_ml_implied,

            p.run_diff_pred,
            p.run_diff_lo,
            p.run_diff_hi,
            p.run_diff_std,

            p.total_runs_pred,
            p.total_runs_lo,
            p.total_runs_hi,
            p.total_runs_std,

            p.ml_pick,
            p.runline_pick,
            p.ou_pick,

            -- TOP PLAYS
            p.play_rank,
            p.play_type,
            p.play_score,
            p.play_detail,

            -- MARKET
            p.market_home_prob,
            p.market_away_prob,
            p.market_total_line,
            p.market_home_ml,
            p.market_away_ml,
            p.best_home_ml,
            p.best_away_ml,
            p.model_edge

        FROM games g
        LEFT JOIN predictions p     ON p.game_pk = g.game_pk
        LEFT JOIN game_probables gp ON gp.game_pk = g.game_pk
        WHERE CAST(g.official_date AS DATE) = :d
        ORDER BY g.game_date_utc ASC NULLS LAST, g.game_pk
        """
    )

    df = pd.read_sql(sql, engine, params={"d": game_date})

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        has_prediction = (
            pd.notna(row.get("ml_pick")) or pd.notna(row.get("home_win_prob"))
        )

        results.append({
            "id":                    safe_int(row.get("game_pk")),
            "game_pk":               safe_int(row.get("game_pk")),
            "game_date":             safe_str(row.get("official_date")),
            "game_date_utc":         safe_str(row.get("game_date_utc")),

            "home_team":             safe_str(row.get("home_team")),
            "away_team":             safe_str(row.get("away_team")),
            "home_logo":             team_logo_by_id(row.get("home_team_id")),
            "away_logo":             team_logo_by_id(row.get("away_team_id")),

            "home_starting_pitcher": safe_str(row.get("home_starting_pitcher")),
            "away_starting_pitcher": safe_str(row.get("away_starting_pitcher")),
            "home_sp_id":            safe_int(row.get("home_sp_id")),
            "away_sp_id":            safe_int(row.get("away_sp_id")),

            "ok":                    has_prediction,

            # Model outputs
            "home_win_prob":         safe_float(row.get("home_win_prob")),
            "away_win_prob":         safe_float(row.get("away_win_prob")),
            "home_win_prob_lo":      safe_float(row.get("home_win_prob_lo")),
            "home_win_prob_hi":      safe_float(row.get("home_win_prob_hi")),
            "home_win_prob_std":     safe_float(row.get("home_win_prob_std")),

            "home_moneyline_fair":   safe_int(row.get("home_ml_implied")),
            "away_moneyline_fair":   safe_int(row.get("away_ml_implied")),

            "pred_total_runs":       safe_float(row.get("total_runs_pred")),
            "total_runs_lo":         safe_float(row.get("total_runs_lo")),
            "total_runs_hi":         safe_float(row.get("total_runs_hi")),
            "total_runs_std":        safe_float(row.get("total_runs_std")),

            "pred_run_diff":         safe_float(row.get("run_diff_pred")),
            "run_diff_lo":           safe_float(row.get("run_diff_lo")),
            "run_diff_hi":           safe_float(row.get("run_diff_hi")),
            "run_diff_std":          safe_float(row.get("run_diff_std")),

            "ml_pick":               safe_str(row.get("ml_pick")),
            "runline_pick":          safe_str(row.get("runline_pick")),
            "ou_pick":               safe_str(row.get("ou_pick")),

            # Top plays
            "play_rank":             safe_int(row.get("play_rank")),
            "play_type":             safe_str(row.get("play_type")),
            "play_score":            safe_float(row.get("play_score")),
            "play_detail":           safe_str(row.get("play_detail")),

            # Market odds
            "market_home_prob":      safe_float(row.get("market_home_prob")),
            "market_away_prob":      safe_float(row.get("market_away_prob")),
            "market_total_line":     safe_float(row.get("market_total_line")),
            "market_home_ml":        safe_int(row.get("market_home_ml")),
            "market_away_ml":        safe_int(row.get("market_away_ml")),
            "best_home_ml":          safe_int(row.get("best_home_ml")),
            "best_away_ml":          safe_int(row.get("best_away_ml")),
            "model_edge":            safe_float(row.get("model_edge")),

            "error":                 None if has_prediction else "No prediction available",
        })

    return {
        "ok":          True,
        "date":        game_date,
        "count":       len(results),
        "predictions": results,
    }


# ---------------------------------------------------------------------------
# API: pitch mix
# ---------------------------------------------------------------------------

@router.get("/api/pitch_mix")
def api_pitch_mix(pitcher_id: int, season: int):
    sql = text(
        """
        SELECT pitcher_id, pitcher_name, season, pitch_type, usage_pct, pitch_count
        FROM pitch_mix
        WHERE pitcher_id = :pitcher_id
          AND season     = :season
        ORDER BY pitch_count DESC
        """
    )

    df = pd.read_sql(sql, engine, params={"pitcher_id": pitcher_id, "season": season})
    df = df.where(pd.notna(df), other=None)

    return {"ok": True, "pitch_mix": df.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# API: matchup grid (batter vs pitcher heatmap)
# ---------------------------------------------------------------------------

@router.get("/api/matchup_grid")
def api_matchup_grid(game_pk: int):
    sql_game = text("""
        SELECT
            g.game_pk,
            g.home_team,
            g.away_team,
            COALESCE(gp.home_sp_name, g.home_starting_pitcher) AS home_sp_name,
            COALESCE(gp.away_sp_name, g.away_starting_pitcher) AS away_sp_name,
            COALESCE(gp.home_sp_id, NULL) AS home_sp_id,
            COALESCE(gp.away_sp_id, NULL) AS away_sp_id
        FROM games g
        LEFT JOIN game_probables gp ON gp.game_pk = g.game_pk
        WHERE g.game_pk = :game_pk
    """)
    game_df = pd.read_sql(sql_game, engine, params={"game_pk": game_pk})
    if game_df.empty:
        return {"ok": False, "error": "Game not found"}

    g = game_df.iloc[0]
    home_team    = safe_str(g.get("home_team"))
    away_team    = safe_str(g.get("away_team"))
    home_sp_name = safe_str(g.get("home_sp_name")) or "TBD"
    away_sp_name = safe_str(g.get("away_sp_name")) or "TBD"
    home_sp_id   = safe_int(g.get("home_sp_id"))
    away_sp_id   = safe_int(g.get("away_sp_id"))

    def get_batters(side: str) -> list:
        sql = text("""
            SELECT batting_order, player_id, player_name, position
            FROM lineups
            WHERE game_pk = :game_pk AND side = :side
            ORDER BY batting_order
        """)
        df = pd.read_sql(sql, engine, params={"game_pk": game_pk, "side": side})
        return df.to_dict(orient="records") if not df.empty else []

    def get_bvp(batter_ids: list, pitcher_id: Optional[int]) -> dict:
        if not batter_ids or not pitcher_id:
            return {}
        sql = text("""
            SELECT batter_id, pa, ab, hits, home_runs, strikeouts, walks,
                   avg, obp, slg, hard_hit_pct, avg_exit_velo, k_pct, bb_pct
            FROM batter_vs_pitcher
            WHERE batter_id = ANY(:bids) AND pitcher_id = :pid
        """)
        df = pd.read_sql(sql, engine, params={"bids": batter_ids, "pid": pitcher_id})
        if df.empty:
            return {}
        df = df.where(pd.notna(df), other=None)
        return {row["batter_id"]: row for _, row in df.iterrows()}

    away_batters = get_batters("away")
    away_bids    = [b["player_id"] for b in away_batters]
    away_bvp     = get_bvp(away_bids, home_sp_id)

    home_batters = get_batters("home")
    home_bids    = [b["player_id"] for b in home_batters]
    home_bvp     = get_bvp(home_bids, away_sp_id)

    def merge_bvp(batters: list, bvp_map: dict) -> list:
        out = []
        for b in batters:
            pid   = b["player_id"]
            stats = bvp_map.get(pid, {})
            out.append({
                "batting_order": b["batting_order"],
                "player_id":     pid,
                "player_name":   b["player_name"],
                "position":      b.get("position"),
                "pa":            safe_int(stats.get("pa")) or 0,
                "ab":            safe_int(stats.get("ab")) or 0,
                "hits":          safe_int(stats.get("hits")) or 0,
                "home_runs":     safe_int(stats.get("home_runs")) or 0,
                "strikeouts":    safe_int(stats.get("strikeouts")) or 0,
                "walks":         safe_int(stats.get("walks")) or 0,
                "avg":           safe_float(stats.get("avg")),
                "obp":           safe_float(stats.get("obp")),
                "slg":           safe_float(stats.get("slg")),
                "hard_hit_pct":  safe_float(stats.get("hard_hit_pct")),
                "avg_exit_velo": safe_float(stats.get("avg_exit_velo")),
                "k_pct":         safe_float(stats.get("k_pct")),
                "bb_pct":        safe_float(stats.get("bb_pct")),
            })
        return out

    return {
        "ok":              True,
        "game_pk":         game_pk,
        "home_team":       home_team,
        "away_team":       away_team,
        "home_sp_name":    home_sp_name,
        "away_sp_name":    away_sp_name,
        "home_sp_id":      home_sp_id,
        "away_sp_id":      away_sp_id,
        "away_vs_home_sp": merge_bvp(away_batters, away_bvp),
        "home_vs_away_sp": merge_bvp(home_batters, home_bvp),
    }