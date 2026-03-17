from __future__ import annotations

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


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def team_logo_by_id(team_id):
    if team_id is None or pd.isna(team_id):
        return ""
    return f"https://www.mlbstatic.com/team-logos/team-cap-on-dark/{int(team_id)}.svg"


@router.get("/", response_class=HTMLResponse)
def homepage(request: Request, game_date: Optional[str] = Query(default=None)):
    if not game_date:
        game_date = date.today().isoformat()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "game_date": game_date,
        },
    )


@router.get("/matchups", response_class=HTMLResponse)
def matchups_page(request: Request, game_date: Optional[str] = Query(default=None)):
    if not game_date:
        game_date = date.today().isoformat()

    return templates.TemplateResponse(
        "matchups.html",
        {
            "request": request,
            "game_date": game_date,
        },
    )


@router.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request, game_date: Optional[str] = Query(default=None)):
    if not game_date:
        game_date = date.today().isoformat()

    return templates.TemplateResponse(
        "predictions.html",
        {
            "request": request,
            "game_date": game_date,
        },
    )


@router.get("/api/games")
def api_games(game_date: str = Query(..., description="YYYY-MM-DD")):
    sql = text(
        """
        SELECT
            g.game_pk,
            g.official_date,
            g.home_team,
            g.away_team,
            COALESCE(gp.home_sp_name, g.home_starting_pitcher) AS home_starting_pitcher,
            COALESCE(gp.away_sp_name, g.away_starting_pitcher) AS away_starting_pitcher,
            COALESCE(gp.home_sp_id, NULL) AS home_sp_id,
            COALESCE(gp.away_sp_id, NULL) AS away_sp_id,
            g.home_team_id,
            g.away_team_id
        FROM games g
        LEFT JOIN game_probables gp
            ON gp.game_pk = g.game_pk
        WHERE CAST(g.official_date AS DATE) = :d
        ORDER BY g.home_team, g.away_team
        """
    )

    df = pd.read_sql(sql, engine, params={"d": game_date})

    games: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        games.append(
            {
                "game_pk": int(r["game_pk"]) if pd.notna(r.get("game_pk")) else None,
                "game_date": _safe_str(r.get("official_date")),
                "home_team": _safe_str(r.get("home_team")),
                "away_team": _safe_str(r.get("away_team")),
                "home_logo": team_logo_by_id(r.get("home_team_id")),
                "away_logo": team_logo_by_id(r.get("away_team_id")),
                "home_starting_pitcher": r.get("home_starting_pitcher"),
                "away_starting_pitcher": r.get("away_starting_pitcher"),
                "home_sp_id": int(r["home_sp_id"]) if pd.notna(r.get("home_sp_id")) else None,
                "away_sp_id": int(r["away_sp_id"]) if pd.notna(r.get("away_sp_id")) else None,
                "home_team_id": int(r["home_team_id"]) if pd.notna(r.get("home_team_id")) else None,
                "away_team_id": int(r["away_team_id"]) if pd.notna(r.get("away_team_id")) else None,
            }
        )

    return {"ok": True, "count": len(games), "games": games}


@router.get("/api/predict/today")
def api_predict_today(game_date: str = Query(..., description="YYYY-MM-DD")):
    sql = text(
        """
        SELECT
            g.game_pk,
            g.official_date,
            g.home_team,
            g.away_team,
            COALESCE(gp.home_sp_name, g.home_starting_pitcher) AS home_starting_pitcher,
            COALESCE(gp.away_sp_name, g.away_starting_pitcher) AS away_starting_pitcher,
            COALESCE(gp.home_sp_id, NULL) AS home_sp_id,
            COALESCE(gp.away_sp_id, NULL) AS away_sp_id,
            g.home_team_id,
            g.away_team_id,
            p.home_win_prob,
            p.away_win_prob,
            p.home_ml_implied,
            p.away_ml_implied,
            p.run_diff_pred,
            p.total_runs_pred,
            p.ml_pick,
            p.runline_pick,
            p.ou_pick
        FROM games g
        LEFT JOIN predictions p
            ON p.game_pk = g.game_pk
        LEFT JOIN game_probables gp
            ON gp.game_pk = g.game_pk
        WHERE CAST(g.official_date AS DATE) = :d
        ORDER BY g.home_team, g.away_team
        """
    )

    df = pd.read_sql(sql, engine, params={"d": game_date})

    results: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        home_win_prob = row.get("home_win_prob")
        if pd.notna(home_win_prob):
            home_win_prob = float(home_win_prob)
        else:
            home_win_prob = None

        results.append(
            {
                "id": int(row["game_pk"]) if pd.notna(row.get("game_pk")) else None,
                "game_pk": int(row["game_pk"]) if pd.notna(row.get("game_pk")) else None,
                "game_date": _safe_str(row.get("official_date")),
                "home_team": _safe_str(row.get("home_team")),
                "away_team": _safe_str(row.get("away_team")),
                "home_logo": team_logo_by_id(row.get("home_team_id")),
                "away_logo": team_logo_by_id(row.get("away_team_id")),
                "home_starting_pitcher": row.get("home_starting_pitcher"),
                "away_starting_pitcher": row.get("away_starting_pitcher"),
                "home_sp_id": int(row["home_sp_id"]) if pd.notna(row.get("home_sp_id")) else None,
                "away_sp_id": int(row["away_sp_id"]) if pd.notna(row.get("away_sp_id")) else None,
                "ok": True if pd.notna(row.get("ml_pick")) or pd.notna(row.get("home_win_prob")) else False,
                "home_win_prob": home_win_prob,
                "away_win_prob": float(row["away_win_prob"]) if pd.notna(row.get("away_win_prob")) else None,
                "home_moneyline_fair": int(row["home_ml_implied"]) if pd.notna(row.get("home_ml_implied")) else None,
                "away_moneyline_fair": int(row["away_ml_implied"]) if pd.notna(row.get("away_ml_implied")) else None,
                "pred_total_runs": float(row["total_runs_pred"]) if pd.notna(row.get("total_runs_pred")) else None,
                "pred_run_diff": float(row["run_diff_pred"]) if pd.notna(row.get("run_diff_pred")) else None,
                "ml_pick": row.get("ml_pick"),
                "runline_pick": row.get("runline_pick"),
                "ou_pick": row.get("ou_pick"),
                "error": None if pd.notna(row.get("ml_pick")) or pd.notna(row.get("home_win_prob")) else "No prediction available",
            }
        )

    return {
        "ok": True,
        "date": game_date,
        "count": len(results),
        "predictions": results,
    }

@router.get("/api/pitch_mix")
def api_pitch_mix(pitcher_id: int, season: int):
    sql = text(
        """
        SELECT pitcher_id, pitcher_name, season, pitch_type, usage_pct, pitch_count
        FROM pitch_mix
        WHERE pitcher_id = :pitcher_id
          AND season = :season
        ORDER BY pitch_count DESC
        """
    )

    df = pd.read_sql(sql, engine, params={"pitcher_id": pitcher_id, "season": season})

    return {
        "ok": True,
        "pitch_mix": df.to_dict(orient="records")
    }