from __future__ import annotations

from datetime import date
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text

from app.db import engine
from app.predictor import predict_game

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

TEAM_LOGO = {
    "ARI": "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",
    "AZ":  "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/mlb/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/mlb/500/bal.png",
    "BOS": "https://a.espncdn.com/i/teamlogos/mlb/500/bos.png",
    "CHC": "https://a.espncdn.com/i/teamlogos/mlb/500/chc.png",
    "CWS": "https://a.espncdn.com/i/teamlogos/mlb/500/cws.png",
    "CHW": "https://a.espncdn.com/i/teamlogos/mlb/500/cws.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/mlb/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/mlb/500/cle.png",
    "COL": "https://a.espncdn.com/i/teamlogos/mlb/500/col.png",
    "DET": "https://a.espncdn.com/i/teamlogos/mlb/500/det.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/mlb/500/hou.png",
    "KC":  "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",
    "KCR": "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",
    "LAA": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",
    "LAD": "https://a.espncdn.com/i/teamlogos/mlb/500/lad.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/mlb/500/mia.png",
    "MIL": "https://a.espncdn.com/i/teamlogos/mlb/500/mil.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/mlb/500/min.png",
    "NYM": "https://a.espncdn.com/i/teamlogos/mlb/500/nym.png",
    "NYY": "https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png",
    "OAK": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "ATH": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/mlb/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/mlb/500/pit.png",
    "SD":  "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",
    "SDP": "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/mlb/500/sea.png",
    "SF":  "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",
    "SFG": "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",
    "STL": "https://a.espncdn.com/i/teamlogos/mlb/500/stl.png",
    "TB":  "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",
    "TBR": "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",
    "TEX": "https://a.espncdn.com/i/teamlogos/mlb/500/tex.png",
    "TOR": "https://a.espncdn.com/i/teamlogos/mlb/500/tor.png",
    "WSH": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png",
    "WSN": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png",
}


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


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
        SELECT *
        FROM games
        WHERE CAST(official_date AS DATE) = :d
        ORDER BY home_team, away_team
        """
    )

    df = pd.read_sql(sql, engine, params={"d": game_date})

    games: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        home = _safe_str(r.get("home_team"))
        away = _safe_str(r.get("away_team"))

        games.append(
            {
                "id": int(r["id"]) if pd.notna(r.get("id")) else None,
                "game_date": _safe_str(r.get("official_date")),
                "home_team": home,
                "away_team": away,
                "home_logo": TEAM_LOGO.get(home, ""),
                "away_logo": TEAM_LOGO.get(away, ""),
                "home_starting_pitcher": r.get("home_starting_pitcher"),
                "away_starting_pitcher": r.get("away_starting_pitcher"),
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
            gp.home_sp_name,
            gp.away_sp_name,
            p.home_win_prob,
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
        home = _safe_str(row.get("home_team"))
        away = _safe_str(row.get("away_team"))

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
                "home_team": home,
                "away_team": away,
                "home_logo": TEAM_LOGO.get(home, ""),
                "away_logo": TEAM_LOGO.get(away, ""),
                "home_starting_pitcher": row.get("home_sp_name"),
                "away_starting_pitcher": row.get("away_sp_name"),
                "ok": True if pd.notna(row.get("ml_pick")) or pd.notna(row.get("home_win_prob")) else False,
                "home_win_prob": home_win_prob,
                "home_moneyline_fair": int(row["home_ml_implied"]) if pd.notna(row.get("home_ml_implied")) else None,
                "away_moneyline_fair": int(row["away_ml_implied"]) if pd.notna(row.get("away_ml_implied")) else None,
                "pred_total_runs": float(row["total_runs_pred"]) if pd.notna(row.get("total_runs_pred")) else None,
                "pred_run_diff": float(row["run_diff_pred"]) if pd.notna(row.get("run_diff_pred")) else None,
                "ml_pick": row.get("ml_pick"),
                "runline_pick": row.get("runline_pick"),
                "ou_pick": row.get("ou_pick"),
                "error": None if pd.notna(row.get("ml_pick")) or pd.notna(row.get("home_win_prob")) else "No prediction row found",
            }
        )

    return {
        "ok": True,
        "date": game_date,
        "count": len(results),
        "predictions": results,
    }
