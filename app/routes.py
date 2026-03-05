from __future__ import annotations

from datetime import date, datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.db import engine
from app.predictor import predict_game

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


# Simple logo map (ESPN CDN) – reliable and easy
TEAM_LOGO = {
    "ARI": "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/mlb/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/mlb/500/bal.png",
    "BOS": "https://a.espncdn.com/i/teamlogos/mlb/500/bos.png",
    "CHC": "https://a.espncdn.com/i/teamlogos/mlb/500/chc.png",
    "CWS": "https://a.espncdn.com/i/teamlogos/mlb/500/cws.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/mlb/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/mlb/500/cle.png",
    "COL": "https://a.espncdn.com/i/teamlogos/mlb/500/col.png",
    "DET": "https://a.espncdn.com/i/teamlogos/mlb/500/det.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/mlb/500/hou.png",
    "KC":  "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",
    "LAA": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",
    "LAD": "https://a.espncdn.com/i/teamlogos/mlb/500/lad.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/mlb/500/mia.png",
    "MIL": "https://a.espncdn.com/i/teamlogos/mlb/500/mil.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/mlb/500/min.png",
    "NYM": "https://a.espncdn.com/i/teamlogos/mlb/500/nym.png",
    "NYY": "https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png",
    "OAK": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/mlb/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/mlb/500/pit.png",
    "SD":  "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/mlb/500/sea.png",
    "SF":  "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",
    "STL": "https://a.espncdn.com/i/teamlogos/mlb/500/stl.png",
    "TB":  "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",
    "TEX": "https://a.espncdn.com/i/teamlogos/mlb/500/tex.png",
    "TOR": "https://a.espncdn.com/i/teamlogos/mlb/500/tor.png",
    "WSH": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png",
}


@router.get("/", response_class=HTMLResponse)
def homepage(request: Request, game_date: Optional[str] = Query(default=None)):
    """
    Renders the homepage. You can pass ?game_date=YYYY-MM-DD
    """
    if not game_date:
        game_date = date.today().isoformat()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "game_date": game_date,
        },
    )


@router.get("/api/games")
def api_games(game_date: str = Query(..., description="YYYY-MM-DD")):
    # Your games.game_date might be text or timestamp; cast safely
    sql = """
        SELECT *
        FROM games
        WHERE (game_date::date) = :d
        ORDER BY home_team, away_team
    """
    df = pd.read_sql(sql, engine, params={"d": game_date})

    games = []
    for _, r in df.iterrows():
        home = str(r.get("home_team", ""))
        away = str(r.get("away_team", ""))
        games.append(
            {
                "id": int(r.get("id")) if r.get("id") is not None else None,
                "game_date": str(r.get("game_date")),
                "home_team": home,
                "away_team": away,
                "home_logo": TEAM_LOGO.get(home, ""),
                "away_logo": TEAM_LOGO.get(away, ""),
                # keep pitchers if present
                "home_starting_pitcher": r.get("home_starting_pitcher"),
                "away_starting_pitcher": r.get("away_starting_pitcher"),
            }
        )

    return {"ok": True, "count": len(games), "games": games}


@router.get("/api/predict/today")
def api_predict_today(game_date: str = Query(..., description="YYYY-MM-DD")):
    """
    For now this uses whatever feature columns exist in your games table.
    If you don't have these feature columns stored, you'll see ok:false + missing features.
    """
    # Pull games for date
    sql_games = """
        SELECT *
        FROM games
        WHERE (game_date::date) = :d
        ORDER BY home_team, away_team
    """
    g = pd.read_sql(sql_games, engine, params={"d": game_date})

    results: List[Dict[str, Any]] = []
    for _, row in g.iterrows():
        payload = row.to_dict()

        # Map expected feature names if they exist in games table
        features_row = {
            "era_diff": payload.get("era_diff"),
            "whip_diff": payload.get("whip_diff"),
            "home_last10_runs_scored": payload.get("home_last10_runs_scored"),
            "away_last10_runs_scored": payload.get("away_last10_runs_scored"),
            "home_last10_runs_allowed": payload.get("home_last10_runs_allowed"),
            "away_last10_runs_allowed": payload.get("away_last10_runs_allowed"),
            "home_last10_run_diff": payload.get("home_last10_run_diff"),
            "away_last10_run_diff": payload.get("away_last10_run_diff"),
        }

        pred = predict_game(features_row)

        home = str(payload.get("home_team", ""))
        away = str(payload.get("away_team", ""))

        results.append(
            {
                "id": int(payload["id"]) if payload.get("id") is not None else None,
                "game_date": str(payload.get("game_date")),
                "home_team": home,
                "away_team": away,
                "home_logo": TEAM_LOGO.get(home, ""),
                "away_logo": TEAM_LOGO.get(away, ""),
                **pred,
            }
        )

    return {"ok": True, "date": game_date, "count": len(results), "predictions": results}