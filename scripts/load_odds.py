from __future__ import annotations

import math
import os
from datetime import datetime, timezone

from scripts.mlb_engine_daily import (
    conn,
    ensure_tables,
    team_map,
    upsert_schedule,
    upsert_probables,
    build_features_for_date,
    align_to_model,
    load_models,
    save_predictions,
    implied_moneyline,
    DEFAULT_TOTAL_LINE,
    SEASON,
)
import pandas as pd


def _safe_float(val):
    """Return a JSON-safe float or a fallback. Handles nan, inf, None."""
    try:
        if val is None:
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def main():
    today = datetime.now(timezone.utc).date()

    print("Quick update mode")
    print("Date =", today)

    win_model, run_model, total_model = load_models()

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_tables(cur)

            # 1) Update today's schedule
            tmap    = team_map(SEASON)
            n_sched = upsert_schedule(cur, today, today, tmap)

            # 2) Update today's probable starters
            n_prob  = upsert_probables(cur, today, today)

            # 3) Build today's features (includes ERA/WHIP + rolling stats)
            base = build_features_for_date(cur, today)
            if base.empty:
                c.commit()
                print(f"No games found for {today}.")
                return

            # 4) Align features to each model
            X_win = align_to_model(base, win_model)
            X_run = align_to_model(base, run_model)
            X_tot = align_to_model(base, total_model)

            # 5) Predict — sanitize every output value
            if hasattr(win_model, "predict_proba"):
                p = win_model.predict_proba(X_win)
                home_prob_raw = p[:, 1] if p.shape[1] > 1 else p[:, 0]
            else:
                home_prob_raw = win_model.predict(X_win)

            home_prob  = [_safe_float(x) or 0.5  for x in home_prob_raw]
            away_prob  = [round(1.0 - hp, 6)      for hp in home_prob]
            prediction = [1 if hp >= 0.5 else 0   for hp in home_prob]
            run_diff   = [_safe_float(x) or 0.0   for x in run_model.predict(X_run)]
            total_runs = [_safe_float(x) or 0.0   for x in total_model.predict(X_tot)]

            ml_pick      = ["HOME" if hp >= 0.5       else "AWAY"      for hp in home_prob]
            runline_pick = ["HOME -1.5" if rd >= 1.5  else "AWAY +1.5" for rd in run_diff]
            ou_pick      = ["OVER" if tr >= DEFAULT_TOTAL_LINE else "UNDER" for tr in total_runs]

            out = pd.DataFrame({
                "game_pk":         base["game_pk"].astype(int),
                "official_date":   pd.to_datetime(base["official_date"]).dt.date,
                "away_team":       base["away_team"].astype(str),
                "home_team":       base["home_team"].astype(str),
                "prediction":      prediction,
                "win_probability": home_prob,
                "home_win_prob":   home_prob,
                "away_win_prob":   away_prob,
                "home_ml_implied": [implied_moneyline(x) for x in home_prob],
                "away_ml_implied": [implied_moneyline(x) for x in away_prob],
                "run_diff_pred":   run_diff,
                "total_runs_pred": total_runs,
                "ml_pick":         ml_pick,
                "runline_pick":    runline_pick,
                "ou_pick":         ou_pick,
            })

            n_saved = save_predictions(cur, out)
            c.commit()

            print(f"Today's schedule updated:   {n_sched}")
            print(f"Today's probables updated:  {n_prob}")
            print(f"Today's predictions saved:  {n_saved}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()