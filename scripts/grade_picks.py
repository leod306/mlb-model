"""
scripts/grade_picks.py  —  Honest P&L grading for the MLB model.

Run from project root:
    python scripts/grade_picks.py                # whole season
    python scripts/grade_picks.py 2026-06-25     # only from this date on

WHY THIS EXISTS
---------------
The Summary tab in picks_tracker graded every bet type against a flat 52.4%
break-even (the -110 number) and reported win%, not money. That made losing
underdog run-line bets (-200 prices, ~67% break-even) look "green" at 59%.
This script grades every pick at the ACTUAL price you'd have gotten and reports
profit in UNITS, which is the only thing that tells you if you're winning.

PRICE SOURCES (from load_odds.py)
---------------------------------
  best_home_ml / best_away_ml   REAL bettable price (best of 4 books). USE FOR P&L.
  market_home_ml / market_away_ml   SYNTHETIC de-vigged fair odds (vig removed).
                                    Correct for edge detection, WRONG for payout —
                                    you can't bet at fair odds. Not used here.
  market_home_prob/away_prob    de-vigged implied prob — the model's opponent.
  market_total_line             the total. O/U graded at an assumed price.

CLV is NOT computed here — market_odds holds one morning snapshot per game, not
a closing line. CLV needs a second fetch near first pitch (a going-forward build).
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

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

PREDICTIONS_TABLE = os.getenv("MLB_PREDICTIONS_TABLE", "predictions")
ODDS_TABLE        = os.getenv("MLB_ODDS_TABLE",        "market_odds")
GAMES_TABLE       = os.getenv("MLB_GAMES_TABLE",       "games")

# Assumed O/U price when the book price isn't stored (standard juice).
OU_ASSUMED_PRICE = -110
# Assumed run-line prices when RL prices aren't stored. RL prices are NOT in
# market_odds yet, so RL P&L is an ESTIMATE using typical numbers. Favorite
# -1.5 usually pays around +130; underdog +1.5 usually costs around -160.
RL_FAV_ASSUMED_PRICE = 130
RL_DOG_ASSUMED_PRICE = -160


# ---------------------------------------------------------------------------
# Betting math
# ---------------------------------------------------------------------------
def american_payout(ml) -> float | None:
    """Profit per 1u staked at American odds (excludes returned stake)."""
    try:
        m = float(ml)
    except (TypeError, ValueError):
        return None
    if m == 0 or math.isnan(m):
        return None
    return (100.0 / -m) if m < 0 else (m / 100.0)


def american_to_prob(ml) -> float | None:
    p = american_payout(ml)
    if p is None:
        return None
    # break-even prob = 1 / (payout + 1)
    return 1.0 / (p + 1.0)


def settle(win: bool | None, ml) -> float | None:
    """
    Units won/lost on a 1u bet. Win → +payout, loss → -1, push/None → 0/None.
    """
    if win is None:
        return None
    payout = american_payout(ml)
    if payout is None:
        return None
    return payout if win else -1.0


# ---------------------------------------------------------------------------
# Load joined picks + odds
# ---------------------------------------------------------------------------
def load_joined(since: str | None) -> pd.DataFrame:
    # Scores live on the `games` table, not `predictions`. Join games for
    # actual results, and odds for prices. Predictions carries game_pk, so we
    # join odds on game_pk when available (robust to doubleheaders / name
    # mismatches) and fall back to date+teams for any odds rows with null pk.
    where = "WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL"
    params: dict = {}
    if since:
        where += " AND p.official_date >= :since"
        params["since"] = since

    sql = f"""
    SELECT
        p.official_date, p.home_team, p.away_team,
        g.home_score, g.away_score,
        p.home_win_prob, p.total_runs_pred,
        p.ml_pick, p.ou_pick, p.runline_pick,
        mo.best_home_ml, mo.best_away_ml,
        mo.market_home_ml, mo.market_away_ml,
        mo.market_home_prob, mo.market_away_prob,
        mo.market_total_line
    FROM {PREDICTIONS_TABLE} p
    JOIN {GAMES_TABLE} g
      ON p.game_pk = g.game_pk
    LEFT JOIN {ODDS_TABLE} mo
      ON  mo.official_date = p.official_date
      AND mo.home_team     = p.home_team
      AND mo.away_team     = p.away_team
    {where}
    ORDER BY p.official_date, p.home_team
    """
    return pd.read_sql(text(sql), engine, params=params)


# ---------------------------------------------------------------------------
# Grade one row → up to 3 bet results
# ---------------------------------------------------------------------------
def grade_row(r) -> list[dict]:
    out = []
    home, away = r["home_team"], r["away_team"]
    hs, as_ = r["home_score"], r["away_score"]
    if pd.isna(hs) or pd.isna(as_):
        return out
    hs, as_ = float(hs), float(as_)
    total   = hs + as_
    margin  = hs - as_   # home margin

    # ---- Moneyline (real price via best_*_ml) ----
    ml = r.get("ml_pick")
    if ml and ml not in ("PASS", "", None) and not pd.isna(ml):
        if ml == home:
            price = r.get("best_home_ml")
            won   = margin > 0
        elif ml == away:
            price = r.get("best_away_ml")
            won   = margin < 0
        else:
            price, won = None, None
        if price is not None and not pd.isna(price) and won is not None:
            if margin == 0:
                units = None  # extra-inning MLB games don't tie, but guard anyway
            else:
                units = settle(bool(won), price)
            out.append({"bet": "ML", "pick": ml, "price": float(price),
                        "won": bool(won), "units": units,
                        "date": r["official_date"], "measurable": True})
        else:
            out.append({"bet": "ML", "pick": ml, "price": None, "won": None,
                        "units": None, "date": r["official_date"], "measurable": False})

    # ---- Over/Under (assumed -110; real total line) ----
    ou   = r.get("ou_pick")
    line = r.get("market_total_line")
    if ou in ("OVER", "UNDER") and line is not None and not pd.isna(line):
        line = float(line)
        if total == line:
            units, won = 0.0, None   # push
        elif ou == "OVER":
            won   = total > line
            units = settle(won, OU_ASSUMED_PRICE)
        else:
            won   = total < line
            units = settle(won, OU_ASSUMED_PRICE)
        out.append({"bet": "O/U", "pick": ou, "price": OU_ASSUMED_PRICE,
                    "won": won, "units": units, "date": r["official_date"],
                    "measurable": True, "push": total == line})

    # ---- Run line (ESTIMATED price — no RL prices stored yet) ----
    rl = r.get("runline_pick")
    if rl and rl not in ("PASS", "", None) and not pd.isna(rl):
        # parse "TEAM -1.5" / "TEAM +1.5"
        try:
            parts = str(rl).rsplit(" ", 1)
            team  = parts[0]
            spread = float(parts[1])
        except Exception:
            team, spread = None, None
        if team in (home, away) and spread is not None:
            if team == home:
                cover = (margin + spread) > 0
            else:
                cover = ((-margin) + spread) > 0
            price = RL_FAV_ASSUMED_PRICE if spread < 0 else RL_DOG_ASSUMED_PRICE
            units = settle(cover, price)
            out.append({"bet": "RL", "pick": rl, "price": price,
                        "won": bool(cover), "units": units,
                        "date": r["official_date"], "measurable": True,
                        "estimated_price": True})
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def summarize(results: pd.DataFrame) -> None:
    print("=" * 72)
    print("HONEST P&L — graded at real prices (ML: best book; O/U: -110; RL: est.)")
    print("=" * 72)

    for bet in ["ML", "O/U", "RL"]:
        sub = results[results["bet"] == bet]
        if sub.empty:
            continue

        measurable = sub[sub["measurable"]]
        graded     = measurable[measurable["units"].notna()]         # excludes pushes/unmeasurable
        pushes     = int((measurable["units"] == 0).sum())
        unmeasur   = int((~sub["measurable"]).sum())

        n      = len(graded)
        wins   = int(graded["won"].sum()) if "won" in graded else 0
        units  = graded["units"].sum()
        win_pct = wins / n if n else 0.0
        roi     = units / n if n else 0.0

        # win rate you'd NEED to break even at the average price actually taken
        be_probs = [american_to_prob(p) for p in graded["price"] if p is not None]
        be_probs = [b for b in be_probs if b is not None]
        need     = sum(be_probs) / len(be_probs) if be_probs else float("nan")

        verdict = "PROFIT ✅" if units > 0 else ("FLAT" if abs(units) < 1e-9 else "LOSS ❌")

        label = {"ML": "Moneyline", "O/U": "Over/Under", "RL": "Run Line"}[bet]
        note  = "  (RL price ESTIMATED — add real RL odds to confirm)" if bet == "RL" else ""
        print(f"\n{label}{note}")
        print(f"  Record          : {wins}-{n-wins}  ({win_pct:.1%})")
        print(f"  Break-even need : {need:.1%}   (at the prices actually taken)")
        print(f"  Units P&L       : {units:+.2f}u over {n} bets")
        print(f"  ROI / bet       : {roi:+.2%}   →  {verdict}")
        if pushes:
            print(f"  Pushes          : {pushes}")
        if unmeasur:
            print(f"  Unmeasurable    : {unmeasur} (no stored price — not counted)")

    # overall
    graded_all = results[results["measurable"] & results["units"].notna()]
    total_u = graded_all["units"].sum()
    print("\n" + "-" * 72)
    print(f"TOTAL (all bet types, flat 1u): {total_u:+.2f}u over {len(graded_all)} bets"
          f"  →  ROI {total_u/len(graded_all):+.2%}" if len(graded_all) else "No graded bets.")
    print("-" * 72)
    print("\nNote: win% alone is meaningless without price. A 59% run-line record can")
    print("still lose money if those bets pay -160. Units P&L is the scoreboard.")
    print("CLV not shown — needs a closing-line capture near first pitch (not yet built).")


def main() -> None:
    since = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_joined(since)
    if df.empty:
        print("No completed games with predictions found"
              + (f" since {since}." if since else "."))
        return

    print(f"Loaded {len(df)} completed games"
          + (f" since {since}" if since else "") + ".")
    odds_cov = df["best_home_ml"].notna().mean()
    print(f"ML price coverage: {odds_cov:.1%} of games have a bettable moneyline stored.")

    rows = []
    for _, r in df.iterrows():
        rows.extend(grade_row(r))
    results = pd.DataFrame(rows)
    if results.empty:
        print("Nothing gradeable (no picks matched to prices).")
        return

    summarize(results)

    # daily units trend for the last 14 active days — quick 'are the changes working' read
    graded = results[results["measurable"] & results["units"].notna()].copy()
    if not graded.empty:
        daily = graded.groupby(["date", "bet"])["units"].sum().unstack(fill_value=0.0)
        print("\nDaily units by bet type (last 14 days):")
        print(daily.tail(14).round(2).to_string())


if __name__ == "__main__":
    main()