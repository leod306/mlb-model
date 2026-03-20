"""
daily_picks_tracker.py
-----------------------
Single script that does three jobs in order:

  1. EVALUATE yesterday's picks against actual results
  2. SAVE today's predictions as picks to evaluate tomorrow
  3. EXPORT full picks history to Excel with color coding

Run once daily after predictions are generated, before games start.
Add to mlb_quick_update.py after load_odds.py:
  run_step("Daily Picks Tracker", "daily_picks_tracker.py")

Excel file saved to: reports/picks_tracker.xlsx
  - Green  = correct pick
  - Red    = incorrect pick
  - Yellow = pending (no result yet)
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from openpyxl import Workbook, load_workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)
GAMES_TABLE  = os.getenv("MLB_GAMES_TABLE", "games")
PRED_TABLE   = os.getenv("MLB_PREDICTIONS_TABLE", "predictions")
PICKS_TABLE  = "daily_picks"

RUNLINE_SPREAD  = 1.5
DEFAULT_OU_LINE = 8.5

REPORTS_DIR = Path(PROJECT_ROOT) / "reports"
EXCEL_PATH  = REPORTS_DIR / "picks_tracker.xlsx"

# Colors
GREEN    = "C6EFCE"
RED      = "FFC7CE"
YELLOW   = "FFEB9C"
BLUE     = "BDD7EE"
GRAY     = "F2F2F2"
DARK     = "1F4E79"
WHITE    = "FFFFFF"


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_tables(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {PICKS_TABLE} (
            id              SERIAL PRIMARY KEY,
            game_pk         BIGINT NOT NULL,
            pick_date       DATE   NOT NULL,
            home_team       TEXT,
            away_team       TEXT,
            home_sp         TEXT,
            away_sp         TEXT,
            ml_pick         TEXT,
            runline_pick    TEXT,
            ou_pick         TEXT,
            home_win_prob   DOUBLE PRECISION,
            away_win_prob   DOUBLE PRECISION,
            pred_run_diff   DOUBLE PRECISION,
            pred_total_runs DOUBLE PRECISION,
            home_ml_implied INT,
            away_ml_implied INT,
            home_score      INT,
            away_score      INT,
            actual_run_diff DOUBLE PRECISION,
            actual_total    DOUBLE PRECISION,
            ml_correct      BOOLEAN,
            runline_correct BOOLEAN,
            ou_correct      BOOLEAN,
            evaluated       BOOLEAN DEFAULT FALSE,
            evaluated_at    TIMESTAMPTZ,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (game_pk, pick_date)
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_picks_date ON {PICKS_TABLE}(pick_date);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_picks_eval ON {PICKS_TABLE}(evaluated);")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_ml(ml_pick, home_score, away_score):
    if None in (home_score, away_score) or home_score == away_score:
        return None
    home_won = home_score > away_score
    return (ml_pick == "HOME" and home_won) or (ml_pick == "AWAY" and not home_won)


def eval_runline(runline_pick, home_score, away_score):
    if None in (home_score, away_score):
        return None
    diff = home_score - away_score
    if runline_pick == "HOME -1.5": return diff > RUNLINE_SPREAD
    if runline_pick == "AWAY +1.5": return diff < RUNLINE_SPREAD
    return None


def eval_ou(ou_pick, home_score, away_score, pred_total):
    if None in (home_score, away_score):
        return None
    line   = pred_total if pred_total else DEFAULT_OU_LINE
    actual = home_score + away_score
    if actual == line: return None
    return (ou_pick == "OVER" and actual > line) or (ou_pick == "UNDER" and actual < line)


def sym(val):
    if val is None: return "—"
    return "✅" if val else "❌"


def pct_str(wins, total):
    if not total: return "—"
    return f"{int(wins)}/{total} ({round(wins/total*100, 1)}%)"


# ---------------------------------------------------------------------------
# Step 1: Evaluate yesterday
# ---------------------------------------------------------------------------

def evaluate_yesterday(cur):
    yesterday = date.today() - timedelta(days=1)
    print(f"{'='*55}")
    print(f"  STEP 1 — Evaluating picks for {yesterday}")
    print(f"{'='*55}\n")

    cur.execute(f"""
        SELECT id, game_pk, home_team, away_team,
               ml_pick, runline_pick, ou_pick, pred_total_runs
        FROM {PICKS_TABLE}
        WHERE pick_date = %s AND evaluated = FALSE
        ORDER BY game_pk
    """, (yesterday,))
    picks = cur.fetchall()

    if not picks:
        print(f"  No unevaluated picks for {yesterday}.\n")
        return

    ml_res = []
    rl_res = []
    ou_res = []

    for pick in picks:
        pick_id, game_pk, home_team, away_team, \
        ml_pick, runline_pick, ou_pick, pred_total = pick

        cur.execute(f"SELECT home_score, away_score FROM {GAMES_TABLE} WHERE game_pk = %s", (game_pk,))
        result = cur.fetchone()

        if not result or result[0] is None:
            print(f"  {away_team} @ {home_team}: no result yet, skipping")
            continue

        home_score, away_score = result
        actual_total    = home_score + away_score
        actual_run_diff = home_score - away_score

        ml_correct = eval_ml(ml_pick, home_score, away_score)
        rl_correct = eval_runline(runline_pick, home_score, away_score)
        ou_correct = eval_ou(ou_pick, home_score, away_score, pred_total)

        print(f"  {away_team} ({away_score}) @ {home_team} ({home_score})")
        print(f"    ML:  {(ml_pick or '—'):<14} {sym(ml_correct)}")
        print(f"    RL:  {(runline_pick or '—'):<14} {sym(rl_correct)}")
        print(f"    O/U: {(ou_pick or '—'):<14} {sym(ou_correct)}  "
              f"(pred: {round(pred_total,1) if pred_total else '—'} | actual: {actual_total})\n")

        if ml_correct is not None: ml_res.append(ml_correct)
        if rl_correct is not None: rl_res.append(rl_correct)
        if ou_correct is not None: ou_res.append(ou_correct)

        cur.execute(f"""
            UPDATE {PICKS_TABLE} SET
                home_score=%(hs)s, away_score=%(as_)s,
                actual_run_diff=%(ard)s, actual_total=%(at_)s,
                ml_correct=%(ml)s, runline_correct=%(rl)s, ou_correct=%(ou)s,
                evaluated=TRUE, evaluated_at=NOW()
            WHERE id=%(id)s
        """, dict(hs=home_score, as_=away_score, ard=actual_run_diff,
                  at_=actual_total, ml=ml_correct, rl=rl_correct,
                  ou=ou_correct, id=pick_id))

    print(f"  --- {yesterday} Summary ---")
    print(f"  Moneyline : {pct_str(sum(ml_res), len(ml_res))}")
    print(f"  Run Line  : {pct_str(sum(rl_res), len(rl_res))}")
    print(f"  Over/Under: {pct_str(sum(ou_res), len(ou_res))}\n")

    cur.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE ml_correct IS NOT NULL),
            COALESCE(SUM(CASE WHEN ml_correct THEN 1 ELSE 0 END), 0),
            COUNT(*) FILTER (WHERE runline_correct IS NOT NULL),
            COALESCE(SUM(CASE WHEN runline_correct THEN 1 ELSE 0 END), 0),
            COUNT(*) FILTER (WHERE ou_correct IS NOT NULL),
            COALESCE(SUM(CASE WHEN ou_correct THEN 1 ELSE 0 END), 0)
        FROM {PICKS_TABLE} WHERE evaluated = TRUE
    """)
    row = cur.fetchone()
    if row and row[0]:
        ml_tot, ml_w, rl_tot, rl_w, ou_tot, ou_w = row
        print(f"  --- Season Totals ---")
        print(f"  Moneyline : {pct_str(ml_w, ml_tot)}")
        print(f"  Run Line  : {pct_str(rl_w, rl_tot)}")
        print(f"  Over/Under: {pct_str(ou_w, ou_tot)}")
        print(f"  Break-even: 52.4% | Sharp target: 57%+\n")


# ---------------------------------------------------------------------------
# Step 2: Save today's picks
# ---------------------------------------------------------------------------

def save_today_picks(cur):
    today = date.today()
    print(f"{'='*55}")
    print(f"  STEP 2 — Saving picks for {today}")
    print(f"{'='*55}\n")

    cur.execute(f"""
        SELECT
            g.game_pk, g.home_team, g.away_team,
            COALESCE(gp.home_sp_name, g.home_starting_pitcher),
            COALESCE(gp.away_sp_name, g.away_starting_pitcher),
            p.ml_pick, p.runline_pick, p.ou_pick,
            p.home_win_prob, p.away_win_prob,
            p.run_diff_pred, p.total_runs_pred,
            p.home_ml_implied, p.away_ml_implied
        FROM {GAMES_TABLE} g
        LEFT JOIN {PRED_TABLE} p ON p.game_pk = g.game_pk
        LEFT JOIN game_probables gp ON gp.game_pk = g.game_pk
        WHERE g.official_date = %s
          AND g.game_type = 'R'
          AND p.ml_pick IS NOT NULL
        ORDER BY g.game_pk
    """, (today,))
    rows = cur.fetchall()

    if not rows:
        print(f"  No predictions for {today}.\n")
        return

    data = []
    for r in rows:
        (game_pk, home_team, away_team, home_sp, away_sp,
         ml_pick, runline_pick, ou_pick,
         home_win_prob, away_win_prob, run_diff_pred, total_runs_pred,
         home_ml_implied, away_ml_implied) = r

        print(f"  {away_team} @ {home_team}  |  ML: {ml_pick} | RL: {runline_pick} | O/U: {ou_pick}")
        data.append((
            int(game_pk), today,
            home_team, away_team, home_sp, away_sp,
            ml_pick, runline_pick, ou_pick,
            home_win_prob, away_win_prob,
            run_diff_pred, total_runs_pred,
            home_ml_implied, away_ml_implied,
        ))

    execute_values(cur, f"""
        INSERT INTO {PICKS_TABLE} (
            game_pk, pick_date, home_team, away_team, home_sp, away_sp,
            ml_pick, runline_pick, ou_pick,
            home_win_prob, away_win_prob, pred_run_diff, pred_total_runs,
            home_ml_implied, away_ml_implied
        ) VALUES %s
        ON CONFLICT (game_pk, pick_date) DO UPDATE SET
            ml_pick=EXCLUDED.ml_pick, runline_pick=EXCLUDED.runline_pick,
            ou_pick=EXCLUDED.ou_pick, home_win_prob=EXCLUDED.home_win_prob,
            away_win_prob=EXCLUDED.away_win_prob,
            pred_run_diff=EXCLUDED.pred_run_diff,
            pred_total_runs=EXCLUDED.pred_total_runs,
            home_ml_implied=EXCLUDED.home_ml_implied,
            away_ml_implied=EXCLUDED.away_ml_implied;
    """, data)
    print(f"\n  ✅ Saved {len(data)} picks for {today}\n")


# ---------------------------------------------------------------------------
# Step 3: Export to Excel
# ---------------------------------------------------------------------------

def result_fill(val):
    if val is True:  return PatternFill("solid", fgColor=GREEN)
    if val is False: return PatternFill("solid", fgColor=RED)
    return PatternFill("solid", fgColor=YELLOW)


def header_style(cell, text):
    cell.value     = text
    cell.font      = Font(bold=True, color=WHITE, name="Arial", size=10)
    cell.fill      = PatternFill("solid", fgColor=DARK)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def export_excel(cur):
    print(f"{'='*55}")
    print(f"  STEP 3 — Exporting to Excel")
    print(f"{'='*55}\n")

    cur.execute(f"""
        SELECT
            pick_date, away_team, home_team, away_sp, home_sp,
            ml_pick, home_win_prob,
            runline_pick, pred_run_diff,
            ou_pick, pred_total_runs,
            home_score, away_score,
            actual_total, actual_run_diff,
            ml_correct, runline_correct, ou_correct,
            evaluated
        FROM {PICKS_TABLE}
        ORDER BY pick_date DESC, game_pk
    """)
    rows = cur.fetchall()

    if not rows:
        print("  No picks to export yet.\n")
        return

    REPORTS_DIR.mkdir(exist_ok=True)
    wb = Workbook()

    # ── Sheet 1: All Picks ──────────────────────────────────────────────────
    ws = wb.active
    ws.title = "All Picks"

    headers = [
        "Date", "Matchup", "Away SP", "Home SP",
        "ML Pick", "Win Prob",
        "RL Pick", "Pred Run Diff",
        "O/U Pick", "Pred Total",
        "Score", "Actual Total", "Actual Run Diff",
        "ML ✓", "RL ✓", "O/U ✓",
    ]
    for col, h in enumerate(headers, 1):
        header_style(ws.cell(row=1, column=col), h)

    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for row_idx, r in enumerate(rows, 2):
        (pick_date, away_team, home_team, away_sp, home_sp,
         ml_pick, home_win_prob,
         runline_pick, pred_run_diff,
         ou_pick, pred_total_runs,
         home_score, away_score,
         actual_total, actual_run_diff,
         ml_correct, runline_correct, ou_correct,
         evaluated) = r

        score_str = f"{away_score}-{home_score}" if home_score is not None else "—"
        matchup   = f"{away_team} @ {home_team}"
        win_prob  = f"{round(home_win_prob*100, 1)}%" if home_win_prob else "—"

        values = [
            str(pick_date),
            matchup,
            away_sp or "TBD",
            home_sp or "TBD",
            ml_pick or "—",
            win_prob,
            runline_pick or "—",
            round(pred_run_diff, 2) if pred_run_diff else "—",
            ou_pick or "—",
            round(pred_total_runs, 1) if pred_total_runs else "—",
            score_str,
            actual_total if actual_total else "—",
            round(actual_run_diff, 1) if actual_run_diff else "—",
            "✅" if ml_correct else ("❌" if ml_correct is False else "—"),
            "✅" if runline_correct else ("❌" if runline_correct is False else "—"),
            "✅" if ou_correct else ("❌" if ou_correct is False else "—"),
        ]

        for col, val in enumerate(values, 1):
            cell = ws.cell(row=row_idx, column=col, value=val)
            cell.font      = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border    = border

            # Color the result columns
            if col == 14:  # ML correct
                cell.fill = result_fill(ml_correct) if evaluated else PatternFill("solid", fgColor=GRAY)
            elif col == 15:  # RL correct
                cell.fill = result_fill(runline_correct) if evaluated else PatternFill("solid", fgColor=GRAY)
            elif col == 16:  # O/U correct
                cell.fill = result_fill(ou_correct) if evaluated else PatternFill("solid", fgColor=GRAY)
            elif row_idx % 2 == 0:
                cell.fill = PatternFill("solid", fgColor=GRAY)

    # Column widths
    widths = [12, 22, 20, 20, 10, 10, 14, 14, 10, 12, 12, 12, 16, 8, 8, 8]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30

    # ── Sheet 2: Summary ────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Summary")

    # Season totals
    cur.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE ml_correct IS NOT NULL)      AS ml_total,
            COALESCE(SUM(CASE WHEN ml_correct THEN 1 ELSE 0 END), 0) AS ml_wins,
            COUNT(*) FILTER (WHERE runline_correct IS NOT NULL) AS rl_total,
            COALESCE(SUM(CASE WHEN runline_correct THEN 1 ELSE 0 END), 0) AS rl_wins,
            COUNT(*) FILTER (WHERE ou_correct IS NOT NULL)      AS ou_total,
            COALESCE(SUM(CASE WHEN ou_correct THEN 1 ELSE 0 END), 0) AS ou_wins,
            MIN(pick_date), MAX(pick_date)
        FROM {PICKS_TABLE} WHERE evaluated = TRUE
    """)
    s = cur.fetchone()

    # Daily breakdown
    cur.execute(f"""
        SELECT
            pick_date,
            COUNT(*) FILTER (WHERE ml_correct IS NOT NULL),
            COALESCE(SUM(CASE WHEN ml_correct THEN 1 ELSE 0 END), 0),
            COUNT(*) FILTER (WHERE runline_correct IS NOT NULL),
            COALESCE(SUM(CASE WHEN runline_correct THEN 1 ELSE 0 END), 0),
            COUNT(*) FILTER (WHERE ou_correct IS NOT NULL),
            COALESCE(SUM(CASE WHEN ou_correct THEN 1 ELSE 0 END), 0)
        FROM {PICKS_TABLE}
        WHERE evaluated = TRUE
        GROUP BY pick_date
        ORDER BY pick_date DESC
    """)
    daily = cur.fetchall()

    # Season totals section
    ws2["A1"] = "MLB Model — Season Summary"
    ws2["A1"].font = Font(bold=True, size=14, name="Arial", color=DARK)
    ws2.merge_cells("A1:G1")

    summary_headers = ["Metric", "Wins", "Total", "Win %", "Break-even", "Edge"]
    for col, h in enumerate(summary_headers, 1):
        header_style(ws2.cell(row=3, column=col), h)

    if s and s[0]:
        ml_tot, ml_w, rl_tot, rl_w, ou_tot, ou_w = s[0], s[1], s[2], s[3], s[4], s[5]
        for row_idx, (label, wins, total) in enumerate([
            ("Moneyline",  ml_w, ml_tot),
            ("Run Line",   rl_w, rl_tot),
            ("Over/Under", ou_w, ou_tot),
        ], 4):
            pct = wins/total if total else 0
            edge = pct - 0.524
            ws2.cell(row=row_idx, column=1, value=label).font = Font(bold=True, name="Arial")
            ws2.cell(row=row_idx, column=2, value=int(wins))
            ws2.cell(row=row_idx, column=3, value=int(total))
            ws2.cell(row=row_idx, column=4, value=f"{round(pct*100,1)}%")
            ws2.cell(row=row_idx, column=5, value="52.4%")
            edge_cell = ws2.cell(row=row_idx, column=6, value=f"{round(edge*100,1)}%")
            edge_cell.fill = PatternFill("solid", fgColor=GREEN if edge > 0 else RED)

            for col in range(1, 7):
                ws2.cell(row=row_idx, column=col).font = Font(name="Arial", size=10)
                ws2.cell(row=row_idx, column=col).alignment = Alignment(horizontal="center")

    # Daily breakdown table
    ws2["A8"] = "Daily Breakdown"
    ws2["A8"].font = Font(bold=True, size=12, name="Arial", color=DARK)

    daily_headers = ["Date", "ML W", "ML Tot", "ML%", "RL W", "RL Tot", "RL%", "O/U W", "O/U Tot", "O/U%"]
    for col, h in enumerate(daily_headers, 1):
        header_style(ws2.cell(row=9, column=col), h)

    for row_idx, d in enumerate(daily, 10):
        pick_date, ml_tot, ml_w, rl_tot, rl_w, ou_tot, ou_w = d
        ws2.cell(row=row_idx, column=1,  value=str(pick_date))
        ws2.cell(row=row_idx, column=2,  value=int(ml_w))
        ws2.cell(row=row_idx, column=3,  value=int(ml_tot))
        ws2.cell(row=row_idx, column=4,  value=f"{round(ml_w/ml_tot*100,1)}%" if ml_tot else "—")
        ws2.cell(row=row_idx, column=5,  value=int(rl_w))
        ws2.cell(row=row_idx, column=6,  value=int(rl_tot))
        ws2.cell(row=row_idx, column=7,  value=f"{round(rl_w/rl_tot*100,1)}%" if rl_tot else "—")
        ws2.cell(row=row_idx, column=8,  value=int(ou_w))
        ws2.cell(row=row_idx, column=9,  value=int(ou_tot))
        ws2.cell(row=row_idx, column=10, value=f"{round(ou_w/ou_tot*100,1)}%" if ou_tot else "—")

        bg = GRAY if row_idx % 2 == 0 else WHITE
        for col in range(1, 11):
            cell = ws2.cell(row=row_idx, column=col)
            cell.font      = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center")
            cell.fill      = PatternFill("solid", fgColor=bg)

    for i, w in enumerate([12, 8, 8, 8, 8, 8, 8, 8, 8, 8], 1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    ws2.freeze_panes = "A10"

    wb.save(EXCEL_PATH)
    print(f"  ✅ Excel saved to: {EXCEL_PATH}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*55}")
    print(f"  daily_picks_tracker.py  |  {date.today()}")
    print(f"{'='*55}\n")

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_tables(cur)
            c.commit()

            evaluate_yesterday(cur)
            c.commit()

            save_today_picks(cur)
            c.commit()

            export_excel(cur)

        print("✅ Done.")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()