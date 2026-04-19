"""
daily_picks_tracker.py
-----------------------
Single script that does three jobs in order:

  1. EVALUATE yesterday's picks against actual results
  2. SAVE today's predictions as picks to evaluate tomorrow
  3. EXPORT full picks history to Excel with color coding

Excel file saved to: reports/picks_tracker.xlsx
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = (
    os.getenv("DATABASE_URL", "")
    .replace("postgres://", "postgresql://", 1)
    .replace("postgresql+psycopg2://", "postgresql://", 1)
)
GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")
PRED_TABLE  = os.getenv("MLB_PREDICTIONS_TABLE", "predictions")
PICKS_TABLE = "daily_picks"

DEFAULT_OU_LINE = 8.5

REPORTS_DIR = Path(PROJECT_ROOT) / "reports"
EXCEL_PATH  = REPORTS_DIR / "picks_tracker.xlsx"

# Colors
GREEN_FILL  = "C6EFCE"
GREEN_FONT  = "276221"
RED_FILL    = "FFC7CE"
RED_FONT    = "9C0006"
YELLOW_FILL = "FFEB9C"
YELLOW_FONT = "9C5700"
BLUE_FILL   = "DEEAF1"
BLUE_FONT   = "1F4E79"
GRAY_FILL   = "F2F2F2"
DARK_HEADER = "1F4E79"
WHITE       = "FFFFFF"
LIGHT_GREEN = "EAF3DE"
LIGHT_RED   = "FCEBEB"
LIGHT_GRAY  = "F9F9F9"


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_tables(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {PICKS_TABLE} (
            id                SERIAL PRIMARY KEY,
            game_pk           BIGINT NOT NULL,
            pick_date         DATE   NOT NULL,
            home_team         TEXT,
            away_team         TEXT,
            home_sp           TEXT,
            away_sp           TEXT,
            ml_pick           TEXT,
            runline_pick      TEXT,
            ou_pick           TEXT,
            home_win_prob     DOUBLE PRECISION,
            away_win_prob     DOUBLE PRECISION,
            pred_run_diff     DOUBLE PRECISION,
            pred_total_runs   DOUBLE PRECISION,
            market_total_line DOUBLE PRECISION,
            home_ml_implied   INT,
            away_ml_implied   INT,
            home_score        INT,
            away_score        INT,
            actual_run_diff   DOUBLE PRECISION,
            actual_total      DOUBLE PRECISION,
            ml_correct        BOOLEAN,
            runline_correct   BOOLEAN,
            ou_correct        BOOLEAN,
            evaluated         BOOLEAN DEFAULT FALSE,
            evaluated_at      TIMESTAMPTZ,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (game_pk, pick_date)
        );
    """)
    cur.execute(f"ALTER TABLE {PICKS_TABLE} ADD COLUMN IF NOT EXISTS market_total_line DOUBLE PRECISION;")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_picks_date ON {PICKS_TABLE}(pick_date);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_picks_eval ON {PICKS_TABLE}(evaluated);")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sym(val):
    if val is None: return "—"
    return "✅" if val else "❌"


def pct_str(wins, total):
    if not total: return "—"
    return f"{int(wins)}/{total} ({round(wins / total * 100, 1)}%)"


def result_fill(val):
    if val is True:  return PatternFill("solid", fgColor=GREEN_FILL)
    if val is False: return PatternFill("solid", fgColor=RED_FILL)
    return PatternFill("solid", fgColor=YELLOW_FILL)


def result_font(val):
    if val is True:  return Font(name="Arial", size=10, color=GREEN_FONT, bold=True)
    if val is False: return Font(name="Arial", size=10, color=RED_FONT, bold=True)
    return Font(name="Arial", size=10, color=YELLOW_FONT)


def header_cell(cell, text, size=10):
    cell.value     = text
    cell.font      = Font(bold=True, color=WHITE, name="Arial", size=size)
    cell.fill      = PatternFill("solid", fgColor=DARK_HEADER)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def win_prob_fill(prob):
    """Color win prob cell: green >60%, yellow 50-60%, red <50%"""
    if prob is None:
        return PatternFill("solid", fgColor=GRAY_FILL)
    if prob >= 0.60:
        return PatternFill("solid", fgColor=GREEN_FILL)
    if prob >= 0.50:
        return PatternFill("solid", fgColor=YELLOW_FILL)
    return PatternFill("solid", fgColor=RED_FILL)


def win_prob_font(prob):
    if prob is None:
        return Font(name="Arial", size=10)
    if prob >= 0.60:
        return Font(name="Arial", size=10, color=GREEN_FONT, bold=True)
    if prob >= 0.50:
        return Font(name="Arial", size=10, color=YELLOW_FONT, bold=True)
    return Font(name="Arial", size=10, color=RED_FONT, bold=True)


def pct_fill(pct_val):
    """Color percentage cells in summary sheet"""
    if pct_val >= 57:
        return PatternFill("solid", fgColor=GREEN_FILL)
    if pct_val >= 52.4:
        return PatternFill("solid", fgColor=YELLOW_FILL)
    return PatternFill("solid", fgColor=RED_FILL)


def pct_font(pct_val):
    if pct_val >= 57:
        return Font(name="Arial", size=10, color=GREEN_FONT, bold=True)
    if pct_val >= 52.4:
        return Font(name="Arial", size=10, color=YELLOW_FONT, bold=True)
    return Font(name="Arial", size=10, color=RED_FONT, bold=True)


# ---------------------------------------------------------------------------
# Evaluation logic — PASS picks always return None (not graded)
# ---------------------------------------------------------------------------

def eval_ml(ml_pick, home_team, away_team, home_score, away_score):
    if None in (home_score, away_score):        return None
    if not ml_pick or ml_pick in ("PASS", ""):  return None
    if home_score == away_score:                 return None
    winner = home_team if home_score > away_score else away_team
    return ml_pick == winner


def eval_runline(runline_pick, home_team, away_team, home_score, away_score):
    if None in (home_score, away_score):                  return None
    if not runline_pick or runline_pick in ("PASS", ""):  return None
    try:
        parts = str(runline_pick).rsplit(" ", 1)
        if len(parts) != 2: return None
        team, spread_str = parts
        spread = float(spread_str)
    except Exception:
        return None
    diff = home_score - away_score
    if team == home_team:
        adjusted = diff + spread
        return None if adjusted == 0 else adjusted > 0
    if team == away_team:
        adjusted = (-diff) + spread
        return None if adjusted == 0 else adjusted > 0
    return None


def eval_ou(ou_pick, home_score, away_score, market_line, pred_total):
    if None in (home_score, away_score):          return None
    if not ou_pick or ou_pick in ("PASS", ""):    return None
    actual = home_score + away_score
    if market_line and market_line > 0:
        line = market_line
    elif pred_total and pred_total > 0:
        line = pred_total
    else:
        line = DEFAULT_OU_LINE
    if actual == line: return None
    return (ou_pick == "OVER" and actual > line) or (ou_pick == "UNDER" and actual < line)


# ---------------------------------------------------------------------------
# Predicted score helper
# ---------------------------------------------------------------------------

def predicted_scores(home_win_prob, pred_run_diff, pred_total_runs):
    """
    Back-calculate predicted home/away scores from model outputs.
    pred_total = home + away
    pred_run_diff = home - away
    → home = (total + diff) / 2
    → away = (total - diff) / 2
    """
    if pred_total_runs is None or pred_run_diff is None:
        return None, None
    home = (pred_total_runs + pred_run_diff) / 2
    away = (pred_total_runs - pred_run_diff) / 2
    return round(home, 1), round(away, 1)


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
               ml_pick, runline_pick, ou_pick,
               pred_total_runs, market_total_line
        FROM {PICKS_TABLE}
        WHERE pick_date = %s AND evaluated = FALSE
        ORDER BY game_pk
    """, (yesterday,))
    picks = cur.fetchall()

    if not picks:
        print(f"  No unevaluated picks for {yesterday}.\n")
        return

    ml_res, rl_res, ou_res = [], [], []

    for pick in picks:
        pick_id, game_pk, home_team, away_team, \
        ml_pick, runline_pick, ou_pick, pred_total, market_line = pick

        cur.execute(f"SELECT home_score, away_score FROM {GAMES_TABLE} WHERE game_pk = %s", (game_pk,))
        result = cur.fetchone()

        if not result or result[0] is None or result[1] is None:
            print(f"  {away_team} @ {home_team}: no result yet, skipping")
            continue

        home_score, away_score = result
        actual_total    = home_score + away_score
        actual_run_diff = home_score - away_score

        ml_correct = eval_ml(ml_pick, home_team, away_team, home_score, away_score)
        rl_correct = eval_runline(runline_pick, home_team, away_team, home_score, away_score)
        ou_correct = eval_ou(ou_pick, home_score, away_score, market_line, pred_total)

        if ou_pick and ou_pick not in ("PASS", ""):
            ou_line_used = market_line if (market_line and market_line > 0) else (pred_total or DEFAULT_OU_LINE)
            ou_line_src  = "Vegas" if (market_line and market_line > 0) else "Model"
            ou_display   = f"({ou_line_src} line: {ou_line_used} | actual: {actual_total})"
        else:
            ou_display = "(PASS — not graded)"

        print(f"  {away_team} ({away_score}) @ {home_team} ({home_score})")
        print(f"    ML:  {(ml_pick or '—'):<20} {sym(ml_correct)}")
        print(f"    RL:  {(runline_pick or '—'):<20} {sym(rl_correct)}")
        print(f"    O/U: {(ou_pick or '—'):<20} {sym(ou_correct)}  {ou_display}\n")

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
        """, {
            "hs": home_score, "as_": away_score,
            "ard": actual_run_diff, "at_": actual_total,
            "ml": ml_correct, "rl": rl_correct, "ou": ou_correct,
            "id": pick_id,
        })

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
    if row:
        ml_tot, ml_w, rl_tot, rl_w, ou_tot, ou_w = row
        print("  --- Season Totals ---")
        print(f"  Moneyline : {pct_str(ml_w, ml_tot)}")
        print(f"  Run Line  : {pct_str(rl_w, rl_tot)}")
        print(f"  Over/Under: {pct_str(ou_w, ou_tot)}")
        print("  Break-even: 52.4% | Sharp target: 57%+\n")


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
            p.market_total_line,
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
         home_win_prob, away_win_prob,
         run_diff_pred, total_runs_pred, market_total_line,
         home_ml_implied, away_ml_implied) = r

        ou_line_str = f"Vegas: {market_total_line}" if market_total_line else f"Model: {round(total_runs_pred,1) if total_runs_pred else '—'}"
        print(f"  {away_team} @ {home_team}  |  ML: {ml_pick} | RL: {runline_pick} | O/U: {ou_pick} ({ou_line_str})")

        data.append((
            int(game_pk), today,
            home_team, away_team, home_sp, away_sp,
            ml_pick, runline_pick, ou_pick,
            home_win_prob, away_win_prob,
            run_diff_pred, total_runs_pred, market_total_line,
            home_ml_implied, away_ml_implied,
        ))

    execute_values(cur, f"""
        INSERT INTO {PICKS_TABLE} (
            game_pk, pick_date, home_team, away_team, home_sp, away_sp,
            ml_pick, runline_pick, ou_pick,
            home_win_prob, away_win_prob,
            pred_run_diff, pred_total_runs, market_total_line,
            home_ml_implied, away_ml_implied
        ) VALUES %s
        ON CONFLICT (game_pk, pick_date) DO UPDATE SET
            ml_pick           = EXCLUDED.ml_pick,
            runline_pick      = EXCLUDED.runline_pick,
            ou_pick           = EXCLUDED.ou_pick,
            home_win_prob     = EXCLUDED.home_win_prob,
            away_win_prob     = EXCLUDED.away_win_prob,
            pred_run_diff     = EXCLUDED.pred_run_diff,
            pred_total_runs   = EXCLUDED.pred_total_runs,
            market_total_line = EXCLUDED.market_total_line,
            home_ml_implied   = EXCLUDED.home_ml_implied,
            away_ml_implied   = EXCLUDED.away_ml_implied;
    """, data)

    print(f"\n  ✅ Saved {len(data)} picks for {today}\n")


# ---------------------------------------------------------------------------
# Step 3: Export to Excel
# ---------------------------------------------------------------------------

def export_excel(cur):
    print(f"{'='*55}")
    print(f"  STEP 3 — Exporting to Excel")
    print(f"{'='*55}\n")

    cur.execute(f"""
        SELECT
            pick_date, away_team, home_team, away_sp, home_sp,
            ml_pick, home_win_prob, away_win_prob,
            runline_pick, pred_run_diff,
            ou_pick, pred_total_runs, market_total_line,
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
    thin = Side(style="thin", color="E0E0E0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # ── Sheet 1: All Picks ──────────────────────────────────────────────────
    ws = wb.active
    ws.title = "All Picks"

    headers = [
        "Date", "Matchup", "Away SP", "Home SP",
        "ML Pick", "Win Prob",
        "Pred Score",       # new: away-home predicted
        "Actual Score",     # new: away-home actual
        "RL Pick", "Run Diff",
        "O/U Pick", "Model Total", "Vegas Line",
        "ML ✓", "RL ✓", "O/U ✓",
    ]
    for col, h in enumerate(headers, 1):
        header_cell(ws.cell(row=1, column=col), h)

    ws.row_dimensions[1].height = 32

    for row_idx, r in enumerate(rows, 2):
        (pick_date, away_team, home_team, away_sp, home_sp,
         ml_pick, home_win_prob, away_win_prob,
         runline_pick, pred_run_diff,
         ou_pick, pred_total_runs, market_total_line,
         home_score, away_score,
         actual_total, actual_run_diff,
         ml_correct, runline_correct, ou_correct,
         evaluated) = r

        # Predicted score
        pred_home, pred_away = predicted_scores(home_win_prob, pred_run_diff, pred_total_runs)
        pred_score_str = f"{pred_away}-{pred_home}" if pred_home is not None else "—"

        # Actual score
        actual_score_str = f"{away_score}-{home_score}" if home_score is not None and away_score is not None else "—"

        matchup  = f"{away_team} @ {home_team}"
        win_prob = f"{round(home_win_prob * 100, 1)}%" if home_win_prob is not None else "—"

        # Row background — light green if ML correct, light red if wrong, white if pending
        if evaluated and ml_correct is True:
            row_bg = LIGHT_GREEN
        elif evaluated and ml_correct is False:
            row_bg = LIGHT_RED
        else:
            row_bg = WHITE if row_idx % 2 == 0 else LIGHT_GRAY

        values = [
            str(pick_date),
            matchup,
            away_sp or "TBD",
            home_sp or "TBD",
            ml_pick or "—",
            win_prob,
            pred_score_str,
            actual_score_str,
            runline_pick or "—",
            round(pred_run_diff, 2)     if pred_run_diff     is not None else "—",
            ou_pick or "—",
            round(pred_total_runs, 1)   if pred_total_runs   is not None else "—",
            round(market_total_line, 1) if market_total_line is not None else "—",
            "✅" if ml_correct      else ("❌" if ml_correct      is False else "—"),
            "✅" if runline_correct  else ("❌" if runline_correct  is False else "—"),
            "✅" if ou_correct       else ("❌" if ou_correct       is False else "—"),
        ]

        for col, val in enumerate(values, 1):
            cell = ws.cell(row=row_idx, column=col, value=val)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border    = border
            cell.fill      = PatternFill("solid", fgColor=row_bg)
            cell.font      = Font(name="Arial", size=10)

            # Special formatting per column
            if col == 6:  # Win prob
                cell.fill = win_prob_fill(home_win_prob)
                cell.font = win_prob_font(home_win_prob)
            elif col == 14:  # ML correct
                cell.fill = result_fill(ml_correct) if evaluated else PatternFill("solid", fgColor=GRAY_FILL)
                cell.font = result_font(ml_correct)  if evaluated else Font(name="Arial", size=10)
            elif col == 15:  # RL correct
                cell.fill = result_fill(runline_correct) if evaluated else PatternFill("solid", fgColor=GRAY_FILL)
                cell.font = result_font(runline_correct)  if evaluated else Font(name="Arial", size=10)
            elif col == 16:  # O/U correct
                cell.fill = result_fill(ou_correct) if evaluated else PatternFill("solid", fgColor=GRAY_FILL)
                cell.font = result_font(ou_correct)  if evaluated else Font(name="Arial", size=10)
            elif col == 13:  # Vegas line — blue tint
                cell.fill = PatternFill("solid", fgColor=BLUE_FILL)
                cell.font = Font(name="Arial", size=10, color=BLUE_FONT)

    widths = [12, 20, 18, 18, 12, 10, 12, 12, 14, 10, 10, 12, 11, 8, 8, 8]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.freeze_panes = "E2"

    # ── Sheet 2: Summary ────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Summary")
    ws2.sheet_view.showGridLines = False

    # Title
    ws2["B2"] = "MLB Model — Season Summary"
    ws2["B2"].font      = Font(bold=True, size=16, name="Arial", color=DARK_HEADER)
    ws2["B2"].alignment = Alignment(vertical="center")
    ws2.row_dimensions[2].height = 30
    ws2.merge_cells("B2:H2")

    # Season totals header
    season_headers = ["Metric", "Wins", "Total", "Win %", "Edge", "Status"]
    for col, h in enumerate(season_headers, 2):
        header_cell(ws2.cell(row=4, column=col), h)
    ws2.row_dimensions[4].height = 24

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
    s = cur.fetchone()

    if s:
        ml_tot, ml_w, rl_tot, rl_w, ou_tot, ou_w = s
        metrics = [
            ("Moneyline",  int(ml_w), int(ml_tot)),
            ("Run Line",   int(rl_w), int(rl_tot)),
            ("Over/Under", int(ou_w), int(ou_tot)),
        ]
        for row_i, (label, wins, total) in enumerate(metrics, 5):
            pct_val  = round(wins / total * 100, 1) if total else 0
            edge_val = round(pct_val - 52.4, 1)
            status   = "Above break-even ✓" if edge_val >= 0 else "Below break-even ✗"

            row_bg = GREEN_FILL if edge_val >= 0 else RED_FILL

            cells_vals = [label, wins, total, f"{pct_val}%", f"{'+' if edge_val >= 0 else ''}{edge_val}%", status]
            for col_i, val in enumerate(cells_vals, 2):
                cell = ws2.cell(row=row_i, column=col_i, value=val)
                cell.font      = Font(name="Arial", size=11, bold=(col_i == 2))
                cell.alignment = Alignment(horizontal="center" if col_i > 2 else "left", vertical="center")
                cell.border    = Border(
                    left=Side(style="thin", color="DDDDDD"),
                    right=Side(style="thin", color="DDDDDD"),
                    top=Side(style="thin", color="DDDDDD"),
                    bottom=Side(style="thin", color="DDDDDD"),
                )
                # Color the pct and edge columns
                if col_i == 5:  # Win %
                    cell.fill = pct_fill(pct_val)
                    cell.font = pct_font(pct_val)
                elif col_i == 6:  # Edge
                    cell.fill = PatternFill("solid", fgColor=GREEN_FILL if edge_val >= 0 else RED_FILL)
                    cell.font = Font(name="Arial", size=11, bold=True,
                                     color=GREEN_FONT if edge_val >= 0 else RED_FONT)
                elif col_i == 7:  # Status
                    cell.fill = PatternFill("solid", fgColor=GREEN_FILL if edge_val >= 0 else RED_FILL)
                    cell.font = Font(name="Arial", size=10,
                                     color=GREEN_FONT if edge_val >= 0 else RED_FONT)
                else:
                    cell.fill = PatternFill("solid", fgColor=LIGHT_GRAY)

            ws2.row_dimensions[row_i].height = 22

    # Break-even note
    note_cell = ws2.cell(row=9, column=2, value="Break-even threshold: 52.4% | Sharp target: 57%+")
    note_cell.font      = Font(name="Arial", size=10, italic=True, color="888888")
    note_cell.alignment = Alignment(horizontal="left")
    ws2.merge_cells("B9:H9")

    # Daily breakdown header
    ws2.cell(row=11, column=2, value="Daily Breakdown").font = Font(bold=True, size=13, name="Arial", color=DARK_HEADER)
    ws2.row_dimensions[11].height = 24

    daily_headers = ["Date", "ML W", "ML Tot", "ML %", "RL W", "RL Tot", "RL %", "O/U W", "O/U Tot", "O/U %"]
    for col, h in enumerate(daily_headers, 2):
        header_cell(ws2.cell(row=12, column=col), h)
    ws2.row_dimensions[12].height = 22

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

    for row_idx, d in enumerate(daily, 13):
        pick_date, ml_tot, ml_w, rl_tot, rl_w, ou_tot, ou_w = d
        mp = round(ml_w / ml_tot * 100, 1) if ml_tot else 0
        rp = round(rl_w / rl_tot * 100, 1) if rl_tot else 0
        op = round(ou_w / ou_tot * 100, 1) if ou_tot else 0

        vals = [
            str(pick_date),
            int(ml_w), int(ml_tot), f"{mp}%" if ml_tot else "—",
            int(rl_w), int(rl_tot), f"{rp}%" if rl_tot else "—",
            int(ou_w), int(ou_tot), f"{op}%" if ou_tot else "—",
        ]
        bg = LIGHT_GRAY if row_idx % 2 == 0 else WHITE
        for col_i, val in enumerate(vals, 2):
            cell = ws2.cell(row=row_idx, column=col_i, value=val)
            cell.font      = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center" if col_i > 2 else "left", vertical="center")
            cell.fill      = PatternFill("solid", fgColor=bg)
            cell.border    = Border(
                bottom=Side(style="thin", color="EEEEEE"),
                left=Side(style="thin", color="EEEEEE"),
                right=Side(style="thin", color="EEEEEE"),
            )
            # Color the pct columns
            if col_i == 5 and ml_tot:   # ML %
                cell.fill = pct_fill(mp)
                cell.font = pct_font(mp)
            elif col_i == 8 and rl_tot: # RL %
                cell.fill = pct_fill(rp)
                cell.font = pct_font(rp)
            elif col_i == 11 and ou_tot: # O/U %
                cell.fill = pct_fill(op)
                cell.font = pct_font(op)

        ws2.row_dimensions[row_idx].height = 20

    # Column widths for summary sheet
    summary_widths = {"B": 16, "C": 8, "D": 8, "E": 10, "F": 10, "G": 8, "H": 8, "I": 8, "J": 8, "K": 10}
    for col, w in summary_widths.items():
        ws2.column_dimensions[col].width = w

    ws2.freeze_panes = "B13"

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