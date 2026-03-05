# app/main.py
import os
import psycopg2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

app = FastAPI(title="MLB Predictions")

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

def conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)

def team_logo(team_id: int | None) -> str | None:
    if not team_id:
        return None
    # MLB static cap logo (works well on dark backgrounds)
    return f"https://www.mlbstatic.com/team-logos/team-cap-on-dark/{team_id}.svg"

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MLB Predictions</title>
<style>
  :root{--bg:#0b1220;--card:#101b33;--text:#e8eefc;--muted:#a6b3d1;--accent:#4f8cff;--border:rgba(255,255,255,.08);--shadow:0 10px 30px rgba(0,0,0,.35);}
  *{box-sizing:border-box}
  body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;background:radial-gradient(1200px 600px at 20% -10%, rgba(79,140,255,.25), transparent 60%),radial-gradient(900px 500px at 95% 0%, rgba(49,208,170,.18), transparent 55%),var(--bg);color:var(--text);}
  .wrap{max-width:1100px;margin:0 auto;padding:28px 18px 60px;}
  header{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-bottom:18px;}
  h1{margin:0;font-size:28px;letter-spacing:.2px;}
  .sub{margin:6px 0 0;color:var(--muted);font-size:13px;}
  .pill{border:1px solid var(--border);background:rgba(255,255,255,.04);padding:8px 10px;border-radius:999px;color:var(--muted);font-size:12px;white-space:nowrap;}
  .toolbar{display:flex;gap:10px;align-items:center;justify-content:space-between;margin:14px 0 18px;flex-wrap:wrap;}
  .search{flex:1;min-width:240px;display:flex;gap:10px;align-items:center;border:1px solid var(--border);background:rgba(255,255,255,.04);border-radius:12px;padding:10px 12px;}
  .search input{width:100%;border:none;outline:none;background:transparent;color:var(--text);font-size:14px;}
  .btn{border:1px solid var(--border);background:rgba(255,255,255,.04);color:var(--text);padding:10px 12px;border-radius:12px;cursor:pointer;}
  .btn:hover{background:rgba(255,255,255,.07);}
  .grid{display:grid;grid-template-columns:1fr;gap:12px;}
  .card{background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.03));border:1px solid var(--border);border-radius:18px;box-shadow:var(--shadow);overflow:hidden;}
  .top{display:flex;justify-content:space-between;gap:14px;padding:14px 16px;background:rgba(0,0,0,.10);align-items:center;}
  .left{display:flex;gap:12px;align-items:center;}
  .logos{display:flex;gap:10px;align-items:center;}
  .logo{width:34px;height:34px;display:grid;place-items:center;border:1px solid var(--border);border-radius:10px;background:rgba(255,255,255,.04);overflow:hidden;}
  .logo img{width:30px;height:30px;}
  .matchup{font-size:18px;font-weight:800;}
  .meta{color:var(--muted);font-size:12px;margin-top:4px;}
  .right{min-width:320px;display:flex;flex-direction:column;align-items:flex-end;gap:8px;}
  .row2{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end;}
  .chip{border:1px solid var(--border);background:rgba(255,255,255,.04);border-radius:999px;padding:6px 10px;color:var(--muted);font-size:12px;}
  .pick{font-weight:800;color:var(--text);}
  .mid{display:grid;grid-template-columns: 1fr 1fr;gap:12px;padding:12px 16px 16px;}
  .box{border:1px solid var(--border);border-radius:14px;background:rgba(255,255,255,.03);padding:12px;}
  .box h3{margin:0 0 8px;font-size:12px;color:var(--muted);font-weight:700;letter-spacing:.3px;text-transform:uppercase;}
  .box .big{font-size:16px;font-weight:800;}
  .small{color:var(--muted);font-size:12px;margin-top:6px;}
  .empty{border:1px dashed var(--border);color:var(--muted);padding:16px;border-radius:16px;background:rgba(255,255,255,.03);}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>MLB Predictions</h1>
      <div class="sub">Moneyline • Run Line • Totals • Probable Starters</div>
    </div>
    <div class="pill" id="status">Loading…</div>
  </header>

  <div class="toolbar">
    <div class="search">
      <span style="color:var(--muted);font-size:12px;">Search</span>
      <input id="q" placeholder="Type a team (e.g., BOS, NYY)…"/>
    </div>
    <button class="btn" id="refresh">Refresh</button>
    <div class="pill">API: <code>/api/predictions/today</code></div>
  </div>

  <div id="list" class="grid"></div>
</div>

<script>
  const listEl = document.getElementById("list");
  const statusEl = document.getElementById("status");
  const qEl = document.getElementById("q");

  function fmtML(x){
    if (x === null || x === undefined) return "—";
    const n = Number(x);
    if (Number.isNaN(n)) return "—";
    return (n > 0 ? "+" : "") + n;
  }
  function pct(x){
    if (x === null || x === undefined) return "—";
    const n = Number(x);
    if (Number.isNaN(n)) return "—";
    return Math.round(n * 100) + "%";
  }
  function num(x, d=2){
    if (x === null || x === undefined) return "—";
    const n = Number(x);
    if (Number.isNaN(n)) return "—";
    return n.toFixed(d);
  }

  function render(items){
    const q = (qEl.value || "").trim().toUpperCase();
    const filtered = q ? items.filter(x =>
      (x.home_team||"").toUpperCase().includes(q) ||
      (x.away_team||"").toUpperCase().includes(q)
    ) : items;

    if (!filtered.length){
      listEl.innerHTML = '<div class="empty">No games found.</div>';
      return;
    }

    listEl.innerHTML = filtered.map(g => `
      <div class="card">
        <div class="top">
          <div class="left">
            <div class="logos">
              <div class="logo">${g.away_logo ? `<img src="${g.away_logo}" alt="${g.away_team}"/>` : g.away_team}</div>
              <div class="logo">${g.home_logo ? `<img src="${g.home_logo}" alt="${g.home_team}"/>` : g.home_team}</div>
            </div>
            <div>
              <div class="matchup">${g.away_team} @ ${g.home_team}</div>
              <div class="meta">Date: ${g.date} • game_pk: ${g.game_pk}</div>
              <div class="meta">Probables: ${g.away_sp || "TBD"} vs ${g.home_sp || "TBD"}</div>
            </div>
          </div>

          <div class="right">
            <div class="row2">
              <div class="chip"><span class="pick">ML:</span> ${g.ml_pick} • ${pct(g.home_win_prob)} home</div>
              <div class="chip"><span class="pick">Implied:</span> H ${fmtML(g.home_ml_implied)} / A ${fmtML(g.away_ml_implied)}</div>
            </div>
            <div class="row2">
              <div class="chip"><span class="pick">Run Line:</span> ${g.runline_pick} (margin ${num(g.run_diff_pred,2)})</div>
              <div class="chip"><span class="pick">Total:</span> ${g.ou_pick} • pred ${num(g.total_runs_pred,2)}</div>
            </div>
          </div>
        </div>

        <div class="mid">
          <div class="box">
            <h3>Moneyline</h3>
            <div class="big">${g.ml_pick}</div>
            <div class="small">Home win prob: ${pct(g.home_win_prob)} • Away: ${pct(g.away_win_prob)}</div>
          </div>
          <div class="box">
            <h3>Run Line & Total</h3>
            <div class="big">${g.runline_pick} • ${g.ou_pick}</div>
            <div class="small">Pred margin: ${num(g.run_diff_pred,2)} • Pred total: ${num(g.total_runs_pred,2)}</div>
          </div>
        </div>
      </div>
    `).join("");
  }

  async function load(){
    statusEl.textContent = "Loading…";
    const res = await fetch("/api/predictions/today", { cache: "no-store" });
    const data = await res.json();
    statusEl.textContent = `Loaded ${Array.isArray(data) ? data.length : 0} game(s)`;
    render(Array.isArray(data) ? data : []);
  }

  document.getElementById("refresh").addEventListener("click", load);
  qEl.addEventListener("input", load);
  setInterval(load, 60000);
  load();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(PAGE)

@app.get("/api/predictions/today")
def api_today():
    sql = """
    WITH d AS (
      SELECT COALESCE(
        (SELECT CURRENT_DATE WHERE EXISTS (SELECT 1 FROM predictions WHERE official_date = CURRENT_DATE)),
        (SELECT MIN(official_date) FROM predictions WHERE official_date >= CURRENT_DATE)
      ) AS day
    )
    SELECT
      p.game_pk,
      p.official_date,
      p.away_team, p.home_team,
      p.home_win_prob, p.away_win_prob,
      p.home_ml_implied, p.away_ml_implied,
      p.run_diff_pred, p.total_runs_pred,
      p.ml_pick, p.runline_pick, p.ou_pick,
      g.home_team_id, g.away_team_id,
      pr.home_sp_name, pr.away_sp_name
    FROM predictions p
    JOIN d ON p.official_date = d.day
    LEFT JOIN games g ON g.game_pk = p.game_pk
    LEFT JOIN game_probables pr ON pr.game_pk = p.game_pk
    ORDER BY p.game_pk;
    """
    with conn() as c:
        with c.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    out = []
    for r in rows:
        game_pk = r[0]
        day = r[1]
        away_team, home_team = r[2], r[3]
        home_win_prob, away_win_prob = r[4], r[5]
        home_ml, away_ml = r[6], r[7]
        run_diff_pred, total_runs_pred = r[8], r[9]
        ml_pick, runline_pick, ou_pick = r[10], r[11], r[12]
        home_id, away_id = r[13], r[14]
        home_sp, away_sp = r[15], r[16]

        out.append({
            "game_pk": int(game_pk),
            "date": str(day),
            "away_team": away_team,
            "home_team": home_team,
            "home_win_prob": float(home_win_prob) if home_win_prob is not None else None,
            "away_win_prob": float(away_win_prob) if away_win_prob is not None else None,
            "home_ml_implied": int(home_ml) if home_ml is not None else None,
            "away_ml_implied": int(away_ml) if away_ml is not None else None,
            "run_diff_pred": float(run_diff_pred) if run_diff_pred is not None else None,
            "total_runs_pred": float(total_runs_pred) if total_runs_pred is not None else None,
            "ml_pick": ml_pick,
            "runline_pick": runline_pick,
            "ou_pick": ou_pick,
            "home_logo": team_logo(home_id),
            "away_logo": team_logo(away_id),
            "home_sp": home_sp,
            "away_sp": away_sp,
        })

    return JSONResponse(out)