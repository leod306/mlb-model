from app.db import engine
from sqlalchemy import text

with engine.begin() as conn:
    # Lineup count
    n_lineups = conn.execute(text("SELECT COUNT(*) FROM lineups")).scalar()
    print(f"Lineups rows: {n_lineups}")

    # BvP count
    try:
        n_bvp = conn.execute(text("SELECT COUNT(*) FROM batter_vs_pitcher")).scalar()
        print(f"BvP rows: {n_bvp}")
        if n_bvp > 0:
            row = conn.execute(text("SELECT batter_id, pitcher_id, pa, avg FROM batter_vs_pitcher LIMIT 1")).fetchone()
            print(f"Sample BvP: batter={row[0]} pitcher={row[1]} pa={row[2]} avg={row[3]}")
    except Exception as e:
        print(f"BvP table error: {e}")
