from app.db import engine
from sqlalchemy import text

with engine.begin() as conn:
    # Graded results
    r = conn.execute(text("SELECT result, COUNT(*) FROM player_props WHERE result IS NOT NULL GROUP BY result")).fetchall()
    print("Graded counts:", list(r))

    # Total and by date
    r2 = conn.execute(text("SELECT COUNT(*) FROM player_props")).scalar()
    print("Total rows:", r2)

    r3 = conn.execute(text("""
        SELECT prop_date::text, 
               COUNT(*) as total,
               COUNT(*) FILTER (WHERE result IS NOT NULL) as graded
        FROM player_props 
        GROUP BY prop_date 
        ORDER BY prop_date DESC 
        LIMIT 5
    """)).fetchall()
    print("By date (total / graded):", [(r[0], r[1], r[2]) for r in r3])
