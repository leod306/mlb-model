from app.db import engine
from sqlalchemy import text

with engine.connect() as conn:
    # Total rows
    r = conn.execute(text("SELECT COUNT(*) FROM player_props")).scalar()
    print("Total player_props rows:", r)

    # Check columns
    cols = conn.execute(text("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'player_props'
        ORDER BY ordinal_position
    """)).fetchall()
    print("Columns:", [c[0] for c in cols])

    # Check if result column has data
    try:
        r2 = conn.execute(text("SELECT result, COUNT(*) FROM player_props WHERE result IS NOT NULL GROUP BY result")).fetchall()
        print("Graded counts:", list(r2))
    except Exception as e:
        print("Error querying result:", e)

    # By date
    try:
        r3 = conn.execute(text("""
            SELECT prop_date::text, COUNT(*) as total,
                   COUNT(*) FILTER (WHERE result IS NOT NULL) as graded
            FROM player_props GROUP BY prop_date ORDER BY prop_date DESC LIMIT 5
        """)).fetchall()
        print("By date (total/graded):", [(x[0], x[1], x[2]) for x in r3])
    except Exception as e:
        print("Error querying by date:", e)
