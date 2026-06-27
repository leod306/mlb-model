from app.db import engine
from sqlalchemy import text
with engine.begin() as conn:
    rows = conn.execute(text("SELECT player_name FROM lineups LIMIT 10")).fetchall()
    for r in rows:
        print(r[0])
