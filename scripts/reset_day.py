import os
import psycopg2

# 👇 ADD THIS
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(
    DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://", 1)
)

cur = conn.cursor()

cur.execute("""
UPDATE daily_picks
SET
    evaluated = FALSE,
    evaluated_at = NULL,
    ml_correct = NULL,
    runline_correct = NULL,
    ou_correct = NULL,
    home_score = NULL,
    away_score = NULL,
    actual_run_diff = NULL,
    actual_total = NULL
WHERE pick_date = %s;
""", ("2026-04-07",))

conn.commit()
cur.close()
conn.close()

print("✅ Reset complete")