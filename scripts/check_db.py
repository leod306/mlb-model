import os, psycopg2
from dotenv import load_dotenv
load_dotenv('/Users/leodahlen/PycharmProjects/mlb-model/.env')

DATABASE_URL = os.getenv("DATABASE_URL","").replace("postgresql+psycopg2://","postgresql://",1)
c = psycopg2.connect(DATABASE_URL)
cur = c.cursor()

cur.execute("SELECT COUNT(*) FROM lineups")
print("Total rows:", cur.fetchone()[0])

cur.execute("SELECT official_date, COUNT(*) FROM lineups GROUP BY official_date ORDER BY official_date DESC LIMIT 5")
print("Dates:", cur.fetchall())

cur.execute("SELECT game_pk, side, COUNT(*), MIN(player_id), MAX(player_id) FROM lineups WHERE official_date = '2026-05-26' GROUP BY game_pk, side LIMIT 10")
print("Today's games:", cur.fetchall())
cur.execute("""
    SELECT game_pk, home_sp_name, home_sp_id, away_sp_name, away_sp_id 
    FROM game_probables 
    WHERE game_pk IN (822811, 822898, 823294) 
""")
print("Probables:", cur.fetchall())