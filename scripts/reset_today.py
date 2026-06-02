import os, psycopg2
from datetime import date
from dotenv import load_dotenv

load_dotenv('/Users/leodahlen/PycharmProjects/mlb-model/.env')
DATABASE_URL = os.getenv("DATABASE_URL","").replace("postgresql+psycopg2://","postgresql://",1)
c = psycopg2.connect(DATABASE_URL)
cur = c.cursor()

today = date.today()

cur.execute("DELETE FROM predictions WHERE official_date = %s", (today,))
print(f"Cleared predictions for {today}")

cur.execute("DELETE FROM daily_picks WHERE pick_date = %s", (today,))
print(f"Cleared daily_picks for {today}")

c.commit()
cur.close()
c.close()