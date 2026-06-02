import os, psycopg2
from dotenv import load_dotenv
load_dotenv('/Users/leodahlen/PycharmProjects/mlb-model/.env')
DATABASE_URL = os.getenv("DATABASE_URL","").replace("postgresql+psycopg2://","postgresql://",1)
c = psycopg2.connect(DATABASE_URL)
cur = c.cursor()

cur.execute("DELETE FROM batter_vs_pitcher WHERE pa = 0")
c.commit()
print("Deleted empty rows")

cur.execute("SELECT COUNT(*) FROM batter_vs_pitcher")
print("Remaining BvP rows:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM batter_vs_pitcher WHERE pa >= 3")
print("Usable rows (pa>=3):", cur.fetchone()[0])

cur.close()
c.close()