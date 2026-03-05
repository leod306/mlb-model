from flask import Flask, jsonify
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

DATABASE_URL = os.getenv("DATABASE_URL").replace("postgresql+psycopg2://", "postgresql://")


def conn():
    return psycopg2.connect(DATABASE_URL)


@app.route("/api/predictions/today")
def today_predictions():

    sql = """
    SELECT
        game_pk,
        official_date,
        away_team,
        home_team,
        prediction,
        win_probability
    FROM predictions
    WHERE official_date = CURRENT_DATE
    ORDER BY game_pk
    """

    c = conn()
    cur = c.cursor()

    cur.execute(sql)
    rows = cur.fetchall()

    results = []

    for r in rows:
        results.append({
            "game_pk": r[0],
            "date": str(r[1]),
            "away_team": r[2],
            "home_team": r[3],
            "prediction": r[4],
            "win_probability": float(r[5])
        })

    cur.close()
    c.close()

    return jsonify(results)


if __name__ == "__main__":
    app.run(port=5001, debug=True)