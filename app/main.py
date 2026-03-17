import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

from app.routes import router

app = FastAPI(title="MLB Predictions")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(router)