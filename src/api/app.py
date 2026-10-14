from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..database.migrate import migrate_up
from .routes import cv, jd, pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    migrate_up()
    yield


app = FastAPI(
    title="Intervideo-Pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(pipeline.router)
app.include_router(cv.router)
app.include_router(jd.router)

@app.get("/")
def read_root():
    return {"message": "Intervideo-Pipeline API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}
