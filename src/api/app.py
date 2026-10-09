from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import pipeline

app = FastAPI(title="Recroot API", version="1.0.0")

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

@app.get("/")
def read_root():
    return {"message": "Recroot API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}
