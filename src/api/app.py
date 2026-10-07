from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "InterVideo API"}

@app.get("/health")
def health():
    return {"status": "ok"}
