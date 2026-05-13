from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="AI Academic Advisor",
    version="0.1.0",
    description="Local-first academic planning assistant backend.",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "AI Academic Advisor",
        "status": "running",
        "docs": "/docs",
    }
