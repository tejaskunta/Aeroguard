from fastapi import FastAPI

from app.api.detection import router as detection_router

# NOTE: this is a Week-1/2 snapshot of the AeroShield service, kept here only so
# the merge history (9c1d287) stays intact. The live service — API-key auth,
# PostgreSQL/PostGIS models, missions/reports/copilot routers — is in the
# separate repository tejaskunta/AeroShield. Do not extend this copy.

app = FastAPI(
    title="AeroShield API",
    description="Backend API for the AeroShield AI-based drone landmine detection and mapping system.",
    version="1.0.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "AeroShield API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


app.include_router(detection_router)
