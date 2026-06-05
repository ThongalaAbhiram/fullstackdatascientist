from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .competitor_analysis.competitor_endpoint import router as competitor_router

try:
    # ANN prediction app (heavy imports: pandas, numpy, tensorflow). Importing
    # it at module load time mirrors running the service as a single process.
    from .models.ANN_Model import predict_endpoint as ann_module
except Exception:
    ann_module = None


app = FastAPI(
    title="Startup Validation API",
    version="0.1.0",
    description="FastAPI backend for startup analysis, CRUD, prediction, and competition insights.",
)

# Minimal CORS so frontend + external tooling can call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Primary API router (CRUD, health, predict proxy that uses internal store)
app.include_router(api_router)

# Competitor analysis router
app.include_router(competitor_router, prefix="/competitors", tags=["competitors"])

# Mount the ANN model app if available. This exposes the model at /ann/*
if ann_module is not None and hasattr(ann_module, "app"):
    app.mount("/ann", ann_module.app)


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {"message": "Startup Validation API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
