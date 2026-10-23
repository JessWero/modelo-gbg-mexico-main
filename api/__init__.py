from fastapi import FastAPI
from api.routes import router as api_router

app = FastAPI(
    title="GBG México - API de Modelado Matemático",
    description="API para simulación, calibración y escenarios del modelo GBG (Gusano Barrenador del Ganado)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "mensaje": "GBG México - API de Modelado Matemático",
        "version": "1.0.0",
        "documentacion": "/docs",
        "health_check": "/api/v1/health"
    }