"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import audit, decisions, loan
from api.schemas import HealthResponse
from database.connection import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise DB tables and pre-load the ML model."""
    init_db()
    # Pre-warm the singleton predictor (loads model + SHAP explainer)
    try:
        from models.predictor import CreditRiskPredictor
        CreditRiskPredictor.get_instance()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Model not pre-loaded (run 'python main.py train' first): %s", exc
        )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Agent Credit Risk Decision System",
        description=(
            "A multi-agent system using AutoGen that collaborates to approve/deny loans, "
            "explain decisions with SHAP, and audit for fairness."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(loan.router, tags=["Loan Evaluation"])
    app.include_router(decisions.router, tags=["Decisions"])
    app.include_router(audit.router, tags=["Audit"])

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health_check():
        from models.predictor import CreditRiskPredictor
        model_loaded = CreditRiskPredictor._instance is not None
        return HealthResponse(status="ok", model_loaded=model_loaded, db_connected=True)

    return app


app = create_app()
