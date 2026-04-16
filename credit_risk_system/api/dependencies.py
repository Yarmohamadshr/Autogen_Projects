"""FastAPI dependency injection providers."""

from database.connection import get_db, SessionLocal
from models.predictor import CreditRiskPredictor


def get_db_session():
    """Yield a database session."""
    yield from get_db()


def get_predictor() -> CreditRiskPredictor:
    """Return the singleton CreditRiskPredictor."""
    return CreditRiskPredictor.get_instance()
