"""
CLI entry point.

Commands:
    python main.py train                         Train XGBoost on LendingClub data
    python main.py evaluate                      Evaluate model on test set
    python main.py run --application app.json    Run agent pipeline on a single application
    python main.py serve                         Start FastAPI server
    python main.py generate-synthetic            Generate synthetic edge-case dataset
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Multi-Agent Credit Risk Decision System")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


@app.command()
def train(
    data_path: str = typer.Option("data/raw/loan.csv", help="Path to LendingClub CSV"),
    output_dir: str = typer.Option("models/artifacts/", help="Directory to save model artifacts"),
    test_size: float = typer.Option(0.2, help="Fraction of data held out for testing"),
    cv_folds: int = typer.Option(5, help="Number of cross-validation folds"),
):
    """Train the XGBoost credit default model on LendingClub data."""
    from sklearn.model_selection import train_test_split

    from config.feature_config import MODEL_FEATURES, TARGET_COLUMN
    from data.loader import LendingClubLoader
    from data.preprocessor import LoanPreprocessor
    from models.evaluator import ModelEvaluator
    from models.trainer import XGBoostTrainer

    console.print("[bold cyan]Loading data …[/bold cyan]")
    loader = LendingClubLoader(data_path)
    df = loader.load()

    console.print(f"[green]Loaded {len(df):,} rows.[/green]")

    # Preprocess
    prep = LoanPreprocessor()
    X = prep.fit_transform(df)
    y = df[TARGET_COLUMN]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    console.print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Cross-validate
    trainer = XGBoostTrainer()
    console.print("[bold cyan]Cross-validating …[/bold cyan]")
    cv_results = trainer.cross_validate(X_train, y_train, n_folds=cv_folds)
    console.print(f"CV AUC: [bold]{cv_results['auc_mean']:.4f}[/bold] ± {cv_results['auc_std']:.4f}")

    # Full train
    console.print("[bold cyan]Training final model …[/bold cyan]")
    model = trainer.train(X_train, y_train, X_val, y_val)

    # Save artifacts
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = str(Path(output_dir) / "model.json")
    prep_path = str(Path(output_dir) / "preprocessor.pkl")
    trainer.save_model(model, model_path)
    prep.save(prep_path)

    # Evaluate on test set
    import numpy as np
    evaluator = ModelEvaluator()
    y_prob = model.predict_proba(X_test)[:, 1]
    report = evaluator.generate_report(y_test.values, y_prob)
    evaluator.plot_roc_curve(y_test.values, y_prob, "evaluation/outputs/roc_curve.png")

    _print_metrics_table(report)
    console.print(f"\n[bold green]Model saved to {model_path}[/bold green]")


@app.command()
def evaluate(
    data_path: str = typer.Option("data/raw/loan.csv", help="Path to evaluation dataset"),
):
    """Evaluate the trained model on a dataset."""
    from data.loader import LendingClubLoader
    from evaluation.model_metrics import run_model_evaluation

    console.print("[bold cyan]Loading data …[/bold cyan]")
    loader = LendingClubLoader(data_path)
    df = loader.load()

    report = run_model_evaluation(df)
    _print_metrics_table(report)


@app.command("run")
def run_agent(
    application_file: str = typer.Option(
        ..., "--application", "-a", help="Path to JSON file with loan application"
    ),
    db_path: str = typer.Option("credit_risk.db", help="Path to SQLite database"),
):
    """Run the multi-agent pipeline on a single loan application JSON file."""
    from database.connection import SessionLocal, init_db
    from agents.orchestrator import run_evaluation

    console.print("[bold cyan]Initialising database …[/bold cyan]")
    init_db()

    with open(application_file) as f:
        application = json.load(f)

    console.print(f"[bold cyan]Running agent pipeline for application …[/bold cyan]")
    console.print_json(json.dumps(application, indent=2))

    db = SessionLocal()
    try:
        result = run_evaluation(application=application, db_session=db)
    finally:
        db.close()

    console.print("\n[bold green]DECISION RESULT:[/bold green]")
    console.print_json(json.dumps(result, indent=2, default=str))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the FastAPI server."""
    import uvicorn
    console.print(f"[bold green]Starting server at http://{host}:{port}[/bold green]")
    console.print(f"[dim]API docs: http://localhost:{port}/docs[/dim]")
    uvicorn.run("api.app:app", host=host, port=port, reload=reload)


@app.command("generate-synthetic")
def generate_synthetic(
    output_path: str = typer.Option("data/synthetic/edge_cases.parquet", help="Output file path"),
):
    """Generate synthetic edge-case loan applications."""
    from data.synthetic_generator import SyntheticEdgeCaseGenerator

    gen = SyntheticEdgeCaseGenerator()
    df = gen.generate_all()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    console.print(f"[green]Generated {len(df)} synthetic records → {output_path}[/green]")

    table = Table("source_type", "count")
    for source, count in df["source_type"].value_counts().items():
        table.add_row(source, str(count))
    console.print(table)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_metrics_table(report: dict) -> None:
    table = Table("Metric", "Value", title="Model Evaluation Metrics")
    for key, val in report.items():
        if isinstance(val, float):
            table.add_row(key, f"{val:.4f}")
        else:
            table.add_row(key, str(val))
    console.print(table)


if __name__ == "__main__":
    app()
