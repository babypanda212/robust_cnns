import typer
from .training.base_trainer import AdversarialTrainer

app = typer.Typer()

@app.command()
def train(config_path: str = "config.yaml"):
    """Main training entry point"""
    # Load config, initialize components, and start training
    trainer = AdversarialTrainer(config_path)
    trainer.train()