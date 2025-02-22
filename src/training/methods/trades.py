# src/training/methods/trades.py

from src.training.base_trainer import BaseTrainer

class TRADESTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, device):
        super().__init__(model, optimizer, scheduler, device)

    def train_epoch(self, loader, loss_fn, attack_fn=None, **kwargs):
        """TRADES training epoch."""
        return super().train_epoch(loader=loader, loss_fn=loss_fn, attack_fn=attack_fn, **kwargs)
