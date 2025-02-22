# src/training/methods/adversarial_training.py

from src.training.base_trainer import BaseTrainer

class AdversarialTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, device):
        super().__init__(model, optimizer, scheduler, device)

    def train_epoch(self, loader, loss_fn, attack_fn=None, **kwargs):
        """Adversarial training epoch."""
        return super().train_epoch(loader=loader, loss_fn=loss_fn, attack_fn=attack_fn, **kwargs)
