# src/training/base_trainer.py
import torch
from hydra.utils import instantiate

class BaseTrainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, loader, loss_fn, attack_fn=None, awp=None, **kwargs):
        """Generic training epoch."""
        self.model.train() # Ensure model is in training mode
        total_loss, total_err = 0., 0.
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            # 1. Attack (if applicable)
            if attack_fn:
                with torch.enable_grad():  # Crucial for attack generation!
                    delta = instantiate(attack_fn, model=self.model, X=X, y=y, **kwargs)
                X_adv = X + delta
            else:
                X_adv = X  # Use clean examples if no attack

            # 2. Perturb Weights (if AWP is applied)
            if awp:
                awp.attack_backward(X, y) #Awp attack before forward pass

            # 3. Forward Pass
            yp = self.model(X_adv)

            # 4. Calculate Loss
            loss = loss_fn(model=self.model, X=X, y=y, yp=yp, attack_fn=attack_fn, **kwargs) # Pass yp

            # 5. Backward Pass and Optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]

        if self.scheduler:
            self.scheduler.step()

        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    @torch.no_grad()
    def eval_epoch(self, loader, loss_fn, attack_fn=None, **kwargs):
        """Generic evaluation epoch."""
        self.model.eval() # Ensure model is in evaluation mode
        total_loss, total_err = 0., 0.
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            if attack_fn:
                 delta = instantiate(attack_fn, model=self.model, X=X, y=y, **kwargs)
                 yp = self.model(X + delta)

            else:
                yp = self.model(X)

            loss = loss_fn(model=self.model, X=X, y=y, yp=yp, attack_fn=attack_fn, **kwargs) # Pass yp

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)
