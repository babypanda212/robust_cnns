# src/training/train.py

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from src.data.loader import create_cifar10_loaders  # Assuming you have this
from src.training.base_trainer import BaseTrainer
from src.models.resnet18 import get_resnet18  # Import the function
from src.attacks.attacks import pgd_linf, fgsm
from src.training.awp import AWP #ADD

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loaders
    dataloader_train, dataloader_val, dataloader_test = create_cifar10_loaders(
        batch_size=cfg.data.batch_size, debug=cfg.data.debug)

    # 3. Model, Optimizer, Scheduler
    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer) if cfg.get("scheduler") else None

    # 4. Load Pretrained Weights (if specified)
    if cfg.get("model", {}).get("pretrained_weights"): #Access pretrained weights
        try:
            pretrained_weights = torch.load(cfg.model.pretrained_weights, map_location=device)
            model.load_state_dict(pretrained_weights)
            print(f"Loaded pretrained weights from {cfg.model.pretrained_weights}")
        except FileNotFoundError:
            print(f"Warning: Pretrained weights not found at {cfg.model.pretrained_weights}")
    else:
        print("No pretrained weights specified. Training from scratch.")

    # 5. Trainer
    trainer = BaseTrainer(model, optimizer, scheduler, device)

    # 6. Training Loop
    for epoch in range(cfg.training.epochs):
        train_err, train_loss = trainer.train_epoch(
            loader=dataloader_train,
            loss_fn=instantiate(cfg.loss),
            attack_fn=instantiate(cfg.attack) if cfg.attack else None,
            awp=instantiate(cfg.awp) if cfg.awp else None,
            sigma=cfg.training.get("sigma", 0)
        )
        val_err, val_loss = trainer.eval_epoch(
            loader=dataloader_val,
            loss_fn=instantiate(cfg.loss),
            attack_fn=instantiate(cfg.attack) if cfg.attack else None,
            sigma=cfg.training.get("sigma", 0)
        )

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Error = {val_err:.4f}")

if __name__ == "__main__":
    main()
