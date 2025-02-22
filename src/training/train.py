import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import sys
import os

from src.data.loader import create_cifar10_loaders
from src.training.base_trainer import BaseTrainer
from src.models.resnet18 import get_resnet18
from src.attacks.attacks import pgd_linf, fgsm
from src.training.methods.weight_perturbation import AWP

def setup_device():
    """Determine the device to run on (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(cfg: DictConfig):
    """Create and return the training, validation, and test loaders."""
    try:
        return create_cifar10_loaders(
            batch_size=cfg.data.batch_size,
            debug=cfg.data.debug
        )
    except Exception as e:
        print(f"Error creating data loaders: {e}", file=sys.stderr)
        sys.exit(1)

def instantiate_components(cfg: DictConfig, device: torch.device):
    """Instantiate the model, optimizer, and scheduler from the configuration."""
    try:
        model = instantiate(cfg.model).to(device)
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer) if cfg.get("scheduler") else None
        return model, optimizer, scheduler
    except Exception as e:
        print(f"Error instantiating model/optimizer/scheduler: {e}", file=sys.stderr)
        sys.exit(1)

def load_pretrained_weights(model: torch.nn.Module, cfg: DictConfig, device: torch.device):
    """Load pretrained weights into the model if specified in the config."""
    if cfg.get("model", {}).get("pretrained_weights"):
        try:
            pretrained_weights = torch.load(cfg.model.pretrained_weights, map_location=device)
            model.load_state_dict(pretrained_weights)
            print(f"Loaded pretrained weights from {cfg.model.pretrained_weights}")
        except FileNotFoundError:
            print(f"Warning: Pretrained weights not found at {cfg.model.pretrained_weights}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No pretrained weights specified. Training from scratch.")

def train_epoch_loop(trainer: BaseTrainer, dataloader_train, dataloader_val, cfg: DictConfig):
    """Run the training loop with early stopping and checkpointing."""
    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg.training.get("patience", 10)
    checkpoint_path = cfg.training.get("checkpoint_path", "outputs/best_model.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(cfg.training.epochs):
        try:
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

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Error = {val_err:.4f}")

            # Checkpointing: save model if validation loss improves.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                try:
                    torch.save(trainer.model.state_dict(), checkpoint_path)
                    print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
                except Exception as save_e:
                    print(f"Error saving checkpoint: {save_e}", file=sys.stderr)
            else:
                patience_counter += 1
                print(f"No improvement. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

            if trainer.scheduler:
                trainer.scheduler.step()

        except Exception as e:
            print(f"Error during training/validation at epoch {epoch+1}: {e}", file=sys.stderr)
            sys.exit(1)

    return best_val_loss

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        # 1. Setup device
        device = setup_device()

        # 2. Load data loaders
        dataloader_train, dataloader_val, dataloader_test = load_data(cfg)

        # 3. Instantiate model, optimizer, scheduler
        model, optimizer, scheduler = instantiate_components(cfg, device)

        # 4. Load pretrained weights if available
        load_pretrained_weights(model, cfg, device)

        # 5. Create trainer instance
        trainer = BaseTrainer(model, optimizer, scheduler, device)

        # 6. Train loop with early stopping and checkpointing
        best_val_loss = train_epoch_loop(trainer, dataloader_train, dataloader_val, cfg)

        # 7. Evaluate on the test set
        test_err, test_loss = trainer.eval_epoch(
            loader=dataloader_test,
            loss_fn=instantiate(cfg.loss),
            attack_fn=instantiate(cfg.attack) if cfg.attack else None,
            sigma=cfg.training.get("sigma", 0)
        )
        print(f"Test Loss = {test_loss:.4f}, Test Error = {test_err:.4f}")

        return best_val_loss

    except Exception as e:
        print(f"Unexpected error in main: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
