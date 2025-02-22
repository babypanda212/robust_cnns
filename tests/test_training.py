import unittest
import torch
from omegaconf import OmegaConf, DictConfig
import sys
import os

# Import the functions from your modularized training code
from src.training.train import (
    setup_device,
    load_data,
    instantiate_components,
    load_pretrained_weights,
    train_epoch_loop,
    main as training_main
)

# Dummy data loader that yields one dummy batch
class DummyLoader:
    def __iter__(self):
        # Return a single batch (inputs, targets) with dummy data
        # For example: inputs of shape (batch_size, 10) and targets of shape (batch_size,)
        yield torch.rand(2, 10), torch.randint(0, 2, (2,))

def dummy_create_cifar10_loaders(batch_size, debug):
    return DummyLoader(), DummyLoader(), DummyLoader()

# Monkey-patch the create_cifar10_loaders function in the loader module
import src.data.loader as loader_module
loader_module.create_cifar10_loaders = dummy_create_cifar10_loaders

# Define a dummy trainer class to simulate training without a real model
class DummyTrainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, loader, loss_fn, attack_fn, awp, sigma):
        # Return dummy training error and loss
        return 0.1, 0.5

    def eval_epoch(self, loader, loss_fn, attack_fn, sigma):
        # Return dummy validation error and loss that slightly improves over epochs
        return 0.05, 0.4

class TestTrainingModule(unittest.TestCase):
    def setUp(self):
        # Create a minimal dummy configuration using OmegaConf.
        # Adjust the targets to use simple torch.nn modules.
        self.cfg = OmegaConf.create({
            "data": {"batch_size": 2, "debug": True},
            "model": {
                "_target_": "torch.nn.Linear",
                "in_features": 10,
                "out_features": 2
            },
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "lr": 0.1
            },
            "scheduler": None,
            "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
            "attack": None,
            "awp": None,
            "training": {
                "epochs": 3,
                "patience": 2,
                "checkpoint_path": "outputs/test_model.pt",
                "sigma": 0
            }
        })
        # Ensure checkpoint directory exists for testing
        os.makedirs(os.path.dirname(self.cfg.training.checkpoint_path), exist_ok=True)

    def test_setup_device(self):
        device = setup_device()
        self.assertIsInstance(device, torch.device)

    def test_instantiate_components(self):
        device = setup_device()
        model, optimizer, scheduler = instantiate_components(self.cfg, device)
        self.assertTrue(hasattr(model, "forward"))
        self.assertIsNotNone(optimizer)
        self.assertIsNone(scheduler)

    def test_load_pretrained_weights_no_weights(self):
        device = setup_device()
        model, optimizer, scheduler = instantiate_components(self.cfg, device)
        # Remove pretrained_weights from config if present
        if "pretrained_weights" in self.cfg.model:
            del self.cfg.model.pretrained_weights
        # This function should complete without raising an error
        load_pretrained_weights(model, self.cfg, device)

    def test_train_epoch_loop(self):
        device = setup_device()
        model, optimizer, scheduler = instantiate_components(self.cfg, device)
        dummy_trainer = DummyTrainer(model, optimizer, scheduler, device)
        # Use our dummy loaders
        dummy_train_loader, dummy_val_loader, _ = dummy_create_cifar10_loaders(batch_size=2, debug=True)
        best_val_loss = train_epoch_loop(dummy_trainer, dummy_train_loader, dummy_val_loader, self.cfg)
        self.assertIsInstance(best_val_loss, float)
        self.assertLess(best_val_loss, float("inf"))

    def test_main_integration(self):
        # Integration test: run the main function with the dummy config.
        # Set sys.argv to avoid Hydra interference.
        sys.argv = ["test_main"]
        result = training_main(self.cfg)
        self.assertIsInstance(result, float)

if __name__ == "__main__":
    unittest.main()
