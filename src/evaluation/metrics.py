from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader

class RobustnessMetrics:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def calculate_clean_accuracy(self, model: torch.nn.Module, 
                                loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                
        return correct / total

    def calculate_robust_accuracy(self, model: torch.nn.Module, 
                                 loader: DataLoader, 
                                 attack_fn: callable) -> float:
        # Similar structure with adversarial examples
        ...
