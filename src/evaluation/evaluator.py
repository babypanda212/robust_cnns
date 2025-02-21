from pathlib import Path
import json
import torch
from typing import Dict, Tuple
from .metrics import RobustnessMetrics
from .attacks import AttackEvaluator
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = RobustnessMetrics(device=config['device'])
        logger.info(f"Initialized evaluator on {config['device']}")

    def load_model(self, model_path: Path) -> torch.nn.Module:
        """Load trained model with architecture setup"""
        model = resnet18()
        # Modify architecture as in original notebook
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, 
                                     stride=1, padding=1, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, self.config['num_classes'])
        model.load_state_dict(torch.load(model_path))
        return model.to(self.config['device']).eval()

    def evaluate_model(self, model_path: Path, 
                      dataloader: DataLoader) -> Dict[str, float]:
        """Full evaluation pipeline"""
        model = self.load_model(model_path)
        
        results = {}
        results['clean_acc'] = self.metrics.calculate_clean_accuracy(model, dataloader)
        
        # Evaluate different attack types
        for attack in self.config['attack_types']:
            evaluator = AttackEvaluator(attack, self.config['epsilon'])
            results[f'{attack}_robust_acc'] = self.metrics.calculate_robust_accuracy(
                model, dataloader, evaluator
            )
            
        return results

# Add to evaluator.py
class ModelLoadError(Exception):
    pass

def load_model(self, model_path: Path) -> torch.nn.Module:
    try:
        model = resnet18()
        # ... architecture modifications ...
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        raise ModelLoadError(f"Model file not found at {model_path}")
    except Exception as e:
        raise ModelLoadError(f"Error loading model: {str(e)}")
    return model
