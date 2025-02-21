from autoattack import AutoAttack
from contextlib import redirect_stdout
import io

class AttackEvaluator:
    def __init__(self, attack_type: str, epsilon: float = 8/255, 
                 device: str = 'cuda'):
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.device = device
        
    def prepare_attack(self, model: torch.nn.Module):
        if self.attack_type == 'autoattack':
            return AutoAttack(model, norm='Linf', eps=self.epsilon, 
                            version='standard', device=self.device)
        # Add other attack types
        ...
