# src/training/methods/weight_averaging.py

from torch.optim.swa_utils import AveragedModel

class WeightAveragingTrainer:
    def __init__(self, model_adv, optimizer):
        self.model_adv = model_adv
        self.optimizer = optimizer
        self.model_ema = AveragedModel(model_adv)

    def update_ema(self):
        """Update EMA model parameters."""
        self.model_ema.update_parameters(self.model_adv)
