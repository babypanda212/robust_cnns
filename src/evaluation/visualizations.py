import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

class TrainingVisualizer:
    @staticmethod
    def plot_robust_accuracy(log_path: Path, save_path: Path = None):
        with open(log_path, 'r') as f:
            log = json.load(f)
            
        plt.figure(figsize=(10, 6))
        plt.plot(1 - np.array(log['adv_errors']), label='Robust Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Robust Accuracy Progress')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
