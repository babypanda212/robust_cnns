# main_evaluation.py
from pathlib import Path
from evaluation.evaluator import ModelEvaluator
from evaluation.visualization import TrainingVisualizer
import yaml

def main():
    # Load config
    with open('config/evaluation.yaml') as f:
        config = yaml.safe_load(f)
        
    evaluator = ModelEvaluator(config)
    
    # Evaluate model
    model_path = Path('weights/model_adv.pt')
    results = evaluator.evaluate_model(model_path, test_loader)
    
    print(f"Clean Accuracy: {results['clean_acc']:.2%}")
    print(f"AutoAttack Robust Accuracy: {results['autoattack_robust_acc']:.2%}")
    
    # Generate training curve
    TrainingVisualizer.plot_robust_accuracy(
        Path('logs/training_log.json'),
        save_path=Path('reports/accuracy_curve.png')
    )

if __name__ == '__main__':
    main()
