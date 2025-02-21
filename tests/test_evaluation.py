# tests/test_evaluation.py
def test_clean_accuracy():
    config = {...}
    evaluator = ModelEvaluator(config)
    dummy_model = create_test_model()
    dummy_loader = create_test_loader()
    
    acc = evaluator.metrics.calculate_clean_accuracy(dummy_model, dummy_loader)
    assert 0 <= acc <= 1
