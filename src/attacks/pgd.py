import torch

def pgd_linf(model, X, y, epsilon=8/255, alpha=2/255, num_iter=10):
    """PGD L-inf attack from notebook"""
    delta = torch.zeros_like(X, requires_grad=True)
    
    for _ in range(num_iter):
        loss = torch.nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha * delta.grad.detach().sign()
                         ).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    
    return delta.detach()
