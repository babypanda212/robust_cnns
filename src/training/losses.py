import torch.nn.functional as F

def cross_entropy_loss(model, X, y, delta=0.0):
    """Cross-Entropy Loss"""
    yp = model(X + delta)
    return F.cross_entropy(yp, y)

def trades_loss(model, X, y, delta=0.0, lambda_tradeoff=1.0):
    """TRADES Loss"""
    yp_adv = model(X + delta)
    yp_clean = model(X)
    clean_loss = F.cross_entropy(yp_clean, y)
    robust_loss = F.kl_div(
        F.log_softmax(yp_adv, dim=1),
        F.softmax(yp_clean, dim=1),
        reduction="batchmean"
    )
    return clean_loss + lambda_tradeoff * robust_loss

class LossWrapper:
    def __init__(self, loss_fn, lambda_tradeoff=1.0):
        self.loss_fn = loss_fn
        self.lambda_tradeoff = lambda_tradeoff

    def __call__(self, model, X, y, delta=0.0):
        if self.loss_fn == "CE":
            return cross_entropy_loss(model, X, y, delta)
        elif self.loss_fn == "TRADES":
            return trades_loss(model, X, y, delta, self.lambda_tradeoff)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")
