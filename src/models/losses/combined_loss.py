class CombinedLoss:
    def __init__(self, losses):
        self.losses = [
            (factor, loss_fn())
            for factor, loss_fn in losses
        ]

    def __call__(self, logits, target):
        loss = 0.0

        for factor, fn in self.losses:
            loss = loss + factor * fn(logits, target)

        return loss