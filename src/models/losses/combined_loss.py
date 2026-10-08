import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, losses):
        """
        Loss function that allows for summed losses.
        Add pairs of factors and loss functions.
        """
        super(CombinedLoss, self).__init__()
        self.losses = [
            (factor, loss_fn())
            for factor, loss_fn in losses
        ]

    def forward(self, logits, target):
        """
        """
        loss = 0.0

        for factor, fn in self.losses:
            loss = loss + factor * fn(logits, target)

        return loss