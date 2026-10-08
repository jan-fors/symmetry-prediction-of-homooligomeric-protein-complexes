import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, losses, regularizations):
        """
        Loss function that allows for summed losses.
        Add pairs of factors and loss functions.
        """
        super(CombinedLoss, self).__init__()
        self.losses = [
            (factor, loss_fn())
            for factor, loss_fn in losses
        ]

        if regularizations != None:
            self.regularizations = [
                (factor, regu())
                for factor, regu in regularizations
            ]
        else:
            self.regularizations = None

    def forward(self, logits, target, model):
        """
        """
        loss = 0.0

        # losses
        for factor, fn in self.losses:
            loss = loss + factor * fn(logits, target)

        # regularizations
        if self.regularizations != None:
            for factor, regu in self.regularizations:
                loss = loss + factor * regu(model)

        return loss