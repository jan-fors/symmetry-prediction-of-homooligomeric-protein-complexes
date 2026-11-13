import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, losses, regularizations):
        """
        Loss function that allows for summed losses.
        Add pairs of factors and loss functions.
        """
        super(CombinedLoss, self).__init__()

        self.loss_weights = [factor for factor, _ in losses]
        self.losses = nn.ModuleList([loss_fn() for _, loss_fn in losses])
        
        self.reg_weights = []
        self.regularizations = nn.ModuleList()

        if regularizations is not None:
            self.reg_weights = [factor for factor, _ in regularizations]
            self.regularizations = nn.ModuleList([regu() for _, regu in regularizations])

    def forward(self, logits, target, model):
        """
        """
        loss = 0.0

        # losses
        for weight, fn in zip(self.loss_weights, self.losses):
            loss = loss + weight * fn(logits, target)

        # regularizations
        if self.regularizations != None:
            for weight, regu in zip(self.reg_weights, self.regularizations):
                loss = loss + weight * regu(model)

        return loss