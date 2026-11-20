import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

class MultiLabelRankingLossWithSoftTarget(nn.Module):
    """
    Copied from github.com/microsoft/seq2symm
    """
    def __init__(self, margin=1.0):
        super(MultiLabelRankingLossWithSoftTarget, self).__init__()
        self.margin = margin
    def forward(self, logits, target):
        n_classes = target.shape[1]
        assert (n_classes > 1)
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        # Calculate ranking loss for each sample
        ranking_loss = 0
        for i in range(logits.shape[0]):            
            # Generate all pairs of elements
            pairs = list(product(range(n_classes), repeat=2))
            # Filter pairs where x > y
            valid_pairs = torch.tensor([(probs[i, x], probs[i, y]) for x, y in pairs if target[i][x] > target[i][y]])
            # Calculate pairwise ranking loss
            ranking_loss += F.margin_ranking_loss(input1=valid_pairs[:,0], 
                                                input2=valid_pairs[:,1], 
                                                target=torch.ones(valid_pairs.shape[0]), margin=self.margin)
        # Average the ranking loss over the batch
        ranking_loss /= logits.shape[0]
        return ranking_loss

class MultiLabelRankingLossWithIndicatorTarget(nn.Module):
    """
    Copied from github.com/microsoft/seq2symm
    """
    def __init__(self, margin=1.0):
        super(MultiLabelRankingLossWithIndicatorTarget, self).__init__()
        self.margin = margin
    def forward(self, logits, target):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        # Calculate ranking loss for each sample
        ranking_loss = 0
        for i in range(logits.shape[0]):
            # Indices of positive and negative labels
            positive_indices = target[i].nonzero().view(-1)
            negative_indices = (1 - target[i]).nonzero().view(-1)         
            positive_vals = [probs[i,p] for p in positive_indices]
            negative_vals = [probs[i,n] for n in negative_indices]
            # Calculate pairwise ranking loss for positive and negative pairs
            pairs = torch.tensor(list(product(positive_vals, negative_vals)))  
            ranking_loss += F.margin_ranking_loss(input1=pairs[:,0],  ## max(0, input1-input2)
                                                input2=pairs[:,1], 
                                                target=torch.ones(pairs.shape[0]), margin=self.margin)
        # Average the ranking loss over the batch
        ranking_loss /= logits.shape[0]
        return ranking_loss