import torch
from typing import List

class MultiLabelEncoder():
    def __init__(self, labels : List[str]):
        """
        Receives a list of labels and creates a series of one hot vectors.
        """
        # get amount of labels
        n_labels = len(labels)

        self.one_hot_vectors_per_label = {}

        for label in labels:
            empty_tensor = torch.zeros(n_labels)
            index_of_label = labels.index(label)
            one_hot_tensor = empty_tensor
            one_hot_tensor[index_of_label] = 1

            self.one_hot_vectors_per_label[label] = one_hot_tensor
        

    def __call__(self, labels : List[str]) -> torch.tensor:
        """
        Combines the one hot vectors for the list of labels into one vector.
        """
        one_hot_vectors = []
        for l in labels:
            one_hot_vectors.append(self.one_hot_vectors_per_label[l])

        return torch.sum(torch.stack(one_hot_vectors), dim=0)

    def decode(self, t : torch.tensor) -> List[str]:
        """
        """
        return