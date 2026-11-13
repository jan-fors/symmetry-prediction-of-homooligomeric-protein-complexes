import torch
from typing import List
from src.utils.unique_labels import get_unique_labels
import logging

logger = logging.getLogger(__name__)


class MultiLabelEncoder():
    def __init__(self, config : dict):
        """
        Receives a list of labels and creates a series of one hot vectors.
        """
        # get amount of labels
        self.labels = get_unique_labels(config)
        self.n_labels = len(self.labels)

        self.one_hot_vectors_per_label = {}
        self.labels_per_onehot_vector = {}

        for label in self.labels:
            empty_tensor = torch.zeros(self.n_labels)
            index_of_label = self.labels.index(label)
            one_hot_tensor = empty_tensor
            one_hot_tensor[index_of_label] = 1

            self.one_hot_vectors_per_label[label] = one_hot_tensor
            list_key = ""
            for i in one_hot_tensor:
                if i == 0:
                    list_key += "0"
                else:
                    list_key += "1"
            self.labels_per_onehot_vector[list_key] = label

        # check if label groups exist
        if "label_groups" in config.keys():
            self.label_groups = config["label_groups"]
            self.label_to_labelgroup_mapping = {}
            for key in self.label_groups:
                labelgroup_labels = self.label_groups[key]
                for i in labelgroup_labels:
                    self.label_to_labelgroup_mapping[i] = key
        else:
            self.label_groups = None
            self.label_to_labelgroup_mapping = None

        logger.info("Created LabelEncoder")

    def __call__(self, labels : List[str]) -> torch.tensor:
        """
        Combines the one hot vectors for the list of labels into one vector.
        """
        one_hot_vectors = []
        for l in labels:
            if self.label_to_labelgroup_mapping != None:
                if l in self.label_to_labelgroup_mapping.keys():
                    one_hot_vectors.append(self.one_hot_vectors_per_label[self.label_to_labelgroup_mapping[l]])
                    continue

            one_hot_vectors.append(self.one_hot_vectors_per_label[l])

        return torch.sum(torch.stack(one_hot_vectors), dim=0)

    def decode(self, t : torch.tensor) -> List[str]:
        """
        """
        # generate one-hot tensors
        one_hot_vectors = []
        for i in range(t.shape[0]):
            if t[i] == 1:
                empty_tensor = torch.zeros(self.n_labels)
                empty_tensor[i] = 1
                one_hot_vectors.append(empty_tensor.tolist())

        labels = []
        for j in one_hot_vectors:
            list_key = ""
            for i in j:
                if i == 0:
                    list_key += "0"
                else:
                    list_key += "1"
            labels.append(self.labels_per_onehot_vector[list_key])

        return labels

    def get_n_labels(self):
        """
        """
        return self.n_labels

    def get_labels(self):
        """
        """
        return self.labels