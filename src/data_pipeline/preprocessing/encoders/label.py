import torch
from typing import List
from src.utils.unique_labels import get_unique_labels
from pprint import pprint
import logging

logger = logging.getLogger(__name__)


class MultiLabelEncoder():
    def __init__(self, config : dict):
        """
        Receives a list of labels and creates a series of one hot vectors.
        """
        # get amount of labels
        self.labels = get_unique_labels(config) # returns all unique labels; label groups already replace the single labels

        self.n_labels = len(self.labels) # counts the total number of different labels; defines length of vector

        if config["metalabels"]:
            # add meta labels
            cx_index = self.n_labels
            dx_index = self.n_labels + 1
            hoti_index = self.n_labels + 2
            self.n_labels += 3 # CX, DX, HOTI

        # generate the onbe hot vectors per label
        self.one_hot_vectors_per_label = {}
        self.labels_per_onehot_vector = {}

        # iterate over all unique labels and create one hot vectors
        for label in self.labels:
            # create label to vector mapping
            empty_tensor = torch.zeros(self.n_labels)
            index_of_label = self.labels.index(label)
            one_hot_tensor = empty_tensor
            one_hot_tensor[index_of_label] = 1

            if config["metalabels"]:
                # check for meta labels
                if "C" in label:
                    one_hot_tensor[cx_index] = 1
                if "D" in label:
                    one_hot_tensor[dx_index] = 1
                if "H" in label or "O" in label or "T" in label or "I" in label:
                    one_hot_tensor[hoti_index] = 1

            self.one_hot_vectors_per_label[label] = one_hot_tensor

            # create vector to label mapping
            list_key = ""
            for i in one_hot_tensor:
                if i == len(one_hot_tensor) - 3 and config["metalabels"]:
                    list_key += "000"
                    break
                if i == 0:
                    list_key += "0"
                else:
                    list_key += "1"
                
            self.labels_per_onehot_vector[list_key] = label

        # if config["metalabels"]:
        #     self.labels_per_onehot_vector["0"*(self.n_labels - 3) + "100"] = "CX"
        #     self.labels_per_onehot_vector["0"*(self.n_labels - 3) + "010"] = "DX"
        #     self.labels_per_onehot_vector["0"*(self.n_labels - 3) + "001"] = "HOTI"

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

        #
        if config["metalabels"]:
            self.labels.extend(["CX", "DX", "HOTI"])

        logger.info("Created LabelEncoder")

        # pprint(self.one_hot_vectors_per_label)
        # pprint(self.labels_per_onehot_vector)

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

        res = torch.sum(torch.stack(one_hot_vectors), dim=0)
        res[res == 2] = 1

        return res

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