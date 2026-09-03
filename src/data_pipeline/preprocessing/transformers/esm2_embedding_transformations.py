

class Precalc_Mean_Representations():
    def __init__(self, layer : int):
        """
        """
        self.layer = layer

    def __call__(self, sample):
        """
        """
        return sample["mean_representations"][self.layer]

class Extract_Layer():
    def __init__(self, layer : int):
        """
        """
        self.layer = layer

    def __call__(self, sample):
        pass