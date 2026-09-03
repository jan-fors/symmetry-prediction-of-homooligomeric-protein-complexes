from src.data_pipeline.preprocessing.transformers.esm2_embedding_transformations import Precalc_Mean_Representations, Extract_Layer
from typing import List

TRANSFORMATION_REGSITRY = {
    "mean_representation": Precalc_Mean_Representations,
    "layer": Extract_Layer
}


def create_transformations(augmentations : List) -> List:
    """
    """
    res = []
    for aug in augmentations:
        name = aug["name"]
        transformation = TRANSFORMATION_REGSITRY[name]
        res.append(transformation(**aug["params"]))

    return res