import os

import torch
import copy
from dynamicalgorithmselection.NeurELA.feature_extractor import Feature_Extractor
import pickle


def vector2nn(x, net):
    assert len(x) == sum([param.nelement() for param in net.parameters()]), (
        "dim of x and net not match!"
    )
    x_copy = copy.deepcopy(x)
    params = net.parameters()
    ptr = 0
    for v in params:
        num_of_params = v.nelement()
        temp = torch.FloatTensor(x_copy[ptr : ptr + num_of_params])
        v.data = temp.reshape(v.shape)
        ptr += num_of_params
    return net


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_path = os.path.join(BASE_DIR, "NeurELA.pkl")

feature_extractor_weights = load_data(load_path)

feature_embedder = vector2nn(feature_extractor_weights, Feature_Extractor())
