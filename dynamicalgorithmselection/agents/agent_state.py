import numpy as np
from pflacco.classical_ela_features import (
    calculate_ela_meta,  # Meta-Model (Linear/Quadratic fit)
    calculate_ela_distribution,  # Skewness, Kurtosis of fitness
    calculate_nbc,  # Nearest Better Clustering
    calculate_dispersion,  # Dispersion of good solutions
    calculate_information_content,  # Information Content
)

from dynamicalgorithmselection.NeurELA.NeurELA import feature_embedder


def get_state_representation_model(name: str):
    if name == "NeurELA":
        return feature_embedder
    elif name == "ELA":
        return ela_state_representation
    else:
        raise ValueError("incorrect state representation")


def ela_state_representation(x, y):
    meta_feats = calculate_ela_meta(x, y)
    try:
        distr_feats = calculate_ela_distribution(x, y)
    except np.linalg.LinAlgError:
        # If population is converged (variance is 0), these features are undefined.
        # We return a dictionary of 0s or NaNs so the agent receives a consistent input shape.
        # Note: Check what keys your agent expects. These are the standard keys for version 1.2.2.
        distr_feats = {
            "ela_distr.skewness": 0,
            "ela_distr.kurtosis": 0,
            "ela_distr.number_of_peaks": 1,
        }
    except Exception as e:
        # Catch other potential errors (like too small sample size)
        print(f"Warning: ELA distribution failed: {e}")
        distr_feats = {
            "ela_distr.skewness": 0,
            "ela_distr.kurtosis": 0,
            "ela_distr.number_of_peaks": 1,
        }
    nbc_feats = calculate_nbc(x, y)
    disp_feats = calculate_dispersion(x, y)
    ic_feats = calculate_information_content(x, y)

    all_features = {**meta_feats, **distr_feats, **nbc_feats, **disp_feats, **ic_feats}
    return np.array(list(all_features.values()), dtype=np.float32)
