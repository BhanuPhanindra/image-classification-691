import numpy as np


def normalize_features(features):
    '''
       normalize_features(features) computes the normalized version
       of the features using various norms. For example if the l2 norm of the
       features are used, each column is of unit length.

    '''
    #This is L1 normalization
    for i in range(features.shape[0]):
        features[i] = features[i] / features[i].sum()
    return features
# =============================================================================
#     raise NotImplementedError() # you should implement this
# =============================================================================
