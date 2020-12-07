import numpy as np
from collections import Counter
from math import log2

def _convert_numparray(arr):
    if isinstance(arr, np.ndarray):
        return arr
    else:
        return np.array(arr)

class BaseCriterion():

    def __init__(self, y, y_idx):
        self.y = y
        self.y_idx = y_idx

class gini(BaseCriterion):
    """
    Evaluate the impurity of the current node, i.e.,
    the impurity of samples[:] using the Gini criterion
    """

    def __init__(self,
                 y,
                 y_idx
                 ):
        super().__init__(
            y = y,
            y_idx = y_idx
        )

        unique_label, unique_label_count = np.unique(self.y[self.y_idx], return_counts=True)
        impurity = 1.0
        for i in range(len(unique_label)):
            p_i = unique_label_count[i] / sum(unique_label_count)
            impurity -= p_i ** 2
        self.impurity = impurity

class entropy(BaseCriterion):

    """
    Evaluate the impurity of the current node, i.e.,
    the impurity of samples[:] using the entropy criterion
    """

    def __init__(self,
                 y,
                 y_idx
                 ):
        super().__init__(
            y=y,
            y_idx=y_idx
        )

        unique_label, unique_label_count = np.unique(self.y[self.y_idx], return_counts=True)
        impurity = 1.0
        for i in range(len(unique_label)):
            p_i = unique_label_count[i] / sum(unique_label_count)
            impurity -= p_i * log2(p_i)
        self.impurity = impurity

class mse(BaseCriterion):
    """
    Evaluate the impurity of the current node, i.e.,
    the impurity of samples[:] using the mse criterion
    """

    def __init__(self,
                 y,
                 y_idx
                 ):
        super().__init__(
            y=y,
            y_idx=y_idx
        )

        ybar = np.mean(self.y[self.y_idx])
        self.impurity = np.power(self.y - ybar, 2).mean()

class mae(BaseCriterion):
    """
    Evaluate the impurity of the current node, i.e.,
    the impurity of samples[:] using the mae criterion
    """

    def __init__(self,
                 y,
                 y_idx
                 ):
        super().__init__(
            y=y,
            y_idx=y_idx
        )

        ymed = np.median(self.y[self.y_idx])
        self.impurity = np.abs(self.y - ymed).mean()