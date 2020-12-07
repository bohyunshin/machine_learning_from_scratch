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
                 left_node_index,
                 right_node_index,
                 y
                 ):
        super().__init__(
            left_node_index = left_node_index,
            right_node_index = right_node_index,
            y = y
        )

        left_node_y = self.y[self.left_node_index]
        right_node_y = self.y[self.right_node_index]

        left_counter = Counter(left_node_y)
        right_counter = Counter(right_node_y)

        left_ent = 0
        right_ent = 0

        for c in self.classes:
            left_counter[c] /= self.n
            right_counter[c] /= self.n

            left_ent += -left_counter[c] * log2(left_counter[c])
            right_ent += -right_counter[c] * log2(right_counter[c])

        left_impurity = 1-left_ent
        right_impurity = 1-right_ent

        self.impurity = (len(left_node_y) * left_impurity + len(right_node_y) * right_impurity) / (
                    len(left_node_y) + len(right_node_y))

class mse(BaseCriterion):
    """
    Evaluate the impurity of the current node, i.e.,
    the impurity of samples[:] using the mse criterion
    """

    def __init__(self,
                 left_node_index,
                 right_node_index,
                 y
                 ):
        super().__init__(
            left_node_index=left_node_index,
            right_node_index=right_node_index,
            y=y
        )



        left_node_y = self.y[self.left_node_index]
        right_node_y = self.y[self.right_node_index]

        left_node_avg = np.mean(left_node_y)
        right_node_avg = np.mean(right_node_y)

        left_impurity = np.power(left_node_y - left_node_avg, 2).sum().mean()
        right_impurity = np.power(right_node_y - right_node_avg, 2).sum().mean()

        self.impurity = (len(left_node_y) * left_impurity + len(right_node_y) * right_impurity) / (
                len(left_node_y) + len(right_node_y))

class mae(BaseCriterion):
    """
    Evaluate the impurity of the current node, i.e.,
    the impurity of samples[:] using the mae criterion
    """

    def __init__(self,
                 left_node_index,
                 right_node_index,
                 y
                 ):
        super().__init__(
            left_node_index=left_node_index,
            right_node_index=right_node_index,
            y=y
        )

        left_node_y = self.y[self.left_node_index]
        right_node_y = self.y[self.right_node_index]

        left_node_med = np.median(left_node_y)
        right_node_med = np.median(right_node_y)

        left_impurity = np.abs(left_node_y - left_node_med).sum().mean()
        right_impurity = np.abs(right_node_y - right_node_med).sum().mean()

        self.impurity = (len(left_node_y) * left_impurity + len(right_node_y) * right_impurity) / (
                len(left_node_y) + len(right_node_y))