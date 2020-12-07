import numpy as np
from collections import Counter
from math import log2

class BaseCriterion():

    def __init__(self, left_node_index, right_node_index, y):
        self.left_node_index = left_node_index
        self.right_node_index = right_node_index
        self.y = y
        self.classes = np.unique(y)
        self.n = len(self.y)

class ClassificationCriterion(BaseCriterion):

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

    def gini(self):
        """
        Evaluate the impurity of the current node, i.e.,
        the impurity of samples[:] using the Gini criterion
        """

        left_node_y = self.y[self.left_node_index]
        right_node_y = self.y[self.right_node_index]

        left_counter = Counter(left_node_y)
        right_counter = Counter(right_node_y)

        left_gini = 0
        right_gini = 0

        for c in self.classes:
            left_counter[c] /= self.n
            right_counter[c] /= self.n

            left_gini += left_counter[c]**2
            right_gini += right_counter[c]**2

        left_impurity = 1 - left_gini
        right_impurity = 1 - right_gini

        return (len(left_node_y) * left_impurity + len(right_node_y) * right_impurity) / (len(left_node_y) + len(right_node_y))

    def entropy(self):

        """
        Evaluate the impurity of the current node, i.e.,
        the impurity of samples[:] using the entropy criterion
        """

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

        return (len(left_node_y) * left_impurity + len(right_node_y) * right_impurity) / (
                    len(left_node_y) + len(right_node_y))

class RegressionCriterion(BaseCriterion):

    def mse(self):

        """
        Evaluate the impurity of the current node, i.e.,
        the impurity of samples[:] using the mse criterion
        """

        left_node_y = self.y[self.left_node_index]
        right_node_y = self.y[self.right_node_index]

        left_node_avg = np.mean(left_node_y)
        right_node_avg = np.mean(right_node_y)

        left_impurity = np.power(left_node_y - left_node_avg, 2).sum().mean()
        right_impurity = np.power(right_node_y - right_node_avg, 2).sum().mean()

        return (len(left_node_y) * left_impurity + len(right_node_y) * right_impurity) / (
                len(left_node_y) + len(right_node_y))

    def mae(self):
        """
        Evaluate the impurity of the current node, i.e.,
        the impurity of samples[:] using the mae criterion
        """

        left_node_y = self.y[self.left_node_index]
        right_node_y = self.y[self.right_node_index]

        left_node_med = np.median(left_node_y)
        right_node_med = np.median(right_node_y)

        left_impurity = np.abs(left_node_y - left_node_med).sum().mean()
        right_impurity = np.abs(right_node_y - right_node_med).sum().mean()

        return (len(left_node_y) * left_impurity + len(right_node_y) * right_impurity) / (
                len(left_node_y) + len(right_node_y))