import numpy as np
import numbers
from ._criterion import ClassificationCriterion, RegressionCriterion

CRITERIA_CLF = {"gini": ClassificationCriterion.gini, "entropy": ClassificationCriterion.entropy}
CRITERIA_REG = {"mse": RegressionCriterion.mse, "mae": RegressionCriterion.mae}

class BaseDecisionTree():

    """
    Base class for CART.
    Warning: This (parent) class should not be used directly
    Use child class instead
    """

    def __init__(self, *,
                 criterion,
                 splitter,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_fraction_leaf,
                 max_features,
                 max_leaf_nodes,
                 random_state,
                 min_impurity_decrease,
                 min_impurity_split,
                 class_weight=None,
                 presort='deprecated',
                 ccp_alpha=0.0,
                 cat_feat_indiecs,
                 conti_feat_indices):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort
        self.ccp_alpha = ccp_alpha
        self.cat_feat_indices = cat_feat_indiecs
        self.conti_feat_indices = conti_feat_indices

    def split_classification(self, X, y, var_type, feat_indx):
        n_samples, self.n_features_ = X.shape
        feat = X[:, feat_indx]

        # when the input variable is categorical
        if var_type == 'cat':
            classes = np.unique(feat)
            cl2impurity = {}
            for cl in classes:
                left_indices = np.where(feat == cl)[0]
                right_indices = np.setdiff1d(np.array(range(n_samples)), left_indices)
                criteria = CRITERIA_CLF[self.criterion](left_indices, right_indices, y)
                cl2impurity[cl] = criteria

            min_impurity_class = min(cl2impurity, key=cl2impurity.get)
            min_impurity = min(cl2impurity.values())
            left_indices = np.where(feat == cl)[0]
            right_indices = np.setdiff1d(np.array(range(n_samples)), left_indices)
            return left_indices, right_indices, min_impurity_class, min_impurity

        # when the input variable is continuous
        else:
            # for now just use the mean threshold to split continuous features
            m = np.mean(feat)
            left_indices = np.where(feat <= m)[0]
            right_indices = np.where(feat > m)[0]

            impurity = CRITERIA_CLF[self.criterion](left_indices, right_indices, y)
            return left_indices, right_indices, m, impurity


    def fit(self, X, y, is_classification):

        n_samples, self.n_features_ = X.shape

        if is_classification:
            criteria = CRITERIA_CLF[self.criterion]
        else:
            criteria = CRITERIA_REG[self.criterion]

        max_depth = self.max_depth
        max_leaf_nodes = self.max_leaf_nodes
        min_samples_leaf = self.min_samples_leaf
        min_samples_split = self.min_samples_split
        cat_feat_indices = self.cat_feat_indices
        conti_feat_indices = self.conti_feat_indices

        if isinstance(self.max_features, str):
            if self.max_features == 'auto':
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == 'sqrt':
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == 'log2':
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features is None:
                max_features = self.n_features_
            elif isinstance(self.max_features, numbers.Integral):
                max_features = self.max_features

        if is_classification:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

        # Not yet on weight

        # Build tree
        # for classification tree
        if is_classification:
            cat_feat2impurity = {}
            cat_feat2class = {}
            conti_feat2impurity = {}
            conti_feat2threshold = {}
            for c in cat_feat_indices:
                """
                categorical features loop.
                INPUT: Categorical features
                OUTPUT: Categorical features
                """
                left_indices, right_indices, min_impurity_class, min_impurity = self.split_classification(X, y, 'cat', c)

                cat_feat2impurity[c] = min_impurity
                cat_feat2class[c] = min_impurity_class

            for c in conti_feat_indices: # continuous features loop
                """
                continuous features loop.
                INPUT: Continuous features
                OUTPUT: Categorical features
                """
                left_indices, right_indices, m, min_impurity = self.split_classification(X, y, 'conti', c)

                conti_feat2impurity[c] = min_impurity
                conti_feat2threshold[c] = m

            min_impurity_cat = min(cat_feat2impurity, key=cat_feat2impurity.get)
            min_impurity_conti = min(conti_feat2impurity, key=conti_feat2impurity.get)

            # compare min impurity of categorical variables and continuous variables
            if min(cat_feat2impurity.values()) < min(conti_feat2impurity):
                chosen_variable = min_impurity_cat
            else:
                chosen_variable = min_impurity_conti





