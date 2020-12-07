import numpy as np


from _criterion2 import gini, entropy

CRITERIA_CLF = {"gini": gini, "entropy": entropy}
# CRITERIA_REG = {"mse": RegressionCriterion.mse, "mae": RegressionCriterion.mae}

def _convert_numparray(arr):
    if isinstance(arr, np.ndarray):
        return arr
    else:
        return np.array(arr)


# helper function to count values
def count(label, idx):

    """
    Function that counts the unique values

    Params
    ------
    label: target labels
    idx: index of rows

    Returns
    -------
    dict_label_count: Dictionary of label and counts
    """
    unique_label, unique_label_counts = np.unique(label[idx], return_counts=True)
    dict_label_count = dict(zip(unique_label, unique_label_counts))
    return dict_label_count

class Leaf:
    """
    A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, label, idx):
        self.predictions = count(label, idx)


# https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
class Decision_Node:
    """
    A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 column,
                 value,
                 true_branch,
                 false_branch):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

class BaseDecisionTree:
    """
    This is base decision tree class used for classificatio tree, regression tree.
    Note that this class is not used directly for CART.
    Instead, this class plays a role as parent class, embodied to children class.
    See sklearn implementation for more details

    Finished Implementation
    -----------------------
    max_depth
    min_samples_leaf
    min_samples_split
    feature_importances

    Not-Yet Implementation
    ----------------------
    min_weight_fraction_leaf
    max_features
    max_leaf_nodes
    min_impurity_decrease
    ccp_alpha
    """
    def __init__(self,
                 criterion,
                 num_vars,
                 max_depth=-1,
                 min_samples_split=None,
                 min_samples_leaf=None):
        self.criterion = criterion
        self.num_vars = num_vars
        self.feature_importances_ = np.zeros(self.num_vars)
        self.current_depth = 0
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.best_col_list = []


    # def __init__(self, *,
    #              criterion,
    #              splitter,
    #              max_depth,
    #              min_samples_split,
    #              min_samples_leaf,
    #              min_weight_fraction_leaf,
    #              max_features,
    #              max_leaf_nodes,
    #              random_state,
    #              min_impurity_decrease,
    #              min_impurity_split,
    #              class_weight=None,
    #              presort='deprecated',
    #              ccp_alpha=0.0,
    #              cat_feat_indiecs,
    #              conti_feat_indices):
    #     self.criterion = criterion
    #     self.splitter = splitter
    #     self.max_depth = max_depth
    #     self.min_samples_split = min_samples_split
    #     self.min_samples_leaf = min_samples_leaf
    #     self.min_weight_fraction_leaf = min_weight_fraction_leaf
    #     self.max_features = max_features
    #     self.max_leaf_nodes = max_leaf_nodes
    #     self.random_state = random_state
    #     self.min_impurity_decrease = min_impurity_decrease
    #     self.min_impurity_split = min_impurity_split
    #     self.class_weight = class_weight
    #     self.presort = presort
    #     self.ccp_alpha = ccp_alpha
    #     self.cat_feat_indices = cat_feat_indiecs
    #     self.conti_feat_indices = conti_feat_indices

    def partition(self, X, col_name, var_type, value):
        """
        Partitions on the current node

        Params
        ------
        X: design matrix
        col_name: selected column to split in current node
        var_type: ['conti', 'cat'], whether the selected column is continuous or not
        value: selected value.
            For continuous value, this is numeric.
            For categorical value, this is one of sub category

        Returns
        -------
        left_idx: index of left node
        right_idx: index of right node
        Here, we calculate index based on original dataset, X (or y)
        """

        # for continuous variable
        if var_type == 'conti':
            left_indx = X[lambda x: x[col_name] <= value].index
            right_indx = X[lambda x: x[col_name] > value].index

        # for categorical variable
        else:
            left_indx = X[lambda x: x[col_name] == value].index
            right_indx = X[lambda x: x['col_name'] != value].index
        return left_indx, right_indx

    def information_gain(self, y, left_idx, right_idx, impurity):
        """
        Get how much information has been gained by splitting nodes.
        We use IG when determining which feature is used on splitting node.
        That is, this function calculates
        H(T) - H(T|a)
        where H(T) is entropy(or gini, mse, mae) of prior node and
        H(T|a) is entropy of current node

        Params
        ------
        y: response variable
        left_idx: index of left child
        right_idx: index of right child
        impurity: impurity of prior node
        """
        p = len(left_idx) / (len(left_idx) + len(right_idx))
        left_gini = gini(y, left_idx).impurity
        right_gini = gini(y, right_idx).impurity
        info_gain = impurity - p*left_gini - (1-p)*right_gini
        return info_gain

    def find_best_split(self, X, y, y_idx, X_conti_cols):
        """
        Find best split by circulating each feature.
        There are 4 scenarios.
        1) continuous input -> continuous output
        2) continuous input -> categorical output
        3) categorical input -> continuous output
        4) categorical input -> categorical output

        Currently, when 1), 2), we just loop through all the unique input value,
        and select one that produces minimal impurity (or maximal information gain).
        However, this method is not efficient at all when the size of data is growing larger.
        So, we are considering implementing other algorithm to find optimal threshold for continuous variable.
        When 3), 4), we loop through all sub categories of categorical input value.
        Then, we calculate IG for each sub categories and select one that maximizes IG.
        This approach is classical for categorical input variable.

        Params
        ------
        X: design matrix
        y: response variable
        y_idx: index of elements which are in the current child node
        X_conti_cols: list of continuous columns

        Returns
        -------
        best_gain: besg IG when splitting by best column
        best_col: best column for splitting
        best_value: best threshold for continuous (or categorical) variable
        """

        best_gain = 0
        best_col = None
        best_value = None

        # if current depth is greater than or equal to max depth
        # stop splitting trees
        if self.max_depth == -1 or self.max_depth > self.current_depth:
            pass
        else:
            return best_gain, best_col, best_value

        X = X.loc[y_idx]
        y_idx = y.loc[y_idx].index

        impurity = gini(y, y_idx).impurity

        for col in X.columns:

            # for now, all the unique continuous values are investigated
            if col in X_conti_cols:
                unique_values = set(X[col])
                for value in unique_values:
                    left_idx, right_idx = self.partition(X, col, 'conti', value)

                    if len(left_idx) == 0 or len(right_idx) == 0:
                        continue

                    # if the resulting leaves from split are lower then min_samples_leaf,
                    # stop splitting trees
                    if self.min_samples_leaf is None:
                        pass
                    elif len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                        continue

                    info_gain = self.information_gain(y, left_idx, right_idx, impurity)

                    if info_gain > best_gain:
                        best_gain, best_col, best_value = info_gain, col, value

            else:
                unique_values = set(X[col])
                for value in unique_values:
                    left_idx, right_idx = self.partition(X, col, 'cat', value)

                    if len(left_idx) == 0 or len(right_idx) == 0:
                        continue

                    # if the resulting leaves from split are lower then min_samples_leaf,
                    # stop splitting trees
                    if self.min_samples_leaf is None:
                        pass
                    elif len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                        continue

                    info_gain = self.information_gain(y, left_idx, right_idx, impurity)

                    if info_gain > best_gain:
                        best_gain, best_col, best_value = info_gain, col, value

        # getting feature importances
        best_col_indx = np.where(X.columns == best_col)[0]
        self.feature_importances_[best_col_indx] += best_gain

        # add best_col
        if best_col is None:
            pass
        else:
            self.best_col_list.append(best_col)
            self.current_depth += 1

        return best_gain, best_col, best_value


    def build_tree(self, X, y, y_idx, X_conti_cols):
        """
        Build tree recursively

        Params
        ------
        X: design matrix
        y: response variable
        y_idx: index of elements which are in the current child node
        X_conti_cols: list of continuous columns

        Returns
        -------
        Decision_Node object: used when we printing tree results
        """

        best_gain, best_col, best_value = self.find_best_split(X,y,y_idx, X_conti_cols)

        if best_gain == 0:
            return Leaf(y, y_idx)

        # split current node with best column and value
        if best_col in X_conti_cols:
            left_idx, right_idx = self.partition(X.loc[y_idx], best_col, 'conti', best_value)
        else:
            left_idx, right_idx = self.partition(X.loc[y_idx], best_col, 'cat', best_value)
        true_branch = self.build_tree(X, y, left_idx, X_conti_cols)
        false_branch = self.build_tree(X, y, right_idx, X_conti_cols)

        return Decision_Node(best_col, best_value, true_branch, false_branch)

    # https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
    def print_tree(self, node, spacing=""):
        """
        Printing trees
        """

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print(spacing + "Predict", node.predictions)
            return

        # Print the col and value at this node
        print(spacing + f"[{node.column} <= {node.value}]")

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")

    def fit(self, X, y, y_idx, X_conti_cols, print_result=False):
        self.mytree = self.build_tree(X, y, y_idx, X_conti_cols)

        if print_result:
            self.print_tree(self.mytree)

    def predict(self, test_data, tree):

        """
        Classify unseen examples

        Params
        ------
        test_data: Unseen observation
        tree: tree that has been trained on training data

        Returns
        -------
        The prediction of the observation.
        """

        # Check if we are at a leaf node
        if isinstance(tree, Leaf):
            return max(tree.predictions)

        # the current feature_name and value
        feature_name, feature_value = tree.column, tree.value

        # pass the observation through the nodes recursively
        if test_data[feature_name] <= feature_value:
            return self.predict(test_data, tree.true_branch)

        else:
            return self.predict(test_data, tree.false_branch)

class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, criterion, num_vars):

        super().__init__(
            criterion = criterion,
            num_vars = num_vars
        )

    def fit(self, X, y, y_idx, X_conti_cols):

        super().fit(X, y, y_idx, X_conti_cols, True)



