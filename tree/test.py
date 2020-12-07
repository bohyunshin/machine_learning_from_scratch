import numpy as np
import pandas as pd
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

from _criterion2 import gini
from example import BaseDecisionTree


def partition(X, col_name, var_type, value):
    feat = X[col_name]
    if var_type == 'conti':
        left_indices = np.where(feat <= value)[0]
        right_indices = np.where(feat > value)[0]
    else:
        pass
    return left_indices, right_indices

# loading the data set
dataset = load_iris(as_frame=True)
df= pd.DataFrame(data= dataset.data)

# adding the target and target names to dataframe
target_zip= dict(zip(set(dataset.target), dataset.target_names))
df["target"] = dataset.target
df["target_names"] = df["target"].map(target_zip)

# Seperating to X and Y
X = df.iloc[:, :4]
y = df.iloc[:, -1]

# splitting training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True, random_state=24)

model = BaseDecisionTree(
        criterion = 'gini',
        num_vars = 4,
        max_depth = -1,
        min_samples_split = None,
        min_samples_leaf = None)
# performing a split on the root node
# left_idx, right_idx = model.partition(X_train, "petal width (cm)", 'conti', 1.65)
# impurity = gini(y_train, y_train.index).impurity
# info_gain = model.information_gain(y_train, left_idx, right_idx, impurity)
# best_split = model.find_best_split(X_train, y_train, range(len(y_train)))

my_tree = model.build_tree(X_train, y_train, y_train.index, X.columns)
model.print_tree(my_tree)
example, example_target = X_test.iloc[6], y_test.iloc[6]
print(model.predict(example, my_tree))
