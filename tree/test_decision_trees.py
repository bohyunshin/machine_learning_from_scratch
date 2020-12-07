import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from _criterion import gini
from _classes import BaseDecisionTree, DecisionTreeClassifier_

# My Implementation
df = pd.read_csv('../dataset/iris.csv')
X = df.iloc[:, :4]
y = df.iloc[:, -1]

# splitting training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True, random_state=24)

model = DecisionTreeClassifier_(
        criterion = 'gini',
        num_vars = 4,
        max_depth = -1,
        min_samples_split = None,
        min_samples_leaf = None)
my_tree = model.build_tree(X_train, y_train, y_train.index, X.columns)
model.print_tree(my_tree)
# create a new col of predictions
X_test["predictions"] = X_test.apply(model.predict, axis=1, args=(my_tree,X.columns))
print(f"My Implementation:\nACCURACY: {accuracy_score(y_test, X_test['predictions'])}")
print(model.feature_importances_)

# sklearn Implementation
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
sklearn_y_preds = dt.predict(X_test.iloc[:,:-1])
print(f"Sklearn Implementation:\nACCURACY: {accuracy_score(y_test, sklearn_y_preds)}")
print(dt.feature_importances_)
