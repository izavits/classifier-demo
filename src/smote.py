# A Decision Tree classification algorithm evaluated
# on imbalanced (synthetic) dataset with SMOTE oversampling

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Make a synthetic dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# Define a Pipeline:
# The correct application of oversampling during k-fold cross-validation
# is to apply the method to the training dataset only, then evaluate the model
# on the stratified but non-transformed test set.
# This can be achieved by defining a Pipeline that first transforms the
# training dataset with SMOTE then fits the model.
steps = [('over', SMOTE()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)

# Evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))
