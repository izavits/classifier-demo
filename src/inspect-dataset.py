# Load all the necessary libraries

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the dataset
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Inspect the Dataset
print('Number of instances and attributes: {}'.format(dataset.shape))
print('')
print('Sample RAW data:')
print(dataset.head(20))
print('')
print('Statistical Summary of the Dataset:')
print(dataset.describe())
print('')
print('Class distribution:')
print(dataset.groupby('class').size())

# Plot some data to get a better understanding of the
# Univariate plots to get a better idea of the distribution of input variables
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
f1=plt.figure(1)

# Histogram of each variable
dataset.hist()
f2=plt.figure(2)

# Multivariate plots to see the interaction between the variables
scatter_matrix(dataset)
f3=plt.figure(3)
plt.show()
