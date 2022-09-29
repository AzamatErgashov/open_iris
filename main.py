from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)


import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("iris.csv")
df = dataset.iloc[:,0:4]
df.hist()
plt.show()


dataset = pd.read_csv("iris.csv")
df = dataset.iloc[:,0:4]

scatter_matrix(df, color="gold")
plt.show()




dataset = pd.read_csv("iris.csv")
df = dataset.iloc[:,0:4]

array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# DecisionTree
model1 = DecisionTreeClassifier()

# GaussianNB
model2 = GaussianNB()

# KNeighbors
model3 = KNeighborsClassifier()

# LogisticRegression
model4 = LogisticRegression(solver='liblinear', multi_class='ovr')

# LinearDiscriminant
model5 = LinearDiscriminantAnalysis()

# SVM
model6 = SVC(gamma='auto')
kf = KFold(n_splits=2)

cv_results1 = cross_val_score(model1, X_train, Y_train, cv=kf.get_n_splits(X), scoring='accuracy')
cv_results2 = cross_val_score(model2, X_train, Y_train, cv=kf.get_n_splits(X), scoring='accuracy')
cv_results3 = cross_val_score(model3, X_train, Y_train, cv=kf.get_n_splits(X), scoring='accuracy')
cv_results4 = cross_val_score(model4, X_train, Y_train, cv=kf.get_n_splits(X), scoring='accuracy')
cv_results5 = cross_val_score(model5, X_train, Y_train, cv=kf.get_n_splits(X), scoring='accuracy')
cv_results6 = cross_val_score(model6, X_train, Y_train, cv=kf.get_n_splits(X), scoring='accuracy')


print(f"DecisionTree: {cv_results1[0]} ({cv_results1[1]})")
print(f"GaussianNB: {cv_results2[0]} ({cv_results2[1]})")
print(f"KNeighbors: {cv_results3[0]} ({cv_results3[1]})")
print(f"LogisticRegression: {cv_results4[0]} ({cv_results4[1]})")
print(f"LinearDiscriminant: {cv_results5[0]} ({cv_results5[1]})")
print(f"SVM: {cv_results6[0]} ({cv_results6[1]})")
