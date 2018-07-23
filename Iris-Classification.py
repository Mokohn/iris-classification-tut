## Iris Classification ##
## A step by step solution found here: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/ ## 

# import libraries
import pandas as pd
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

# Load dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = '/Users/Momolino/Dropbox/Machine Learning & Deep Learning/Projects/Iris Classification/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Shape of the data 
# Comment: 150 lines, 5 columns
ln, col = dataset.shape
print('Columns: ', col, 'Lines: ', ln)

# Looking at first 5 rows of data
print(dataset.head(10))

# Statistical descriptions of the dataset
print(dataset.describe())

# Class distribution
# Commentary: Each class has the same distribution (50)
print(dataset.groupby('class').size())


# Box and Whisker plots (given, that variables are numeric)
# Comment: 
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# Histogram for visualizing the distribution
# Comment: sepal-length (Petal length?) and sepal-width have a gaussian distribution
dataset.hist()
#plt.show()

# Scatter Plot Matrix for spotting relationships between input variables
# Comment: 
scatter_matrix(dataset)
#plt.show()


## Now the following steps will be taken ## 

    # Separate out a validation dataset.
    # Set-up the test harness to use 10-fold cross validation.
    # Build 5 different models to predict species from flower measurements
    # Select the best model.

# Creating a validation set 
array = dataset.values
X = array[:, 0:4] # all values of dataset
Y = array[:, 4] # Target variable
val_size = 0.2 # Size of the validation set
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=val_size, random_state=seed)

# 10-fold cross validation is used to split the original set into 10 parts, 9 for training and 1 for validation
# each subsample k will be used as validation set until all subsets are validated
# all k results from the folds will be averaged

# Test options and evaluation metric
seed = 7
scoring = 'accuracy' # accuracy is a metric and a ratio of the number of correctly predicted instances in divided
					 # by the total number of instances in the dataset multiplied by 100 (to give a percentage)

### Comment ###
# Since some of the classes are partially linearly separable in some dimensions we are expecting good results
# Now the following algorithmns are evaluated


    # Logistic Regression (LR)
    # Linear Discriminant Analysis (LDA)
    # K-Nearest Neighbors (KNN).
    # Classification and Regression Trees (CART).
    # Gaussian Naive Bayes (NB).
    # Support Vector Machines (SVM).

### End Comment ###

# Models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluating 
# Comment: SVM and KNN have the highest accuracy (nearly 100%)
results = []
names = []
for name, model in models: 
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Visualizing results 
fig = plt.figure()
fig.suptitle('Algorithmn Comparison')
ax = fig.add_subplot(111) # 111 are subplot grid parameters --> 1x1 grid, first subplot
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

## Now we will make some predictions on the validation dataset ##
# We are using the KNN model, because it has the highest accuracy (according to the crossfold)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions)) # A conf. matrix (summarizes the performance of a classification algorithm) shows, besides the classification accuracy, what the classification model gets right and what errors it makes
print(classification_report(Y_validation, predictions))

# More on confusion matrizes: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

