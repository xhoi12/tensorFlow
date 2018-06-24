# todo
# import dataSet
# partition into train and test
# train a classifier
# call prediction method
# calculate accurancy
######################################

# import datasets
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data  # features or input
y = iris.target  # labels or output

# after importing the dataSet I partition it below in train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# now I willl create a tree DecisionTreeClassifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# trains the classifier using the training data
my_classifier.fit(X_train, y_train)

# call the predict method and use it to classify the testing data
predictions = my_classifier.predict(X_test)
print(predictions)

# to calculate the accurancy I compare the predicted labels against the true labels
# to do so I import the method from scikit
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
