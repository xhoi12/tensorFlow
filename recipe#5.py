# import random
# I'm using the k-nearest neighbor algorithm (k-NN) for the classifier
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)
# code from recipe#4 I build a custom classifier instead of importing it
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
        #   label = random.choice(self.y_train) replaced with below
            label = self.closest(row) # finds the closest training point to the test point
            predictions.append(label)
        return predictions
    def closest(self, row): # implementing the k-NN distance
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

# import datasets
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data  # features or input
y = iris.target  # labels or output

# after importing the dataSet I partition it below in train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# # now I willl create a tree DecisionTreeClassifier
# from sklearn import tree
my_classifier = ScrappyKNN()

# trains the classifier using the training data
my_classifier.fit(X_train, y_train) # fit does the training

# call the predict method and use it to classify the testing data
predictions = my_classifier.predict(X_test)
print(predictions)

# to calculate the accurancy I compare the predicted labels against the true labels
# to do so I import the method from scikit
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
