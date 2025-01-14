from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def classify(features_train, labels_train, features_test, labels_test):
    """Train GaussianNB classifier and return model with accuracy"""

    # Create classifier
    cls = GaussianNB()

    # Fit the classifier on the training data
    cls.fit(features_train, labels_train)

    print(cls.predict(features_train))
    # Predict using the test data
    pred = cls.predict(features_test)

    # Calculate accuracy
    accuracy = accuracy_score(labels_test, pred)

    return cls, accuracy
