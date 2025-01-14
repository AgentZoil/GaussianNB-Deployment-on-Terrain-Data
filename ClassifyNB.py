def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier

    ### your code goes here!
    from sklearn.naive_bayes import GaussianNB
    cls = GaussianNB()
    cls.fit(features_train, labels_train)
    print(cls.predict(features_train))
    return cls
