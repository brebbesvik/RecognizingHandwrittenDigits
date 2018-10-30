import pandas as pd
from sklearn.model_selection import train_test_split
import timeit
from memory_profiler import profile

# Read the data from CSV files. Mye experience is using pandas for reading is faster
X = pd.read_csv('../handwritten_digits_images.csv', header=None).values
y = pd.read_csv('../handwritten_digits_labels.csv', header=None).values

# Split the data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=2)


@profile
def neural_network(x_train, x_test, y_train, y_test, heatmap=False):
    """
    Trains and validates a neural network.

    It trains the neural network using two convolutional layers to speed up the process when working with images.
    It also prints the time used to train the model, time spent to predict the test set and the accuracy of the model.
    Additional information information such as confusional matrix and classification report can be printed.

    Parameters:
        x_train:    The training set. The input data which should be classified.
        x_test:     The test set. Input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.
        y_test:     The test set which contains the class labels corresponding to the x_test set.
        heatmap=False:  Print a confusion matric and classification report, which gives further information about the performance of the produced model.
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.utils import np_utils

    # Reshape the data in X. Both training and test sets. Put the channel to the end
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Convert to float and represent the data from 0-1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # The output should be 10 neurons
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Set up the model with layers an neurons
    model = Sequential() # initialize
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.01))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Inspect the created model
    print(model.summary())

    # Compile model
    start_time_fit = timeit.default_timer()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model (X is features, Y is output, this is for training data)
    model.fit(x_train, y_train, batch_size=200, epochs=2, validation_data=(x_test, y_test), verbose=1)
    print("Time to fit the model", '{0:.2f}'.format(timeit.default_timer() - start_time_fit))

    # Validate model
    start_time_predict = timeit.default_timer()
    print("Accuracy: ", model.evaluate(x_test, y_test, batch_size=200))
    print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
    if heatmap:
        start_time_predict = timeit.default_timer()
        predictions = model.predict(x_test)
        print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
        heatmap_predictions(y_test.argmax(axis=1), predictions.argmax(axis=1))
        classification_report_predictions(y_test.argmax(axis=1), predictions.argmax(axis=1))


@profile
def k_nearest_neihbour(x_train, x_test, y_train, y_test, heatmap=False):
    """
    Trains and validates a model based on k nearest neighbour.

    It trains the model based on the k nearest neighbour classifier. It uses n_neighbour=1, which was the best value found by the cross validator.
    It also prints the time used to train the model, time spent to predict the test set and the accuracy of the model.
    Additional information information such as confusional matrix can be printed.

    Parameters:
        x_train:    The training set. The input data which should be classified.
        x_test:     The test set. Input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.
        y_test:     The test set which contains the class labels corresponding to the x_test set.
        heatmap=False:  Print a confusion matrix, which gives further information about the performance of the produced model.
    """
    print("K Nearest Neighbour")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    start_time_fit = timeit.default_timer()
    model = knn.fit(x_train, y_train.ravel())
    print("Time to fit the model", '{0:.2f}'.format(timeit.default_timer() - start_time_fit))
    start_time_predict = timeit.default_timer()
    print("Accuracy: ", '{0:.2f}'.format(model.score(x_test, y_test.ravel())))
    print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
    if heatmap:
        start_time_predict = timeit.default_timer()
        predictions = model.predict(x_test)
        print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
        heatmap_predictions(predictions, y_test)


def best_parameters_knn(x_train, y_train):
    """
    Finds the best n_neighbor parameter for K Nearest Neighbour.

    Finds the best n_neighbor parameter for K Nearest Neighbor using cross validation. N_neighbour is the number of the nearest neighbors.

    Parameters:
        x_train:    The training set. The input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.

    Returns:
        integer:   The best value for hyperparameter n_neighbour
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    best_mean_knn = [0, 0]
    for k in [1, 2, 3, 4, 5]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores_knn = cross_val_score(knn, x_train, y_train.ravel(), cv=6)
        if scores_knn.mean() > best_mean_knn[0]:
            best_mean_knn[0] = scores_knn.mean()
            best_mean_knn[1] = k
    return best_mean_knn[1]


@profile
def logistic_regression(x_train, x_test, y_train, y_test, c=1, heatmap=False):
    """
    Trains and validates a model based on logistic regression.

    It trains the model based on the logistic regression classifier. It uses C=60, which was the best value found by the cross validator.
    It also prints the time used to train the model, time spent to predict the test set and the accuracy of the model.
    Additional information information such as confusional matrix can be printed.

    Parameters:
        x_train:    The training set. The input data which should be classified.
        x_test:     The test set. Input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.
        y_test:     The test set which contains the class labels corresponding to the x_test set.
        heatmap=False:  Print a confusion matrix, which gives further information about the performance of the produced model.
    """
    from sklearn.linear_model import LogisticRegression
    linear = LogisticRegression(C=60)
    start_time_fit = timeit.default_timer()
    model = linear.fit(x_train, y_train.ravel())
    print("Time to fit the model", '{0:.2f}'.format(timeit.default_timer() - start_time_fit))
    start_time_predict = timeit.default_timer()
    print("Accuracy logistic: ", model.score(x_test, y_test.ravel()))
    print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
    if heatmap:
        start_time_predict = timeit.default_timer()
        predictions = model.predict(x_test)
        print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
        heatmap_predictions(predictions, y_test)


def best_parameters_logistic(x_train, y_train):
    """
    Finds the best C for logistic regression.

    Finds the best C parameter for Logistic Regression using cross validation.

    Parameters:
        x_train:    The training set. The input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.

    Returns:
        integer:   The best value for hyperparameter C
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    best_mean_log_reg = [0, 0]
    for c in [1, 30, 60, 90, 120]:
        logistic = LogisticRegression(C=c)
        scores_log_reg = cross_val_score(logistic, x_train, y_train.ravel(), cv=6)
        if scores_log_reg.mean() > best_mean_log_reg[0]:
            best_mean_log_reg[0] = scores_log_reg.mean()
            best_mean_log_reg[1] = c
    return best_mean_log_reg[1]


@profile
def decision_tree(x_train, x_test, y_train, y_test, heatmap=False):
    """
    Trains and validates a model based on decision tree.

    It trains the model based on the decision tree classifier. It uses max_depth=15, which was the best value found by the cross validator.
    It also prints the time used to train the model, time spent to predict the test set and the accuracy of the model.
    Additional information information such as confusional matrix can be printed.

    Parameters:
        x_train:    The training set. The input data which should be classified.
        x_test:     The test set. Input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.
        y_test:     The test set which contains the class labels corresponding to the x_test set.
        heatmap=False:  Print a confusion matrix, which gives further information about the performance of the produced model.
    """
    from sklearn import tree
    decision = tree.DecisionTreeClassifier(max_depth=15)
    start_time_fit = timeit.default_timer()
    model = decision.fit(x_train, y_train.ravel())
    print("Time to fit the model", '{0:.2f}'.format(timeit.default_timer() - start_time_fit))
    start_time_predict = timeit.default_timer()
    print("Accuracy decision: ", model.score(x_test, y_test.ravel()))
    print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
    if heatmap:
        start_time_predict = timeit.default_timer()
        predictions = model.predict(x_test)
        print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
        heatmap_predictions(predictions, y_test)


def best_parameters_decision(x_train, y_train):
    """
    Finds the best max_depth parameter for Decision Tree.

    Finds the best max_depth parameter for Decision Tree using cross validation. Max_depth is the max_depth of the tree (number of levels)

    Parameters:
        x_train:    The training set. The input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.

    Returns:
        integer:   The best value for hyperparameter max_depth
    """
    from sklearn import tree
    from sklearn.model_selection import cross_val_score
    best_mean_decision = [0, 0]
    for d in [5, 10, 15, 20, 25, 30, 35]:
        decision = tree.DecisionTreeClassifier(max_depth=d)
        scores_decision = cross_val_score(decision, x_train, y_train.ravel(), cv=6)
        if scores_decision.mean() > best_mean_decision[0]:
            best_mean_decision[0] = scores_decision.mean()
            best_mean_decision[1] = d
    return best_mean_decision[1]


@profile
def random_forest(x_train, x_test, y_train, y_test, heatmap=False):
    """
    Trains and validates a random forest.

    It trains the model based on the random forest. It uses n_estimator=150, which was the best value found by the cross validator.
    n_estimator is the number of trees in the forest.
    It also prints the time used to train the model, time spent to predict the test set and the accuracy of the model.
    Additional information information such as confusional matrix and classification report can be printed.

    Parameters:
        x_train:    The training set. The input data which should be classified.
        x_test:     The test set. Input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.
        y_test:     The test set which contains the class labels corresponding to the x_test set.
        heatmap=False:  Print a confusion matric and classification report, which gives further information about the performance of the produced model.
    """
    from sklearn.ensemble import RandomForestClassifier
    start_time_fit = timeit.default_timer()
    forest = RandomForestClassifier(n_estimators=150)
    model = forest.fit(x_train, y_train.ravel())
    print("Time to fit the model", '{0:.2f}'.format(timeit.default_timer() - start_time_fit))
    start_time_predict = timeit.default_timer()
    print("Accuracy forest: ", model.score(x_test, y_test.ravel()))
    print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
    if heatmap:
        start_time_predict = timeit.default_timer()
        predictions = model.predict(x_test)
        print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
        heatmap_predictions(predictions, y_test)
        classification_report_predictions(predictions, y_test)


def best_parameters_forest(x_train, y_train):
    """
    Finds the best _estimators for Random Forest.

    Finds the best n_estimator parameter for Random Forest using cross validation. n_estimator is the number of trees in the forest.

    Parameters:
        x_train:    The training set. The input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.

    Returns:
        integer:   The best value for hyperparameter n_estimator
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    best_mean_forest = [0, 0]
    for n in [10, 25, 50, 75, 100, 125, 150]:
        forest = RandomForestClassifier(n_estimators=n)
        scores_forest = cross_val_score(forest, x_train, y_train.ravel(), cv=6)
        if scores_forest.mean() > best_mean_forest[0]:
            best_mean_forest[0] = scores_forest.mean()
            best_mean_forest[1] = n
    return best_mean_forest[1]


@profile
def bayesian_ridge_regression(x_train, x_test, y_train, y_test, heatmap=False):
    """
    Trains and validates a model based on Bayesian Multinominal

    It trains the model based on the Bayesian Multinominal classifier. It uses alpha=0, which was the best value found by the cross validator.
    It also prints the time used to train the model, time spent to predict the test set and the accuracy of the model.
    Additional information information such as confusional matrix can be printed

    Parameters:
        x_train:    The training set. The input data which should be classified.
        x_test:     The test set. Input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.
        y_test:     The test set which contains the class labels corresponding to the x_test set.
        heatmap=False:  Print a confusion matrix, which gives further information about the performance of the produced model.
    """
    from sklearn.naive_bayes import MultinomialNB
    start_time_fit = timeit.default_timer()
    bayesian_ridge = MultinomialNB(alpha=0)
    model = bayesian_ridge.fit(x_train, y_train)
    print("Time to fit the model", '{0:.2f}'.format(timeit.default_timer() - start_time_fit))
    start_time_predict = timeit.default_timer()
    print("Accuracy Bayesian ridge: ", model.score(x_test, y_test.ravel()))
    print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
    if heatmap:
        start_time_predict = timeit.default_timer()
        predictions = model.predict(x_test)
        print("Time to predict the data", '{0:.2f}'.format(timeit.default_timer() - start_time_predict))
        heatmap_predictions(predictions, y_test)


def best_parameters_bayesian(x_train, y_train):
    """
    Finds the best alpha for Bayesian Multinominal.

    Finds the best alpha parameter for Bayesian Multinominal using cross validation.

    Parameters:
        x_train:    The training set. The input data which should be classified.
        y_train:    The training set which contains the class labels corresponding to the x_train set.

    Returns:
        integer:   The best value for hyperparameter alpha
    """
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import cross_val_score
    best_mean_bayesian = [0, 0]
    for a in [0, 1, 50, 200, 600]:
        bayesian = MultinomialNB(alpha=a)
        scores_bayesian = cross_val_score(bayesian, x_train, y_train.ravel(), cv=6)
        if scores_bayesian.mean() > best_mean_bayesian[0]:
            best_mean_bayesian[0] = scores_bayesian.mean()
            best_mean_bayesian[1] = a
    return best_mean_bayesian[1]


def heatmap_predictions(predictions, y_test):
    """
    Prints a confusion matrix

    Prints a confusion matrix based on the predictions and an answer key.

    Parameters:
        predictions:    The predicted class labels
        y_test:    The answer key of the class labels
    """
    from sklearn import metrics
    cm = metrics.confusion_matrix(y_test, predictions)
    print(cm)


def classification_report_predictions(predictions, y_test):
    """
    Prints a classification report

    Prints a classification report based on the predictions and an answer key.

    Parameters:
        predictions:    The predicted class labels
        y_test:    The answer key of the class labels
    """
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, predictions)
    print(cr)


# Please comment out the functions you don't want to run. Each of them is pretty time consuming

# Classifier
random_forest(X_train, X_test, Y_train, Y_test, heatmap=True)
bayesian_ridge_regression(X_train, X_test, Y_train, Y_test)
decision_tree(X_train, X_test, Y_train, Y_test, heatmap=True)
logistic_regression(X_train[:1000], X_test, Y_train[:1000], Y_test)
k_nearest_neihbour(X_train[:1000], X_test, Y_train[:1000], Y_test)
neural_network(X_train, X_test, Y_train, Y_test, heatmap=True)

# Cross validation
print(best_parameters_knn(X_train[:1000], Y_train[:1000]))
print("Best logistic parameter", best_parameters_logistic(X_train[:1000], Y_train[:1000]))
print("Best bayesian parameter", best_parameters_bayesian(X_train, Y_train))
print("Best decision tree parameter", best_parameters_decision(X_train, Y_train))
print("Best forest parameter", best_parameters_forest(X_train, Y_train))

