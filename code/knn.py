import sys

import numpy as np
import scipy.spatial.distance as sp
import scipy.stats as st

from code import data


# computing the matrix of pairwise distances between all vector elements
# (rows) of "X" and "Y" matrices. scipy.spacial.distance.cdist is used to 
# calculate the distances efficiently. "metric" argument may contain ":" 
# separating metric name from additional parameter (e.g. "minkowski:3")
def get_distances(X, Y, metric):
    if ":" in metric:
        m, n = metric.split(":")
        return sp.cdist(X, Y, m, int(n))
    else:
        return sp.cdist(X, Y, metric)

# classifying the set of inputs "X_inputs" based on their "k"-nearest neighbors
# from "X_data" and their respective labels from "T_data". The value of "metric"
# argument is used to specify the actual metric name in scipy terms
def classify(X_inputs, X_data, T_data, k, metric):
    distances = get_distances(X_inputs, X_data, metric)
    nearest_neighbors = np.argpartition(distances, k-1, axis=1)[:, :k]
    nearest_labels = T_data[nearest_neighbors]
    return st.mode(nearest_labels, axis=1)[0]

# comparing predicted vs. true labels and
# returning the corresponding error score
def get_error_score(T_predicted, T_true):
    count = np.sum(T_predicted.ravel() != T_true.ravel())
    return count * 100.0 / len(T_predicted)


# k-fold cross-validation of k-NN model based on "k" and "metric".
def cross_validate(X, T, folds, k, metric):
    error_scores = []
    fold_len = len(X) / folds

    # for each fold:
    for fold in range(folds):
        # determining start and end
        # positions for the fold
        f_start = fold * fold_len
        f_end = f_start + fold_len

        # splitting the data into the validation set (the fold)
        # and the correspondong training set (the rest of data)
        X_training = np.concatenate((X[:f_start], X[f_end:]))
        T_training = np.concatenate((T[:f_start], T[f_end:]))
        X_validation = X[f_start:f_end]
        T_validation = T[f_start:f_end]

        # classifying the validation data and storing the resulting error score
        T_predicted = classify(X_validation, X_training, T_training, k, metric)
        error_scores.append(get_error_score(T_predicted, T_validation))

    # outputting the array of the error 
    # scores for each of the folds
    return np.array(error_scores)



if __name__ == "__main__":

    # command-line arguments (and their default values)

    ntrain = 60000      # number of training samples used
    ntest = 10000       # number of testing samples used

    deskew = True       # deskew input images or not (by default: yes)
    normalize = False   # normalize input vectors or not (by default: no)
    evaluate = False    # evaluate on testing data or not (by default: no)

    folds = 6           # number of folds for cross-validation

    metrics = ["cityblock", "sqeuclidean", "cosine"]    # different metrics tried (in scipy terms)
    ks = [1, 3, 5, 7, 9, 17, 33]                        # different k-values tried


    # processing command-line arguments

    while len(sys.argv) > 1:
        option = sys.argv[1]; del sys.argv[1]

        if option == "-ntrain":
            ntrain = int(sys.argv[1]); del sys.argv[1]
        elif option == "-ntest":
            ntest = int(sys.argv[1]); del sys.argv[1]
        elif option == "-deskew":
            deskew = int(sys.argv[1]); del sys.argv[1]
        elif option == "-normalize":
            normalize = int(sys.argv[1]); del sys.argv[1]
        elif option == "-evaluate":
            evaluate = int(sys.argv[1]); del sys.argv[1]
        elif option == "-metrics":
            metrics = sys.argv[1].split(","); del sys.argv[1]
        elif option == "-ks":
            ks = [int(k) for k in sys.argv[1].split(",")]; del sys.argv[1]
        elif option == "-folds":
            folds = int(sys.argv[1]); del sys.argv[1]
        else:
            print sys.argv[0], ": invalid option", option
            sys.exit(1)


    print "K-Nearest Neighbors"
    print

    print "Reading data..."
    # reading the data and applying configured pre-processing steps
    X_train, T_train = data.get_training_data(ntrain, normalize=normalize, deskew=deskew)
    X_test, T_test = data.get_testing_data(ntest, normalize=normalize, deskew=deskew)
    print "{0} training data read".format(len(X_train))
    print "{0} testing data read".format(len(X_test))
    print


    errors, params = [], []

    print "{0:25}{1:50}Avg. Error".format("(Metric, K)", "[Cross Validation Fold Errors]")
    print "-----------------------------------------------------------------------------------------"

    for metric in metrics:
        for k in ks:
            # cross-validating a model for each combination of hyperparameters
            fold_errors = cross_validate(X_train, T_train, folds, k, metric)

            # storing the hyperparameters and the corresponding average
            # error score resulting from the above cross-validation
            params.append((metric, k))
            errors.append(np.mean(fold_errors))

            print "{0:25}{1:50}{2:.3f}".format(
                params[-1], 
                [round(f, 2) for f in fold_errors], 
                errors[-1]
            )

    # selecting the set of hyperparameters, 
    # which caused the lowest average error
    P_selected = params[np.argmin(errors)]
    T_predicted = classify(X_test, X_train, T_train, 
        P_selected[1], P_selected[0])

    print "-----------------------------------------------------------------------------------------"
    print "Best Params: {0}, validation error: {1:.3f}".format(P_selected, np.min(errors))
    print


    if evaluate:
        # evaluating the model performance on the testing set
        print "Testing Set Error: {0:.3f}".format(
            get_error_score(T_predicted, T_test)
        )
        print

