import sys

import numpy as np
import scipy.spatial.distance as sp

from code import data


# computing covariance matrix for GP by means of SE kernel:
# depending on what Xs and Ys are, this can be K, K*, or K**.
# scipy.spatial.distance.pdist and .cdist are used to compute
# the matrix of squared Euclidian distances efficiently.
# "lsquared" is an argument, sigmaF is assumed to be 1.
def compute_SE_kernel_matrix(Xs, Ys, lsquared):
    if Xs is Ys:
        dist = sp.squareform(sp.pdist(Xs, "sqeuclidean"))
    else:
        dist = sp.cdist(Xs, Ys, "sqeuclidean")
    return np.exp(-dist / (2.0 * lsquared))


# actual Gaussian Process: the target values for unobserved data
# X* ("X_s") are inferred based on the observed dataset "X" and the
# corresponding labels "T". "lsquared" is used while computing the
# SE-kernel function. sigmaF and sigmaN are taken as 1 and 0, as
# only mean, and not variance, is of practical importance here.
def gaussian_process(X, T, X_s, lsquared):
    # computing the K matrix for the observed data
    K = compute_SE_kernel_matrix(X, X, lsquared)
    print "K matrix computed"

    # computing the K* transposed matrix
    K_s_T = compute_SE_kernel_matrix(X_s, X, lsquared)
    print "K*^T matrix computed"

    # inverting the K matrix
    K_inv = np.linalg.inv(K)
    print "K matrix inverted"

    # multiplying the K*^T and K^-1 matrices
    K_s_T_times_K_inv = np.dot(K_s_T, K_inv)
    print "K*^T times K^-1 matrix computed"

    inputs, classes = len(X_s), T.max()+1
    predictions = np.zeros((inputs, classes))

    # for each class k:
    for k in range(classes):
        # transforming target labels into k-class vs. rest
        # representation: 1.0 for the k-class, -1.0 for rest
        k_class_vs_rest = np.where(T == k, 1.0, -1.0)

        # inferring the corresponding k-class (1.0) vs. the rest (-1.0) values 
        # for the unobserved data by multiplying pre-computed [K*^T times K^-1]
        # matrix by the above "k_class_vs_rest" vector. what is obtained is the
        # set of the mean values of the k-class vs. rest regression in the 
        # unobserved data points X* ("X_s")
        result = np.dot(K_s_T_times_K_inv, k_class_vs_rest)

        # storing the predicted k-class vs. 
        # rest means in the data points X*
        predictions[:,k] = result

    print "{0} binary classifications done".format(classes)

    # inferring actual target labels in accordance with
    # the highest predicted k-class vs. rest mean
    labels = np.argmax(predictions, axis=1)
    print "Class labels detected"
    print

    return labels


# comparing predicted vs. true labels 
# and returning the corresponding error score
def get_error_score(T_predicted, T_true):
    count = np.count_nonzero(T_predicted != T_true)
    return count * 100.0 / len(T_predicted)



if __name__ == "__main__":

    # command-line arguments (and their default values)

    ntrain = 60000      # number of training samples used
    ntest = 10000       # number of testing samples used

    deskew = True       # deskew input images or not (by default: yes)
    normalize = True    # normalize input vectors or not (by default: yes)

    lsquared = 33.0     # l^2 used in SE kernel computation


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
        elif option == "-lsquared":
            lsquared = float(sys.argv[1]); del sys.argv[1]
        else:
            print sys.argv[0], ": invalid option", option
            sys.exit(1)


    print "Gaussian Processes"
    print

    print "Reading data..."
    # reading the data and applying configured pre-processing steps
    X_train, T_train = data.get_training_data(ntrain, normalize=normalize, deskew=deskew)
    X_test, T_test = data.get_testing_data(ntest, normalize=normalize, deskew=deskew)
    print "{0} training data read".format(len(X_train))
    print "{0} testing data read".format(len(X_test))
    print


    # running a Gaussian process on training and testing sets, with "lsquared"
    T_predicted = gaussian_process(X_train, T_train, X_test, lsquared=lsquared)


    # evaluating the model performance on the testing set
    print "Testing Set Error: {0:.3f}".format(
        get_error_score(T_predicted, T_test)
    )
    print

