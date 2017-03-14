import sys

import numpy as np

from code import data


# softmax function computed for an array of 
# input vectors (inputs are rows of matrix "X")
def softmax(X):
    exped = np.nan_to_num(np.exp(X))
    summed = np.sum(exped, axis=1)
    return exped / summed[:,np.newaxis]


# classifying the data "X"/"T" using the weights "W"
# and returning the corresponding error score
def get_error_score(X, W, T):
    linear = np.dot(X, W)
    classified = np.argmax(linear, axis=1)
    incorrect = np.sum(classified != T)
    return incorrect * 100.0 / len(X)

# cross-entropy loss function, with L2-regularization
def get_loss(X, W, T, regularization):
    sm = softmax(np.dot(X, W))
    logs = np.nan_to_num(np.log(sm[np.arange(len(X)),T]))
    regs = 0.5 * regularization * np.sum(W * W)
    return -1.0 / len(X) * np.sum(logs) + regs

# gradient of the loss function with respect 
# to the weights "W", with L2-regularization
def get_gradient(X, W, T, regularization):
    delta = softmax(np.dot(X, W))
    delta[np.arange(len(X)), T] -= 1
    regs = regularization * W
    return np.dot(X.T, delta) / len(X) + regs


# fitting logistic regression model to "X"/"T" data
# by means of mini-batch stochastic gradient descent.
# the rest of the argument names seem self-explanatory
def train_logreg(X, T, epochs, batch_size, learning_rate, regularization, dataset_split, verbose=False):
    # splitting the data into training and validation sets
    # according to the value of "dataset_split" argument
    training_set_size = int(dataset_split * len(X))
    X_training, T_training = X[:training_set_size].copy(), T[:training_set_size].copy()
    X_validation, T_validation = X[training_set_size:].copy(), T[training_set_size:].copy()

    inputs, outputs = X.shape[1], T.max()+1
    W = np.zeros((inputs, outputs))

    weights = []
    training_losses, training_errors = [], []
    validation_losses, validation_errors = [], []


    # storing the initial values of
    # weights, errors, and losses
    weights.append(W.copy())
    training_losses.append(get_loss(X_training, W, T_training, regularization))
    training_errors.append(get_error_score(X_training, W, T_training))
    validation_losses.append(get_loss(X_validation, W, T_validation, regularization))
    validation_errors.append(get_error_score(X_validation, W, T_validation))

    if verbose:
        print
        print "Training with {0}".format((batch_size, learning_rate, regularization))
        print "Stage\t\tTr. Loss/Error\tVal. Loss/Error"
        print "------------------------------------------------------"
        print "Initial\t\t{0:.3f}\t{1:.2f}\t{2:.3f}\t{3:.2f}".format(
            training_losses[-1], training_errors[-1],
            validation_losses[-1], validation_errors[-1]
        )

    # for each training epoch:
    for epoch in range(0, epochs):
        # randomly shuffling the training set
        p = np.random.permutation(len(X_training))
        X_training, T_training = X_training[p], T_training[p]

        # for each mini-batch (of "batch_size") computing
        # the gradient and updating the weights "W" (subtracting
        # gradient) after scaling the gradient by the "learning_rate"
        for b in range(0, X_training.shape[0] / batch_size):
            X_batch = X_training[b * batch_size : (b+1) * batch_size]
            T_batch = T_training[b * batch_size : (b+1) * batch_size]
            W -= learning_rate * get_gradient(X_batch, W, T_batch, regularization)

        # storing the weights, errors, and 
        # losses after each training epoch
        weights.append(W.copy())
        training_losses.append(get_loss(X_training, W, T_training, regularization))
        training_errors.append(get_error_score(X_training, W, T_training))
        validation_losses.append(get_loss(X_validation, W, T_validation, regularization))
        validation_errors.append(get_error_score(X_validation, W, T_validation))

        if verbose:
            print "Epoch #{0}\t{1:.3f}\t{2:.2f}\t{3:.3f}\t{4:.2f}".format(
                epoch + 1,
                training_losses[-1], training_errors[-1],
                validation_losses[-1], validation_errors[-1]
            )

    # selecting the weights resulting from the epoch 
    # with the lowest error score on the validation set
    best_epoch = np.argmin(validation_errors)
    best_weights = weights[best_epoch]

    if verbose:
        print "------------------------------------------------------"
        print "Best Epoch: {0}".format(best_epoch)
        print "Training Loss: {0:.3f}, Training Error: {1:.2f}".format(training_losses[best_epoch], training_errors[best_epoch])
        print "Validation Loss: {0:.3f}, Validation Error: {1:.2f}".format(validation_losses[best_epoch], validation_errors[best_epoch])
        print

    return (best_weights, best_epoch, training_losses, training_errors, validation_losses, validation_errors)



if __name__ == "__main__":

    ntrain = 60000      # number of training samples used
    ntest = 10000       # number of testing samples used

    deskew = True       # deskew input images or not (by default: yes)
    normalize = True    # normalize input vectors or not (by default: yes)
    evaluate = False    # evaluate on testing data or not (by default: no)
    verbose = False     # output details of each training epoch

    epochs = 100            # number of training epochs
    dataset_split = 0.8     # training / validation set split

    batch_sizes = [50, 100, 200]                    # different mini-batch sizes tried
    learning_rates = [0.02, 0.05, 0.1, 0.5, 1.0]    # different learning rates tried
    regularizations = [0.0, 0.0001, 0.0005]         # different regularization parameters tried


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
        elif option == "-verbose":
            verbose = int(sys.argv[1]); del sys.argv[1]
        elif option == "-epochs":
            epochs = int(sys.argv[1]); del sys.argv[1]
        elif option == "-dataset_split":
            dataset_split = float(sys.argv[1]); del sys.argv[1]
        elif option == "-batch_sizes":
            batch_sizes = [int(b) for b in sys.argv[1].split(",")]; del sys.argv[1]
        elif option == "-learning_rates":
            learning_rates = [float(l) for l in sys.argv[1].split(",")]; del sys.argv[1]
        elif option == "-regularizations":
            regularizations = [float(r) for r in sys.argv[1].split(",")]; del sys.argv[1]
        else:
            print sys.argv[0], ": invalid option", option
            sys.exit(1)


    np.seterr(over="ignore", divide="ignore")


    print "Logistic Regression"
    print

    print "Reading data..."
    # reading the data, applying configured pre-processing, and adding 1.0 to each vector as a bias input
    X_train, T_train = data.get_training_data(ntrain, normalize=normalize, deskew=deskew, add_ones=True)
    X_test, T_test = data.get_testing_data(ntest, normalize=normalize, deskew=deskew, add_ones=True)
    print "{0} training data read".format(len(X_train))
    print "{0} testing data read".format(len(X_test))
    print


    weights, errors, params = [], [], []

    print "{0:25}\tV. Loss\t\tV. Error".format("(Batch, Learn, Reg)")
    print "-----------------------------------------------------------"

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for regularization in regularizations:
                # fixing the seed of randomization for the sake of
                # reproducibility of the randomized training process
                np.random.seed(1)

                # training a model for each combination of hyperparameters
                W_best, E_best, T_loss, T_err, V_loss, V_err = train_logreg(
                    X_train, T_train, 
                    epochs, batch_size, 
                    learning_rate, regularization, 
                    dataset_split, verbose=verbose
                )

                # storing the weihgts and the corresponding validation
                # error resulting from the above training, together with
                # the values of the hyperparameters tried
                weights.append(W_best)
                errors.append(V_err[E_best])
                params.append((
                    batch_size, 
                    learning_rate, 
                    regularization
                ))

                print "{0:25}\t{1:.3f}\t\t{2:.3f}".format(
                    params[-1], V_loss[E_best], V_err[E_best]
                )

    # selecting the set of hyperparameters, 
    # which caused the lowest validation error,
    # with the respective resulting weights
    P_selected = params[np.argmin(errors)]
    W_selected = weights[np.argmin(errors)]

    print "-----------------------------------------------------------"
    print "Best Params: {0}, validation error: {1:.3f}".format(P_selected, np.min(errors))
    print


    if evaluate:
        # evaluating the model performance on the testing set
        print "Testing Set Error: {0:.3f}".format(
            get_error_score(X_test, W_selected, T_test)
        )
        print

