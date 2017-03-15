import sys

import numpy as np

import data


# either sigmoid function or its derivative, depending on "deriv",
# computed for an array of input vectors (inputs are rows of matrix "X").
# if "deriv" is True, "X" is assumed to contain the values of the function
def sigmoid(X, deriv=False):
    if not deriv:
        return 1.0 / (1.0 + np.nan_to_num(np.exp(-X)))
    else:
        return X * (1.0 - X)


# either hyperbolic tangent function or its derivative, depending on "deriv",
# computed for an array of input vectors (inputs are rows of matrix "X").
# if "deriv" is True, "X" is assumed to contain the values of the function
def tanh(X, deriv=False):
    if not deriv:
        e2x = np.nan_to_num(np.exp(2.0 * X))
        return (e2x - 1.0) / (e2x + 1.0)
    else:
        return 1.0 - X * X


# either rectified linear unit function or its derivative, depending on "deriv".
# computed for an array of input vectors (inputs are rows of matrix "X").
# if "deriv" is True, "X" is assumed to contain the values of the function
def relu(X, deriv=False):
    if not deriv:
        return np.where(X > 0.0, X, 0.0)
    else:
        return np.where(X > 0.0, 1.0, 0.0)


# softmax function computed for an array of 
# input vectors (inputs are rows of matrix "X")
def softmax(X):
    exped = np.nan_to_num(np.exp(X))
    summed = np.nan_to_num(np.sum(exped, axis=1))
    return exped / summed[:, np.newaxis]


# comparing predicted vs. true labels and 
# returning the corresponding error score
def get_error_score(output, T):
    classified = np.argmax(output, axis=1)
    incorrect = np.sum(classified != T)
    return incorrect * 100.0 / len(T)


# cross-entropy loss function
def get_loss(output, T):
    logs = np.nan_to_num(np.log(output[np.arange(len(T)), T]))
    return -1.0 / len(T) * np.sum(logs)


# delta (error) of the output (softmax) layer
# computed from "output" and target ("T") values
def get_output_delta(output, T):
    delta = output.copy()
    delta[np.arange(len(T)), T] -= 1.0
    return delta


# randomly selecting hidden units for dropping out during a training
# iteration ("dropout_rate" share of units from each of the hidden
# layers is actually dropped out). the set of the weight matrices 
# "W" is used here solely to figure out the number of units 
# in each of the hidden layers
def get_dropped_out_units(W, dropout_rate=0.5):
    if dropout_rate > 0.0:
        dropped = []
        for i in range(len(W)-1):
            drop = np.arange(W[i].shape[1])
            np.random.shuffle(drop)
            drop = drop[:int(len(drop) * (1 - dropout_rate))]
            drop = drop[:, np.newaxis]
            dropped.append(drop)
        return dropped
    else:
        return None


# passing the inputs "X" through the network with weights "W"
# and computing the corresponding activations of each layer. if
# dropout_rate is non-zero (and so the "dropped_out" list returned 
# by "get_dropped_out_units" function is not None), the activations 
# of the dropped out units are explicitly set to zero.
def forward_pass(X, W, functions, dropout_rate=0.0):
    activations = [X]
    dropped_out = get_dropped_out_units(W, dropout_rate)
    for i in range(len(W)):
        linear = np.dot(activations[-1], W[i])
        activation = functions[i](linear)
        if dropped_out is not None and i < len(W)-1:
            activation[np.arange(len(X)), dropped_out[i]] = 0.0
        activations.append(activation)
    return activations


# backpropagation algorithm: first the activations are computed,
# and then the error (delta) of the output layer is propagated back
# through the layers with concomitant computation of the gradient
# of the error function with respect to network weights
def backprop(X, W, T, functions, dropout_rate=0.0):
    activations = forward_pass(X, W, functions, dropout_rate)
    delta = get_output_delta(activations[-1], T)
    dW = [np.dot(activations[-2].T, delta) / len(X)]
    for i in range(0, len(W)-1):
        delta = functions[-i-2](activations[-i-2], deriv=True) * np.dot(delta, W[-i-1].T)
        dW.insert(0, np.dot(activations[-i-3].T, delta) / len(X))
    return dW


# training neural network model based on "X"/"T" data
# by means of mini-batch stochastic gradient descent.
# the rest of the argument names seem self-explanatory
def train_nn(X, T, layers, functions, epochs, batch_size, learning_rate,
             momentum_rate, dropout_rate, dataset_split, verbose=False):
    # splitting the data into training and validation sets
    # according to the value of "dataset_split" argument
    training_set_size = int(dataset_split * len(X))
    X_training, T_training = X[:training_set_size].copy(), T[:training_set_size].copy()
    X_validation, T_validation = X[training_set_size:].copy(), T[training_set_size:].copy()

    W, W_dir = [], []
    # creating the list of weight matrices "W" according to the number of 
    # units in each pair of subsequent layers from "layers". the weights are
    # initialized by sampling from standard normal distribution and normalizing
    # by the square root of the number of incoming links (fan_in)
    for x, y in zip(layers[:-1], layers[1:]):
        W.append(np.random.randn(x, y) / np.sqrt(x))
        W_dir.append(np.zeros((x, y)))

    weights = []
    training_losses, training_errors = [], []
    validation_losses, validation_errors = [], []

    # passing the training and validation sets through
    # the network with the initially picked weights
    O_training = forward_pass(X_training, W, functions)[-1]
    O_validation = forward_pass(X_validation, W, functions)[-1]

    # storing the initial values of
    # weights, errors, and losses
    weights.append(W[:])
    training_losses.append(get_loss(O_training, T_training))
    training_errors.append(get_error_score(O_training, T_training))
    validation_losses.append(get_loss(O_validation, T_validation))
    validation_errors.append(get_error_score(O_validation, T_validation))

    if verbose:
        print
        print "Training with {0}".format((
            functions[0].__name__, 
            ":".join([str(l) for l in layers[1:len(layers)-1]]),
            batch_size, learning_rate, dropout_rate
        ))
        print "Stage\t\tTr. Loss/Error\tVal. Loss/Error"
        print "-----------------------------------------------------------"
        print "Initial\t\t{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(
            training_losses[-1], training_errors[-1],
            validation_losses[-1], validation_errors[-1]
        )

    # for each training epoch:
    for epoch in range(0, epochs):
        # randomly shuffling the training set
        p = np.random.permutation(len(X_training))
        X_training, T_training = X_training[p], T_training[p]

        # for each mini-batch (of "batch_size") computing
        # the gradient and updating the weights "W"
        for b in range(0, len(X_training) / batch_size):
            X_batch = X_training[b * batch_size:(b+1) * batch_size]
            T_batch = T_training[b * batch_size:(b+1) * batch_size]
            dW = backprop(X_batch, W, T_batch, functions, dropout_rate)

            for i in range(len(W)):
                # updating the search direction "W_dir" with
                # the negative gradient and momentum terms
                W_dir[i] = -dW[i] + momentum_rate * W_dir[i]

                # updating the weights with the search 
                # direction scaled by the "learning_rate"
                W[i] += learning_rate * W_dir[i]

        # passing the training and validation sets through
        # the network when the training epoch has ended
        O_training = forward_pass(X_training, W, functions)[-1]
        O_validation = forward_pass(X_validation, W, functions)[-1]

        # storing the weights, errors, and 
        # losses after the training epoch
        weights.append(W[:])
        training_losses.append(get_loss(O_training, T_training))
        training_errors.append(get_error_score(O_training, T_training))
        validation_losses.append(get_loss(O_validation, T_validation))
        validation_errors.append(get_error_score(O_validation, T_validation))

        if verbose:
            print "Epoch #{0}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}".format(
                epoch + 1,
                training_losses[-1], training_errors[-1],
                validation_losses[-1], validation_errors[-1]
            )

    # selecting the weights resulting from the epoch 
    # with the lowest error score on the validation set
    best_epoch = np.argmin(validation_errors)
    best_weights = weights[best_epoch]

    # if dropout has been applied ("dropout_rate" is greater than 0),
    # the weights along the links outgoing from the hidden layers are 
    # re-scaled according to "dropout_rate" applied during training
    if dropout_rate > 0.0:
        for i in range(1, len(best_weights)):
            best_weights[i] *= (1 - dropout_rate)

    if verbose:
        print "-----------------------------------------------------------"
        print "Best Epoch: {0}".format(best_epoch)
        print "Training Loss: {0:.3f}, Training Error: {1:.3f}".format(
            training_losses[best_epoch], training_errors[best_epoch])
        print "Validation Loss: {0:.3f}, Validation Error: {1:.3f}".format(
            validation_losses[best_epoch], validation_errors[best_epoch])
        print

    return best_weights, best_epoch, training_losses, training_errors, validation_losses, validation_errors


if __name__ == "__main__":

    ntrain = 60000      # number of training samples used
    ntest = 10000       # number of testing samples used

    deskew = True       # deskew input images or not (by default: yes)
    normalize = True    # normalize input vectors or not (by default: yes)
    evaluate = False    # evaluate on testing data or not (by default: no)
    verbose = False     # output details of each training epoch

    epochs = 100            # number of training epochs
    dataset_split = 0.8     # training / validation set split
    momentum_rate = 0.9     # coefficient of the momentum term

    functions = [sigmoid, tanh, relu]                   # different hidden layer activation functions tried
    hidden_units = ["250:150", "500:300", "1000:600"]   # different numbers of units in hidden layers tried
    batch_sizes = [50, 100, 200]                        # different mini-batch sizes tried
    learning_rates = [0.01, 0.05, 0.1]                  # different learning rates tried
    dropout_rates = [0.0, 0.5]                          # different dropout rates tried (0 - no dropout)

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
        elif option == "-momentum_rate":
            momentum_rate = float(sys.argv[1]); del sys.argv[1]
        elif option == "-functions":
            functions = [globals()[f] for f in sys.argv[1].split(",")]; del sys.argv[1]
        elif option == "-hidden_units":
            hidden_units = sys.argv[1].split(","); del sys.argv[1]
        elif option == "-batch_sizes":
            batch_sizes = [int(b) for b in sys.argv[1].split(",")]; del sys.argv[1]
        elif option == "-learning_rates":
            learning_rates = [float(l) for l in sys.argv[1].split(",")]; del sys.argv[1]
        elif option == "-dropout_rates":
            dropout_rates = [float(r) for r in sys.argv[1].split(",")]; del sys.argv[1]
        else:
            print sys.argv[0], ": invalid option", option
            sys.exit(1)

    np.seterr(over="ignore", divide="ignore")

    print "Neural Networks"
    print

    print "Reading data..."
    # reading the data, applying configured pre-processing, and adding 1.0 to each vector as a bias input
    X_train, T_train = data.get_training_data(ntrain, normalize=normalize, deskew=deskew, add_ones=True)
    X_test, T_test = data.get_testing_data(ntest, normalize=normalize, deskew=deskew, add_ones=True)
    print "{0} training data read".format(len(X_train))
    print "{0} testing data read".format(len(X_test))
    print

    input_dim = X_train.shape[1]
    output_dim = T_train.max() + 1
    weights, errors, params = [], [], []

    print "{0:40}\tV. Loss\t\tV. Error".format("(Func, Hidden, Batch, Learn, Drop)")
    print "-------------------------------------------------------------------------------"

    for function in functions:
        for hidden in hidden_units:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for dropout_rate in dropout_rates:
                        # fixing the seed of randomization for the sake of
                        # reproducibility of the randomized training process
                        np.random.seed(1)

                        # setting up the numbers of units in each layer
                        # as well as the corresponding activation functions
                        # (output activation function is always "softmax")
                        layers = [input_dim] + [int(h) for h in hidden.split(":")] + [output_dim]
                        functions = [function] * (len(layers)-2) + [softmax]

                        # training a model for each combination of hyperparameters
                        W_best, E_best, T_loss, T_err, V_loss, V_err = train_nn(
                            X_train, T_train, 
                            layers, functions, 
                            epochs, batch_size, 
                            learning_rate, momentum_rate, dropout_rate,
                            dataset_split, verbose=verbose
                        )

                        # storing the weights and the corresponding validation
                        # error resulting from the above training, together with
                        # the values of the hyperparameters tried
                        weights.append(W_best)
                        errors.append(V_err[E_best])
                        params.append((
                            function.__name__, 
                            hidden, batch_size, 
                            learning_rate, dropout_rate
                        ))

                        print "{0:40}\t{1:.3f}\t\t{2:.3f}".format(
                            params[-1], V_loss[E_best], V_err[E_best]
                        )

    # selecting the set of hyperparameters, 
    # which caused the lowest validation error,
    # with the respective resulting weights
    P_selected = params[np.argmin(errors)]
    W_selected = weights[np.argmin(errors)]

    print "-------------------------------------------------------------------------------"
    print "Best Params: {0}, validation error: {1:.3f}".format(P_selected, np.min(errors))
    print

    if evaluate:
        # passing the testing set through the trained network
        O_test = forward_pass(X_test, W_selected, functions)[-1]

        # evaluating the model performance on the testing set
        print "Testing Set Error: {0:.3f}".format(
            get_error_score(O_test, T_test)
        )
        print
