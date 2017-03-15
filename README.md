MNIST Challenge
===========================================================

MNIST handwritten digits database recognition challenge was held as a part of "Machine Learning 1" course (WS16/17) at TU Munich. The rules of the challenge are given (as stated by instructors) in the "/report/rules.txt" file. This repository hosts my submission consisting of the following Python source code files (residing in the "code" folder):

- data.py [reading and pre-processing MNIST]
- knn.py [K-Nearest Neighbor implementation]
- logreg.py [Logistic Regression implementation]
- nn.py [Deep Neural Network implementation]
- gp.py [Gaussian Processes implementation]

All commands mentioned below run one of the above Python scripts. They were tested (and generated the respective outputs from the "report" folder) in Anaconda Python 2.7. All Python source code files are supplied with detailed comments.


Preparation
-----------------------------------------------------------
Four uncompressed original MNIST data files should be placed into the same folder with the scripts. These files are "train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", and "t10k-labels-idx1-ubyte". The original names of the files (just as indicated in the previous sentence) should be preserved.


K-Nearest Neighbors
-----------------------------------------------------------
To reproduce the cross-validation with the found optimal set of hyperparameters and subsequent evaluation on the testing set, following command should be executed from the command line (the expected output of this command run is given in "/report/knn_output_short.txt"):

    python knn.py -metrics cosine -ks 3 -evaluate 1

To reproduce the full hyperparameter search (multiple cross-validations for each considered set of hyperparameters) and subsequent evaluation of the found optimal set of hyperparameters on the testing set, following command should be executed from the command line (the expected output of this command run is given in "/report/knn_output_long.txt"):

    python knn.py -evaluate 1


Logistic Regression
-----------------------------------------------------------
To reproduce the training with the found optimal set of hyperparameters and subsequent evaluation on the testing set, following command should be executed from the command line (the expected output of this command run is given in "/report/logreg_output_short.txt"):

    python logreg.py -batch_sizes 100 -learning_rates 1.0 -regularizations 0.0001 -evaluate 1 -verbose 1

To reproduce the full hyperparameter search (one training for each considered set of hyperparameters) and subsequent evaluation of the found optimal set of hyperparameters on the testing set, following command should be executed from the command line (the expected output of this command run is given in "/report/logreg_output_long.txt"):

    python logreg.py -evaluate 1

The seed of the random number generator is explicitly set to "1" before beginning of each training for the sake of reproducibility of the result.


Neural Networks
-----------------------------------------------------------
To reproduce the training with the found optimal set of hyperparameters and subsequent evaluation on the testing set, following command should be executed from the command line (the expected output of this command run is given in "/report/nn_output_short.txt"):

    python nn.py -functions relu -hidden_units 1000:600 -batch_sizes 100 -learning_rates 0.1 -dropout_rates 0.5 -evaluate 1 -verbose 1

To reproduce the full hyperparameter search (one training for each considered set of hyperparameters) and subsequent evaluation of the found optimal set of hyperparameters on the testing set, following command should be executed from the command line (the expected output of this command run is given in "/report/nn_output_long.txt"):

    python nn.py -evaluate 1

The seed of the random number generator is explicitly set to "1" before beginning of each training for the sake of reproducibility of the result.


Gaussian Processes
-----------------------------------------------------------
To reproduce the "training" process restricted to the first 40,000 training inputs, following command should be executed from the command line (the expected output of this command run is given in "/report/gp_output.txt"):

    python gp.py -ntrain 40000

Successful completion of the above command requires at least 32GB of RAM. If this amount of RAM is not available, the run with the lower amount of training inputs may be attempted (e.g. "python gp.py -ntrain 30000" requires less RAM). If, however, the amount of available RAM is ample enough, the reviewer is encouraged to try running the command with 50000 training inputs, or even with the whole training data (by omitting "-ntrain" command-line argument altogether) to achieve even higher predictive accuracy.
