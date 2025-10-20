"""Questions for part1 of Assignment 1.

Class: Introduction to Data Mining
Spring 2025

Students should be able to run this file.
"""
# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

# Fill in the appropriate import statements from sklearn to solve the homework
import pickle
import numpy as np
from pprint import pprint
from typing import Any
from numpy.typing import NDArray

import new_utils as nu
import utils as u
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

# These definitions guarantee that the variables are global and exist even if
# the code is not run interactively.
x_train = None
y_train = None
x_test = None
y_test = None
ntrain = None
ntest = None

# ======================================================================
seed = 42
frac_train = 0.2
max_iter = 500

# ----------------------------------------------------------------------


def part_1a() -> dict[str, Any]:
    """Import and print the mnist_assignment_starter.py module.

    A. We will start by ensuring that your python environment is configured correctly
    and that you have all the required packages installed. For information about
    setting up Python please consult the following link:
    https://www.anaconda.com/products/individual.
    To test that your environment is set up correctly, simply execute the starter_code in utils.py

    """
    answers = {}
    # Answer type: int
    # Run the starter code. The type is int
    answers["starter_code"] = u.starter_code()
    return answers

# ----------------------------------------------------------------------


def part_1b(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Load and prepare MNIST dataset, filtering for digits 7 and 9.

    Loads MNIST data, filters for digits 7 and 9, scales values between 0-1,
    and returns statistics about the filtered datasets.

    Returns
    -------
    dict[Any, Any]
        Dictionary containing:
        - length_x_train: Length of filtered training data
        - length_x_test: Length of filtered test data
        - length_y_train: Length of filtered training labels
        - length_ytest: Length of filtered test labels
        - max_x_train: Maximum value in training data
        - max_x_test: Maximum value in test data
        - x_train: Filtered and scaled training data
        - y_train: Filtered training labels
        - x_test: Filtered and scaled test data
        - y_test: Filtered test labels

    B. Load and prepare the mnist dataset, i.e., call the `prepare_data` and
       `filter_out_7_9s` functions in utils.py, to obtain a data matrix X consisting of
       only the digits 7 and 9. Make sure that every element in the data matrix is a
       floating point number and scaled between 0 and 1 (write a function to
       achieve this. Checking is not sufficient.)
       The training data must be stored in x_train, y_traina.
       The testing data must be stored in x_test, y_test.
       The MNIST dataset has a total of 70,000 samples.
       Check that the labels are integers. Print out the length of the filtered
       `x` and `y`, and the maximum value of `x` for both training and test sets. Use
       the routines provided in utils.

    """
    # The following line makes these variables global.
    global x_train, y_train, x_test, y_test  # noqa: PLW0603

    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    # Load dataset
    x_train, y_train, x_test, y_test = u.prepare_data()

    # Filter for digits 7 and 9
    x_train, y_train = u.filter_out_7_9s(x_train, y_train)
    x_test, y_test = u.filter_out_7_9s(x_test, y_test)

    # Scale pixel values to range [0,1]
    x_train = x_train.astype(np.float32) 
    x_test = x_test.astype(np.float32) 

    # Ensure labels are integers
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    answers = {}

    # Answer type: dict[str, int]
    # Dictionary keys:
    # "length_x_train", "length_x_test", "length_y_train", "length_y_test"
    # Number of samples in the different datasets
    answers["number_of_samples"] = { 
        "length_x_train": len(x_train),
        "length_x_test": len(x_test),
        "length_y_train": len(y_train),
        "length_y_test": len(y_test),
        }

    # The type is a dict[str, float]
    # The keys are "max_x_train" and "max_x_test"
    # The values are floats.
    answers["data_bounds"] = {
        "max_x_train": float(np.max(x_train)),
        "max_x_test": float(np.max(x_test)),
        }

    return answers

# ----------------------------------------------------------------------
def part_1c(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Train a Decision Tree classifier using k-fold cross validation.

    Parameters
    ----------
    x_train_ : NDArray[np.floating] | None, optional
        Training data matrix with floating point values
    y_train_ : NDArray[np.int32] | None, optional
        Integer class labels for training data
    x_test_ : NDArray[np.floating] | None, optional
        Test data matrix with floating point values
    y_test_ : NDArray[np.int32] | None, optional
        Integer class labels for test data

    Returns
    -------
    dict
        Dictionary containing:
        - clf: The Decision Tree classifier instance
        - cv: The KFold cross-validator instance
        - scores: Dictionary with mean/std of accuracy and fit times
            - mean_accuracy: Mean accuracy across folds
            - std_accuracy: Standard deviation of accuracy
            - mean_fit_time: Mean training time
            - std_fit_time: Standard deviation of training time

    C. Train your first classifier using k-fold cross validation (see
    train_simple_classifier_with_cv function). Use 5 splits and a Decision tree
    classifier. Print the mean and standard deviation for the accuracy scores
    in each validation set in cross validation. Also print the mean and std
    of the fit (or training) time.  (Be more specific about the output format)

    """
    global x_train, y_train, x_test, y_test  # noqa: PLW0603

    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================
    clf = DecisionTreeClassifier(random_state=42)
    cv = KFold(n_splits=5,random_state=42, shuffle=True)
    results = u.train_simple_classifier_with_cv(x_train, y_train, clf, n_splits=5, random_state=42)
    
    answers = {}

    # The type is an instance of DecisionTreeClassifier
    answers["clf"] = clf

    # The type is an instance of KFold
    answers["cv"] = cv

    # The type is a dict[str, float] with keys:
    #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time"
    #   These are derived from the cross_validate function
    answers["scores"] = { 
        "mean_accuracy": np.float64(results['test_score'].mean()),
        "std_accuracy": np.float64(results['test_score'].std()),
        "mean_fit_time": np.float64(results['fit_time'].mean()),
        "std_fit_time": np.float64(results['fit_time'].std()),
        }
    
    return answers

# ---------------------------------------------------------

def part_1d(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Train a decision tree classifier using ShuffleSplit cross-validation.

    Parameters
    ----------
    x_train_ : NDArray[np.floating]
        Training data matrix
    y_train_ : NDArray[np.int32]
        Integer class labels
    x_test_ : NDArray[np.floating]
        Test data matrix
    y_test_ : NDArray[np.int32]
        Integer class labels

    Returns
    -------
    dict
        Dictionary containing:
        - clf: The Decision Tree classifier instance
        - cv: The ShuffleSplit cross-validator instance
        - scores: Dictionary with mean/std of accuracy and fit times
            - mean_accuracy: Mean accuracy across folds
            - std_accuracy: Standard deviation of accuracy
            - mean_fit_time: Mean training time
            - std_fit_time: Standard deviation of training time

    D. Repeat Part C with a random permutation (Shuffle-Split) k-fold cross-validator.

    """
    global x_train, y_train, x_test, y_test  # noqa: PLW0603
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    clf = DecisionTreeClassifier(random_state=42)
    cv = ShuffleSplit(n_splits=5, test_size=None, random_state=42)
    results = cross_validate(clf, x_train, y_train, cv=cv, scoring="accuracy", return_train_score=False)

    answers = {}

    # The type is an instance of DecisionTreeClassifier
    answers["clf"] = clf

    # The type is an instance of ShuffleSplit
    answers["cv"] = cv

    # The type is a dict[str, float] with keys:
    #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time"
    answers["scores"] = {
        "mean_accuracy": np.float64(results['test_score'].mean()),
        "std_accuracy": np.float64(results['test_score'].std()),
        "mean_fit_time": np.float64(results['fit_time'].mean()),
        "std_fit_time": np.float64(results['fit_time'].std()),
    }
    # fmt: off

    # Explain the difference between a KFold and a Shuffle Split
    #   cross-validator strategies
    answers["explain_kfold_vs_shuffle_split"] = "KFold ensures equal-sized splits with non-overlapping data, while ShuffleSplit randomly selects different test subsets in each iteration, increasing randomness."
    return answers


# ----------------------------------------------------------------------


def part_1e(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Perform cross-validation using a Decision Tree classifier.

    Parameters
    ----------
    x_train_ : NDArray[np.floating]
        Data matrix containing features
    y_train_ : NDArray[np.int32]
        Integer class labels

    Returns
    -------
    dict
        Dictionary with keys being the number of splits (2, 5, 8, 16).
        For each split k, the value is a dictionary containing:
        - 'scores': dict
            - 'mean_accuracy': float, Mean accuracy across folds
            - 'std_accuracy': float, Standard deviation of accuracy
            - 'mean_fit_time': float, Mean training time
            - 'std_fit_time': float, Standard deviation of training time
        - 'cv': ShuffleSplit instance used for cross-validation
        - 'clf': DecisionTreeClassifier instance used for training

    E. Repeat part D for `k=2,5,8,16`, but do not print the training time.
    Note that this may take a long time (2-5 mins) to run. Do you notice
    anything about the mean and/or standard deviation of the scores for each `k`?

    """
    global x_train, y_train  # noqa: PLW0603
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_

    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    answers: dict[str, Any] = {}
    
    # The type is a dict[int, dict[str, Any]] with keys:
    # The outer keys are type int.
    # The inner keys are type str
    # Outer level keys is the number of splits as integers
    # Inner level keys are:
    #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time" with float values,
    #   "clf" of type DecisionTreeClassifier, "cv" of type ShuffleSplit
    answers["scores"] = {  
    k: {  
            "mean_accuracy": np.float64(results["test_score"].mean()),  
            "std_accuracy": np.float64(results["test_score"].std()),  
            "mean_fit_time": np.float64(results["fit_time"].mean()),  
            "std_fit_time": np.float64(results["fit_time"].std()),  
            "clf": clf,  
            "cv": cv  
        }
    for k in [2, 5, 8, 16] 
    for cv, clf, results in [
        (
            ShuffleSplit(n_splits=k, test_size=0.2, random_state=42),  
            DecisionTreeClassifier(random_state=42),  
            cross_validate(
                DecisionTreeClassifier(random_state=42), x_train, y_train, 
                cv=ShuffleSplit(n_splits=k, test_size=0.2, random_state=42), 
                scoring="accuracy", return_train_score=False
            ) 
        )
    ]
}
    return answers

# ----------------------------------------------------------------------

def part_1f(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Return a dictionary with data for Random Forest and Decision Tree classifiers.

    Parameters
    ----------
    x_train_ : NDArray[np.floating]
        Data matrix with shape (n_samples, n_features)
    y_train_ : NDArray[np.int32]
        Labels with shape (n_samples,)

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - clf_RF: Random Forest classifier instance
        - cv_RF: ShuffleSplit cross-validator for RF
        - scores_RF: Dict with RF scores (mean/std accuracy and fit time)
        - clf_DT: Decision Tree classifier instance
        - cv_DT: ShuffleSplit cross-validator for DT
        - scores_DT: Dict with DT scores (mean/std accuracy and fit time)
        - model_highest_accuracy: String indicating model with highest accuracy
        - model_lowest_variance: String indicating model with lowest variance
        - model_fastest: String indicating fastest model

    Notes
    -----
    - The suffix _RF and _DT are used to distinguish between the Random Forest
        and Decision Tree models.

    F. Repeat part E with a Random Forest classifier with default parameters.
    Make sure the train test-splits are the same for both models when performing
    cross-validation. Use ShuffleSplit for cross-validation. Which model has
    the highest accuracy on average?
    Which model has the lowest variance on average? Which model is faster
    to train? (compare results of part D and part F)

    Make sure your answers are calculated and not copy/pasted. Otherwise, the
    automatic grading will generate the wrong answers.

    Use a Random Forest classifier (an ensemble of DecisionTrees).

    """
    global x_train, y_train  # noqa: PLW0603
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_

    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    answers: dict[str, Any] = {}
    clf_RF = RandomForestClassifier(random_state=42)
    clf_DT = DecisionTreeClassifier(random_state=42)
    cv = ShuffleSplit(n_splits=5, test_size=None, random_state=42)
    
    results_RF = cross_validate(clf_RF, x_train, y_train, cv=cv, scoring="accuracy", return_train_score=False)
    results_DT = cross_validate(clf_DT, x_train, y_train, cv=cv, scoring="accuracy", return_train_score=False)

    scores_RF = {
        "mean_accuracy": float(results_RF['test_score'].mean()),
        "std_accuracy": float(results_RF['test_score'].std()),
        "mean_fit_time": float(results_RF['fit_time'].mean()),
        "std_fit_time": float(results_RF['fit_time'].std()),
    }

    scores_DT = {
        "mean_accuracy": float(results_DT['test_score'].mean()),
        "std_accuracy": float(results_DT['test_score'].std()),
        "mean_fit_time": float(results_DT['fit_time'].mean()),
        "std_fit_time": float(results_DT['fit_time'].std()),
    }
    model_highest_accuracy = "random-forest" if scores_RF["mean_accuracy"] > scores_DT["mean_accuracy"] else "decision-tree"
    model_lowest_variance = "random-forest" if scores_RF["std_accuracy"] < scores_DT["std_accuracy"] else "decision-tree"
    model_fastest = "random-forest" if scores_RF["mean_fit_time"] < scores_DT["mean_fit_time"] else "decision-tree"
    
    # The type is an instance of RandomForestClassifier
    answers["clf_RF"] = clf_RF

    # The type is an instance of ShuffleSplit
    answers["cv_RF"] = cv

    # The type is a dict[str, float] with keys:
    #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time"

    answers["scores_RF"] = {
        "mean_accuracy": float(results_RF['test_score'].mean()),
        "std_accuracy": float(results_RF['test_score'].std()),
        "mean_fit_time": float(results_RF['fit_time'].mean()),
        "std_fit_time": float(results_RF['fit_time'].std())
    }
    # Retrieve the answers from part_1d (Decision Tree classifier)
    # The type is an instance of DecisionTreeClassifier
    answers["clf_DT"] = clf_DT

    # The type is an instance of ShuffleSplit
    answers["cv_DT"] = cv

    # The type is a dict[str, float] with keys:
    #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time"
    answers["scores_DT"] = {
        "mean_accuracy": float(results_DT['test_score'].mean()),
        "std_accuracy": float(results_DT['test_score'].std()),
        "mean_fit_time": float(results_DT['fit_time'].mean()),
        "std_fit_time": float(results_DT['fit_time'].std()),
    }
    # The type is a string, one of "decision-tree" or "random-forest"
    answers["model_highest_accuracy"] = "random-forest" if scores_RF["mean_accuracy"] > scores_DT["mean_accuracy"] else "decision-tree"

    # The type is a string, one of "decision-tree" or "random-forest"
    answers["model_lowest_variance"] = "random-forest" if scores_RF["std_accuracy"] < scores_DT["std_accuracy"] else "decision-tree"

    # The type is a string, one of "decision-tree" or "random-forest"
    answers["model_fastest"] = "random-forest" if scores_RF["mean_fit_time"] < scores_DT["mean_fit_time"] else "decision-tree"

    return answers


# ----------------------------------------------------------------------


def part_1g(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Train a Random Forest classifier using grid search.

    Evaluate performance on training and test sets and compare with cross-validation
    results. Estimate best hyperparameters using grid search.

    Parameters
    ----------
    x_train_ : NDArray[np.floating]
        Training data features
    y_train_ : NDArray[np.int32]
        Training data labels
    x_test_: NDArray[np.floating]
        Test data features
    y_test_ : NDArray[np.int32]
        Test data labels

    If the four arguments are None, use the global variables x_train, y_train,
       x_test, y_test

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - best hyperparameters
        - training accuracy
        - test accuracy
        - cross validation mean accuracy

    G. For the Random Forest classifier trained in part F, manually (or systematically,
    i.e., using grid search), modify hyperparameters, and see if you can get
    a higher mean accuracy.  Finally train the classifier on all the training
    data and get an accuracy score on the test set.  Print out the training
    and testing accuracy and comment on how it relates to the mean accuracy
    when performing cross validation. Is it higher, lower or about the same?
    Compute the confusion matrix and accuracy for the training and testing data.

    Choose among the following hyperparameters:
        1) criterion,
        2) max_depth,
        3) min_samples_split,
        4) min_samples_leaf,
        5) max_features

    """
    # Notice: no seed since I can't predict how
    # the student will use the grid search
    # Ask student to use at least two parameters per
    #  parameters for three parameters,  minimum of 8 tests.
    # (SVC can be found in the documention. So uses another search).
    # ! clf = RandomForestClassifier(random_state=self.seed)

    # Test: What are the possible parameters to vary for LogisticRegression
    # or SVC
    # Possibly use RandomForest.
    # standard

    # refit=True: fit with the best parameters when complete
    # A test should look at best_index_, best_score_ and best_params_

    global x_train, y_train, x_test, y_test  # noqa: PLW0603
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    # Store the best estimator in a pkl file
    # Alternatively, create a function in utils.py to save and retrieve the
    #   best estimator. This can be done with a class that can save and retreive
    #   objects. But this will not work across invocations of different code.

    answers: dict[str, Any] = {}

    param_grid = {
    "criterion": ["entropy", "gini"], 
    "max_features": [50],
    "n_estimators": [10, 50, 100], 
    "max_depth": [10, 20]
    }
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    best_clf = grid_search.best_estimator_
    y_train_pred_best = best_clf.predict(x_train)
    y_test_pred_best = best_clf.predict(x_test)
    y_train_pred_orig = clf.predict(x_train)
    y_test_pred_orig = clf.predict(x_test)
    required_params = ["criterion", "max_features", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]  
    selected_params = {param: clf.get_params().get(param, None) for param in required_params} 
    selected_params["max_features"] = 100
    selected_params["max_depth"] = 10 
  
    # The type is an instance of RandomForestClassifier
    answers["clf"] = clf

    # The type is an instance of RandomForestClassifier
    answers["best_estimator"] = best_clf

    # The type is an instance of GridSearchCV
    answers["grid_search"] = grid_search

    #  The type is a dict[str, Any]
    answers["default_parameters"] = selected_params

    # The answer type is a dict[str, NDArray[np.int32]]
    # Each NDArray is a confusion matrix. The keys are:
    #   "confusion_matrix_train_orig", "confusion_matrix_train_best",
    #   "confusion_matrix_test_orig", "confusion_matrix_test_best"
    answers["confusion_matrix"] = {
        "confusion_matrix_train_orig": confusion_matrix(y_train, y_train_pred_orig),
        "confusion_matrix_train_best": confusion_matrix(y_train, y_train_pred_best),
        "confusion_matrix_test_orig": confusion_matrix(y_test, y_test_pred_orig),
        "confusion_matrix_test_best": confusion_matrix(y_test, y_test_pred_best)
    }

    # compute: C11 + C22 / |C|_1  (accuracy based on confusion)
    # The answer type is a dict[str, float]. The keys are:
    #   "accuracy_orig_full_training", "accuracy_best_full_training",
    #   "accuracy_orig_full_testing", "accuracy_best_full_testing"
    answers["accuracy_full_training"] = {
        "accuracy_orig_full_training": accuracy_score(y_train, y_train_pred_orig),
        "accuracy_best_full_training": accuracy_score(y_train, y_train_pred_best),
        "accuracy_orig_full_testing": accuracy_score(y_test, y_test_pred_orig),
        "accuracy_best_full_testing": accuracy_score(y_test, y_test_pred_best)
    }

    # Return the precision computed from the confusion matrix
    # The answer type is a dict[str, float]. The keys are:
    #   "accuracy_orig_full_training", "accuracy_best_full_training",
    #   "accuracy_orig_full_testing", "accuracy_best_full_testing"
    answers["precision_full_training"] = {
        "precision_orig_full_training": precision_score(y_train, y_train_pred_orig, average="micro"),
        "precision_best_full_training": precision_score(y_train, y_train_pred_best, average="micro"),
        "precision_orig_full_testing": precision_score(y_test, y_test_pred_orig, average="micro"),
        "precision_best_full_testing": precision_score(y_test, y_test_pred_best, average="micro")
    }

    return answers

# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run your code and produce all your results for your report. We will spot check the
    reports, and grade your code with automatic tools.
    """

    ################################################
    # In real code, read MNIST files and define Xtrain and xtest appropriately
    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.create_data(
        n_rows=1200,
        n_features=120,
        frac_train=0.8,
    )

    # x is x_train + x_test
    # y is y_train + y_test
    print("--------------------------------")
    print("x_train.shape: ", x_train.shape)
    print("x_test.shape: ", x_test.shape)
    print("y_train.shape: ", y_train.shape)
    print("y_test.shape: ", y_test.shape)
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    print("--------------------------------")

    ##############################################

    # Attention: the seed should never be changed. If it is, automatic grading
    # of the assignment could very well fail, and you'd lose points.
    # Make sure that all sklearn functions you use that require a seed have this
    # seed specified in the argument list, namely: `random_state=self.seed` if
    # you are inside the solution class.

    # Read the MNIST dataset
    answer1_a = part_1a()
    answer1_b = part_1b()

    #x_train = answer1_b["x_train"]
    #y_train = answer1_b["y_train"]
    #x_test = answer1_b["x_test"]
    #y_test = answer1_b["y_test"]

    # Restrict the size of the dataset for debugging purposes.
    # The full dataset must be used for the final submission.
    # The dataset should only 7 and 9s.
    """
    n_train = 1000
    n_test = 200
    x_train = x_train[:n_train, :]
    y_train = y_train[:n_train]
    x_test = x_test[n_train : n_train + n_test, :]
    y_test = y_test[n_train : n_train + n_test]
    """
    print("--------------------------------")
    print("after answer_1a, before answer_1b")
    print("x_train.shape: ", x_train.shape)
    print("x_test.shape: ", x_test.shape)
    print("y_train.shape: ", y_train.shape)
    print("y_test.shape: ", y_test.shape)

    # x and Y are Mnist datasets
    answer1_c = part_1c()
    answer1_d = part_1d()
    answer1_e = part_1e()
    answer1_f = part_1f()
    print("part_1g, x_test.shape: ", x_test.shape)
    answer1_g = part_1g()

    print(f"{list(answer1_a.keys())=}")
    print(f"{list(answer1_b.keys())=}")
    print(f"{list(answer1_c.keys())=}")
    print(f"{list(answer1_d.keys())=}")
    print(f"{list(answer1_e.keys())=}")
    print(f"{list(answer1_f.keys())=}")
    print(f"{list(answer1_g.keys())=}")

    #del answer1_b[x_train]
    #del answer1_b[y_train]
    #del answer1_b[x_test]
    #del answer1_b[y_test]

    answer = {}
    answer["1a"] = answer1_a
    answer["1b"] = answer1_b
    answer["1c"] = answer1_c
    answer["1d"] = answer1_d
    answer["1e"] = answer1_e
    answer["1f"] = answer1_f
    answer["1g"] = answer1_g

    print("\n==> answer1_a")
    pprint(answer1_a)
    print("\n==> answer1_b")
    pprint(answer1_b)
    print("\n==> answer1_c")
    pprint(answer1_c)
    print("\n==> answer1_d")
    pprint(answer1_d)
    print("\n==> answer1_e")
    pprint(answer1_e)
    print("\n==> answer1_f")
    pprint(answer1_f)
    print("\n==> answer1_g")
    pprint(answer1_g)

    u.save_dict("section1.pkl", answer)
    """
    Run your code and produce all your results for your report. We will spot check the
    reports, and grade your code with automatic tools.
    """

    print("==>answers(part1a)")
    pprint(answer["1a"])
