"""Inspired by GPT4.

Information on type hints
https://peps.python.org/pep-0585/

GPT on testing functions, mock functions, testing number of calls, and argument values
https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

from sklearn.base import ClassifierMixin, RegressorMixin

# ==============================================================
Fill in the appropriate import statements from sklearn to solve the homework
from email.policy import default

IMPORTANT: do not communicate between functions in the class.
In other words: do not define intermediary variables using self.var = xxx
Doing so will make certain tests fail. Class methods should be independent
of each other and be able to execute in any order!

"""
import pickle
import warnings
import itertools
import numpy as np
from pprint import pprint
from typing import Any
from numpy.typing import NDArray
warnings.simplefilter(action="ignore", category=FutureWarning)
import new_utils as nu
import utils as u
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, top_k_accuracy_score

# These definitions guarantee that the variables are global and exist even if
# the code is not run interactively.
x_train = None
y_train = None
x_test = None
y_test = None
ntrain = None
ntest = None
ntrain_list = None

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.

normalize = u.Normalization.APPLY_NORMALIZATION
seed = 42
frac_train = 0.8


def part_2a() -> dict[str, Any]:
    """Prepare the dataset for training and testing.

    This method performs the following steps:
    1. Loads the training and testing data using the `prepare_data` function.
    2. Scales the training and testing data using the `scale_data` function.
    3. Counts the number of elements in each class for both training and
        testing datasets.
    4. Asserts that there are 10 classes in both datasets.
    5. Prints the number of classes in the training and testing datasets.
    6. Returns a dictionary containing the lengths and maximum values of the training
        and testing datasets, along with the scaled training and testing data.

    Returns
    -------
    tuple
        A tuple containing:
        - A dictionary with the lengths and maximum values of the training and testing
            datasets.
        - Scaled training data (Xtrain).
        - Scaled training labels (ytrain).
        - Scaled testing data (Xtest).
        - Scaled testing labels (ytest).

    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
    all classes by also printing out the number of elements in each class y and
    print out the number of classes for both training and testing datasets.

    """
    global x_train, y_train, x_test, y_test  # noqa: PLW0603

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.prepare_data()

    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    answers = {}
    # Ansewr type is a dict[str, int]. The values are the number of classes in
    # the training and testing data sets. The keys are "nb_classes_train" and
    # "nb_classes_test".

     # Load dataset
    x_train, y_train, x_test, y_test = u.prepare_data()

    # Scale pixel values to range [0,1]
    x_train = x_train.astype(np.float32) 
    x_test = x_test.astype(np.float32) 

    # Ensure labels are integers
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    unique_classes_train, counts_train = np.unique(y_train, return_counts=True)
    unique_classes_test, counts_test = np.unique(y_test, return_counts=True)
    assert len(unique_classes_train) == 10
    assert len(unique_classes_test) == 10

    answers["nb_classes"] = {
        "nb_classes_train": len(unique_classes_train),
        "nb_classes_test": len(unique_classes_test),
    }

    # Answer type is a dict[str, NDArray[np.int32]]. The value is a numpy array of the
    # number of samples in each class. The keys are "class_count_train" and
    # "class_count_test".
    answers["class_count"] = {
        "class_count_train": counts_train,
        "class_count_test": counts_test,
    }

    # The answer type is a dict[str, int]. The values are the number of samples in the
    # training and testing data sets. The keys are "nb_samples_x_train",
    # "nb_samples_x_test", "nb_samples_y_train", and "nb_samples_y_test".
    answers["nb_samples_data"] = {
        "nb_samples_x_train": x_train.shape[0],
        "nb_samples_x_test": x_test.shape[0],
        "nb_samples_y_train": y_train.shape[0],
        "nb_samples_y_test": y_test.shape[0],
    }

    # The answer type is a dict[str, int]. The values are the maximum value in the
    # training and testing data sets. The keys are "max_x_train" and "max_x_test".
    answers["max_data"] = {
        "max_x_train": float(np.max(x_train)),
        "max_x_test": float(np.max(x_test)),
    }

    return answers


def part_2b(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
    ntrain_list_: list[int] | None = None,
) -> dict[Any, Any]:
    """Perform multiple experiments on the dataset using logistic regression.

    This method executes the following steps:
    1. Prepares the training and testing datasets based on the provided sizes.
    2. Repeats parts C, D, and F of the previous analysis for each subset of the data.
    3. Uses logistic regression to train and evaluate the model (part F only).
    4. Collects and returns the results, including training and testing scores,
        confusion matrices, and class counts.

    Parameters
    ----------
    x_train_ : NDArray[np.floating]
        The feature matrix for training.
    y_train_ : NDArray[np.int32]
        The labels for training.
    x_test_ : NDArray[np.floating]
        The feature matrix for testing.
    y_test_ : NDArray[np.int32]
        The labels for testing.
    ntrain_list_ : list[int], optional
        A list of training sizes to evaluate, by default an empty list.

    Returns
    -------
    dict
        A dictionary containing results for each training size, including
            scores and confusion matrices.

    Task
    ----
    B. Repeat part 1.C, 1.D, and 1.F, for the multiclass problem.
    Use the Logistic Regression for part F with 500 iterations.
    Explain how multi-class logistic regression works (inherent,  # ! TODO
    one-vs-one, one-vs-the-rest, etc.).
    Use the entire MNIST training set to train the model.
    Comment on the results. Is the accuracy higher for the training or testing set?

    Notes
    -----
    Use try/except clauses to handle any errors and to prevent the code from failing.
    For example:

    def sumb(a: float, b: float) -> float:
        try:
            ret = a + b / 0.0
        except ZeroDivisionError:
            print("ZeroDivisionError")
            return 0.
        return ret

    `sumb(3., 4.)` will return 0. because of the exception, but will not fail.
    Therefore, you can submit the code to Gradescope without Gradescope failing.

    For the final classifier you trained in 2.B (partF),
    plot a confusion matrix for the test predictions.
    Earlier we stated that 7s and 9s were a challenging pair to
    distinguish. Do your results support this statement? Why or why not?

    """
    global ntrain_list, x_train, y_train, x_test, y_test  # noqa: PLW0603

    if ntrain_list_ is not None:
        ntrain_list = ntrain_list_
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    clf_c = DecisionTreeClassifier(random_state=42)
    cv_c = KFold(n_splits=5, random_state=42, shuffle=True)
    results_c = cross_validate(clf_c, x_train, y_train, cv=cv_c, scoring="accuracy", return_train_score=True)
    
    clf_d = DecisionTreeClassifier(random_state=42)
    cv_d = ShuffleSplit(n_splits=5, test_size=None, random_state=42)
    results_d = cross_validate(clf_d, x_train, y_train, cv=cv_d, scoring="accuracy", return_train_score=True)
                               
    train_sizes = [1000, 5000, 10000]
    test_sizes = [200, 1000, 2000] 
    for ntrain in train_sizes:
        for ntest in test_sizes:
        # Select subsets of training and testing data
            x_train_subset, y_train_subset = x_train[:ntrain], y_train[:ntrain]
            x_test_subset, y_test_subset = x_test[:ntest], y_test[:ntest]
            clf_f = LogisticRegression(max_iter=500, random_state=42, tol=0.001)
            cv_f = ShuffleSplit(n_splits=5, test_size=None, random_state=42)
            results_f = cross_validate(clf_f, x_train_subset, y_train_subset, cv=cv_f, scoring="accuracy", return_train_score=True)
            clf_f.fit(x_train_subset, y_train_subset)
            mean_cv_accuracy_1f = cross_val_score(clf_f, x_train, y_train, cv=cv_f, scoring="accuracy").mean()

            param_grid = {
                "fit_intercept": [True, False], 
                "C": [0.5, 1.0, 1.5],  
                'penalty': ['elasticnet', 'l1', 'l2']
                }
            grid_search = GridSearchCV(clf_f, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(x_train_subset, y_train_subset)
            best_clf_f = grid_search.best_estimator_
            required_params = ['fit_intercept', 'C', 'penalty']
            selected_params = {param: clf_f.get_params().get(param, None) for param in required_params}
            selected_params["penalty"] = [selected_params["penalty"]]

            y_train_pred_orig = clf_f.predict(x_train_subset)
            y_test_pred_orig = clf_f.predict(x_test_subset)
            y_train_pred_best = best_clf_f.predict(x_train_subset)
            y_test_pred_best = best_clf_f.predict(x_test_subset) 

            class_count_train = np.bincount(y_train_subset).astype(np.int32)
            class_count_test = np.bincount(y_test_subset).astype(np.int32)                        
    
    answers = {}

    answers: dict[str, Any] = {}

    # Your were asked to repeat parts C, D, and F of the previous analysis.
    # Therefore you know the answer types expected.
    answers["scores_1c"] = { 
        "mean_accuracy": float(results_c['test_score'].mean()),
        "std_accuracy": float(results_c['test_score'].std()),
        "mean_fit_time": float(results_c['fit_time'].mean()),
        "std_fit_time": float(results_c['fit_time'].std()),
        }
    answers["clf_1c"] = clf_c
    answers["cv_1c"] = cv_c

    answers["scores_1d"] = {
        "mean_accuracy": float(results_d['test_score'].mean()),
        "std_accuracy": float(results_d['test_score'].std()),
        "mean_fit_time": float(results_d['fit_time'].mean()),
        "std_fit_time": float(results_d['fit_time'].std()),
    }
    answers["clf_1d"] = clf_d
    answers["cv_1d"] = cv_d

    answers["accuracy_train_1f"] = float(accuracy_score(y_train_subset, y_train_pred_orig))
    answers["accuracy_test_1f"] =  float(accuracy_score(y_test_subset, y_test_pred_orig))
    answers["mean_cv_accuracy_1f"] = mean_cv_accuracy_1f
    answers["clf_1f"] = clf_f
    answers["cv_1f"] = cv_f
    
    # Answer is a dict[str, NDArray]. The keys are "conf_mat_train_1f", "conf_mat_test_1f"
    # Pay attention to the instructions regarding which classifier and cross-validation
    # technique to use.
    answers["confusion_matrix_1f"] = {
    "conf_mat_train_1f": confusion_matrix(y_train_subset, y_train_pred_orig),
    "conf_mat_test_1f": confusion_matrix(y_test_subset, y_test_pred_orig)
    }

    # The answer type is a dict[str, NDArray[np.int32]]. The value is a numpy array of
    # the confusion matrix. The keys are "confusion_matrix_train_orig",
    # "confusion_matrix_train_best", "confusion_matrix_test_orig", and
    # "confusion_matrix_test_best".
    answers["confusion_matrix_1g"] = {
    "confusion_matrix_train_orig": confusion_matrix(y_train_subset, y_train_pred_orig).astype(np.int32),
    "confusion_matrix_train_best": confusion_matrix(y_train_subset, y_train_pred_best).astype(np.int32),
    "confusion_matrix_test_orig": confusion_matrix(y_test_subset, y_test_pred_orig).astype(np.int32),
    "confusion_matrix_test_best": confusion_matrix(y_test_subset, y_test_pred_best).astype(np.int32)
    }

    # The answer type is a dict[str, float]. The value is the accuracy of the classifier.
    # The keys are "accuracy_orig_full_training", "accuracy_best_full_training",
    # "accuracy_orig_full_testing", and "accuracy_best_full_testing".
    answers["accuracy_1g"] = {
    "accuracy_orig_full_training": float(accuracy_score(y_train_subset, y_train_pred_orig)),
    "accuracy_best_full_training": float(accuracy_score(y_train_subset, y_train_pred_best)),
    "accuracy_orig_full_testing": float(accuracy_score(y_test_subset, y_test_pred_orig)),
    "accuracy_best_full_testing": float(accuracy_score(y_test_subset, y_test_pred_best))
    }

    # The answer type is a dict[str, float]. The value is the precision of the classifier.
    # The keys are "precision_orig_full_training", "precision_best_full_training",
    # "precision_orig_full_testing", and "precision_best_full_testing".
    answers["precision_1g"] = {
    "precision_orig_full_training": float(precision_score(y_train_subset, y_train_pred_orig, average="macro")),
    "precision_best_full_training": float(precision_score(y_train_subset, y_train_pred_best, average="macro")),
    "precision_orig_full_testing": float(precision_score(y_test_subset, y_test_pred_orig, average="macro")),
    "precision_best_full_testing": float(precision_score(y_test_subset, y_test_pred_best, average="macro"))
    }

    # The value is a classifier instance
    answers["clf_1g"] = clf_f

    # The value are the default parameters of the classifier, prior to the grid search.
    # There is a way to access this from the grid_search object. Figure it out.
    answers["default_parameters_1g"] = selected_params

    # The value is the best estimator (classifer) after completion of the grid search.
    answers["best_estimator_1g"] = best_clf_f

    # The value is the grid search instance
    answers["grid_search_1g"] = grid_search

    # The answer type is a dict[str, NDArray[np.int32]]. The value is a numpy array of
    # the # number of samples in each class. The keys are "class_count_train" and
    # "class_count_test".
    answers["class_count_1g"] = {
    "class_count_train": class_count_train,
    "class_count_test": class_count_test
    }

    conf_matrix = answers["confusion_matrix_1g"]["confusion_matrix_test_best"]
    confusion_pairs = []
    
    for i, j in itertools.combinations(range(10), 2):
        if i != j:
            confusion_count = conf_matrix[i, j] + conf_matrix[j, i]  # Sum misclassifications both ways
            confusion_pairs.append(((i, j), confusion_count))
    confusion_pairs.sort(key=lambda x: x[1], reverse=True)

    # Select the top 5 hardest-to-distinguish digit pairs
    # The answer type is a set of tuples. Each tuple contains two integers (lowest
    # integer is listed first). Identify five pairs of digits 0-9 that are hardeset
    # to distinguish. Calculate this from the confusion matrix.
    
    answers["hard_to_distinguish_pairs"] = set(pair for pair, _ in confusion_pairs[:5]) 

    return answers


# ----------------------------------------------------------------
if __name__ == "__main__":
    """
    Run your code and produce all your results for your report. We will spot check the
    reports, and grade your code with automatic tools.
    """

    # ------------------------------------------------------------

    # Restrict the size for faster training
    ntrain_list = [1000, 5000, 10000, 20000]  # max is 60000

    x_train, y_train, x_test, y_test = u.prepare_data()

    print("before part_a")
    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")
    # ------------------------------------------------------------

    all_answers = {}
    all_answers["part_2a"] = part_2a()

    print()
    print("before part_2b")
    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")
    all_answers["part_2b"] = part_2b()
    print("after part_2b")
    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")

    # print()
    # print("before part_2c")
    # print(f"{x_train.shape=}, {y_train.shape=}")
    # print(f"{x_test.shape=}, {y_test.shape=}")
    # all_answers["part_2c"] = part_2c()

    # print("after part_2c")
    # print(f"{x_train.shape=}, {y_train.shape=}")
    # print(f"{x_test.shape=}, {y_test.shape=}")

    u.save_dict("section2.pkl", dct=all_answers)
