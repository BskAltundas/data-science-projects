"""Part 3 of the assignment."""
# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

import pickle
import numpy as np
import itertools
import warnings
from pprint import pprint
from typing import Any
from numpy.typing import NDArray
warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import new_utils as nu
import utils as u
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, top_k_accuracy_score, make_scorer, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# The following global definitions guarantee that the variables are
#   global and exist even if the code is not run interactively.

x_train = None
y_train = None
x_test = None
y_test = None
ntrain = None
ntest = None

"""
In the first two set of tasks, we will narrowly focus on accuracy -
what fraction of our predictions were correct. However, there are several
popular evaluation metrics. You will learn how (and when) to use evaluation metrics.
"""

normalize = u.Normalization.APPLY_NORMALIZATION
seed = 42
frac_train = 0.8
n_splits = 5

def analyze_class_distribution(
    y: NDArray[np.int32],
) -> dict[str, Any]:
    """Analyzes and prints the class distribution in the dataset.

    Parameters
    ----------
    y : NDArray[np.int32]
        The labels dataset to analyze for class distribution.

    Returns
    -------
    - dict: A dictionary containing the count of elements in each class and the total
        number of classes.

    """
    # Your code here to analyze class distribution
    # Hint: Consider using collections.Counter or numpy.unique for counting

    uniq, counts = np.unique(y, return_counts=True)
    class_counts: dict[np.int32, np.int32] = dict(zip(uniq, counts, strict=True))
    num_classes = len(class_counts)
    print(f"{uniq=}")
    print(f"{counts=}")
    print(f"{class_counts=}")
    print(f"{num_classes=}")
    print(f"{np.sum(counts)=}")

    return {
        "class_counts": class_counts,  # Replace with actual class counts
        "num_classes": num_classes,  # Replace with the actual number of classes
    }


# --------------------------------------------------------------------------


def part_3a(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Check the integrity of the labels and analyze the class distribution.

    Parameters
    ----------
    x : NDArray[np.floating]
        The feature matrix for the dataset.
    y : NDArray[np.int32]
        The labels dataset to be checked and analyzed.
    x_test : NDArray[np.int32]
        The feature matrix for the test dataset.
    y_test : NDArray[np.int32]
        The labels for the test dataset.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the analyzed class distribution and integrity
            check results.

    Task
    ----
    A. Using the same classifier and hyperparameters as the one used at the end
        of part 2.B, get the accuracies of the training/test set scores using
        the top_k_accuracy score for k=1,2,3,4,5.
    Make a plot of k vs. score and comment on the rate of accuracy change.
    Do you think this metric is useful for this dataset?
    Use all 10 classes for this part of the problem.

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
    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================
    
    x_train, y_train, x_test, y_test = u.prepare_data()
    k_values = [1, 2, 3, 4, 5]
    train_accuracies = []
    test_accuracies = []
    
    param_grid = {
                "fit_intercept": [True, False], 
                "C": [0.5, 1.0, 1.5],  
                'penalty': ['elasticnet', 'l1', 'l2']
                }
    grid_search = GridSearchCV(LogisticRegression(max_iter=500, random_state=42, tol=0.001), param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    
    clf_f = grid_search.best_estimator_
    clf_f.fit(x_train, y_train)
    
    y_train_proba = clf_f.predict_proba(x_train)
    y_test_proba = clf_f.predict_proba(x_test)
    
    answers = {}

    # Answer type: dict[int, list[float, float]]
    # The two floats are the train and test accuracy for the top-k accuracy score
    # Remember: the order of the list elements matters
    answers["top_k_accuracy"] = {
        k: [
            float(top_k_accuracy_score(y_train, y_train_proba, k=k)),
            float(top_k_accuracy_score(y_test, y_test_proba, k=k))
        ]
        for k in k_values
    }
    
    train_accuracies = [float(top_k_accuracy_score(y_train, y_train_proba, k=k)) for k in k_values]
    test_accuracies = [float(top_k_accuracy_score(y_test, y_test_proba, k=k)) for k in k_values]

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, train_accuracies, marker='o', linestyle='-', linewidth=2, markersize=6, label='Train Accuracy')
    plt.plot(k_values, test_accuracies, marker='s', linestyle='--', linewidth=2, markersize=6, label='Test Accuracy')
    plt.xlabel('Top-k')
    plt.ylabel('Accuracy')
    plt.title('Top-k Accuracy Score vs. k')
    plt.xticks(k_values)
    plt.ylim(0.90, 1) 
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    return answers

# --------------------------------------------------------------------------


# How to make sure the seed propagates. Perhaps specify in the class constructor.
def part_3b(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Prepare an imbalanced dataset; convert 7s to 0s and 9s to 1s.

    This function filters the input dataset to retain only the classes 7 and 9,
    removes a specified fraction of the 9s, and convert the labels accordingly.
    It also prepares the test dataset in the same manner.

    (other using 10 classes)?

    Parameters
    ----------
    x_train_ : NDArray[np.floating]
        The feature matrix for the training dataset.
    y_train_ : NDArray[np.int32]
        The labels for the training dataset.
    x_test_ : NDArray[np.floating]
        The feature matrix for the test dataset.
    y_test : NDArray[np.int32]
        The labels for the test dataset.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the processed training and test datasets.

    Task
    ----
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s
        removed.  Also convert the 7s to 0s and 9s to 1s.

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

    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    # Filter dataset to keep only 7s and 9s
    train_mask = (y_train == 7) | (y_train == 9)
    test_mask = (y_test == 7) | (y_test == 9)
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Convert labels: 7 -> 0, 9 -> 1
    y_train = np.where(y_train == 7, 0, 1)
    y_test = np.where(y_test == 7, 0, 1)

    # Remove 90% of 9s (now labeled as 1s) in the training set
    ones_train_mask = y_train == 1
    ones_indices = np.where(ones_train_mask)[0]
    num_ones_to_keep = int(len(ones_indices) * 0.1)
    keep_indices = np.random.choice(ones_indices, size=num_ones_to_keep, replace=False)
    keep_mask = np.full(y_train.shape, False)
    keep_mask[keep_indices] = True
    keep_mask |= ~ones_train_mask  # Keep all 0s
    x_train, y_train = x_train[keep_mask], y_train[keep_mask]
    answers: dict[str, Any] = {}

    dct1: dict[str, int] = {}
    dct1["length_x_train"] = len(x_train)
    dct1["length_x_test"] = len(x_test)
    dct1["length_y_train"] = len(y_train)
    dct1["length_y_test"] = len(y_test)
    # The type is a dict[str, int]
    # The keys are "length_x_train", "length_x_test", "length_y_train", and "length_y_test".
    # The values are the number of samples in the training and test sets.
    answers["number_of_samples"] = {
        "length_x_train": len(x_train),
        "length_x_test": len(x_test),
        "length_y_train": len(y_train),
        "length_y_test": len(y_test),
    }

    # The type is a dict[str, float]
    # The keys are "max_x_train" and "max_x_test".
    # The values are floats.
    answers["data_bounds"] = {
        "max_x_train": float(np.max(x_train)),
        "max_x_test": float(np.max(x_test)),
    }

    # The type is a dict[str, int].
    # The keys are "num_0s_train", "num_1s_train", "num_0s_test", and "num_1s_test".
    # The values are the number of samples of each class (0/1) in the training
    # and test sets.
    answers["class_counts"] = {
        "num_0s_train": int(np.sum(y_train == 0)),
        "num_1s_train": int(np.sum(y_train == 1)),
        "num_0s_test": int(np.sum(y_test == 0)),
        "num_1s_test": int(np.sum(y_test == 1)),
    }

    # Both the test set and the training set are imbalanced.

    return answers


# --------------------------------------------------------------------------


def part_3c(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Repeat part 1.C for this dataset with a support vector machine (SVC in sklearn).

    This function implements a support vector machine classifier using a
        stratified cross-validation strategy.
    It evaluates the model's performance by calculating the mean and standard deviation
        of various metrics, including accuracy, F1 score, precision, and recall. The
        function also determines whether precision or recall is higher and provides an
        explanation for the observed results. Finally, the classifier is trained on the
        entire training dataset, and the confusion matrix is generated to assess the
        classification performance.

    Parameters
    ----------
    x_train_ : NDArray[np.floating]
        The feature matrix for the training dataset.
    y_train_ : NDArray[np.int32]
        The labels for the training dataset.
    x_test_ : NDArray[np.floating]
        The feature matrix for the test dataset.
    y_test_ : NDArray[np.int32]
        The labels for the test dataset.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the following keys:
        - "cv": The cross-validation object used for stratified k-fold.
        - "clf": The trained SVC classifier.
        - "mean_F1": The mean F1 score from cross-validation.
        - "mean_recall": The mean recall from cross-validation.
        - "mean_accuracy": The mean accuracy from cross-validation.
        - "mean_precision": The mean precision from cross-validation.
        - "std_F1": The standard deviation of the F1 score from cross-validation.
        - "std_recall": The standard deviation of the recall from cross-validation.
        - "std_accuracy": The standard deviation of the accuracy from cross-validation.
        - "std_precision": The standard deviation of the precision from
            cross-validation.
        - "is_precision_higher_than_recall": A boolean indicating if precision is
            higher than recall.
        - "is_precision_higher_than_recall_explain": A string explanation of the
            precision-recall comparison.
        - "confusion_matrix": The confusion matrix for the classifier's predictions
            on the test dataset.

    Task
    ----
    C. Repeat part 1.C for this dataset with a support vector machine (SVC in sklearn).
        Make sure to use a stratified cross-validation strategy. In addition to regular
        accuracy also print out the mean/std of the F1 score, precision, and recall.
        Is precision or recall higher? Explain. Finally, train the classifier on all
        the training data and plot the confusion matrix.

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

    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    clf = SVC(random_state=42)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    print("before scores", flush=True)
    scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="macro"),
    "recall": make_scorer(recall_score, average="macro"),
    "f1": make_scorer(f1_score, average="macro")
    }
    scores = cross_validate(
        clf,
        x_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,  # What does this do?
    )

    print("after scores", flush=True)

    print("scores= ", scores, flush=True)

    # Train on all the data
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    clf.fit(x, y)
    y_pred = cross_val_predict(clf, x, y, cv=2)
    conf_mat_c = confusion_matrix(y, y_pred, labels= [0, 1])
    print(f"{conf_mat_c=}")

    mean_metrics = {
    "mean_F1": float(np.nan_to_num(scores["test_f1"].mean())),
    "mean_recall": float(np.nan_to_num(scores["test_recall"].mean())),
    "mean_accuracy": float(np.nan_to_num(scores["test_accuracy"].mean())),
    "mean_precision": float(np.nan_to_num(scores["test_precision"].mean()))
    }

    std_metrics = {
    "std_F1": float(np.nan_to_num(scores["test_f1"].std())),
    "std_recall": float(np.nan_to_num(scores["test_recall"].std())),
    "std_accuracy": float(np.nan_to_num(scores["test_accuracy"].std())),
    "std_precision": float(np.nan_to_num(scores["test_precision"].std()))
    }

    answers = {}

    # The value is a StratifiedKFold object
    answers["cv"] = cv

    # The value is a SVC classifier
    answers["clf"] = clf

    # Cross-validation scores
    # The value is a dict[str, float]
    # The keys are "mean_F1", "mean_recall", "mean_accuracy", and "mean_precision".
    # The values are the mean of the F1 score, recall, accuracy, and precision from
    # cross-validation.
    answers["mean_metrics"] = mean_metrics
    # The value is a dict[str, float].
    # The keys are "std_F1", "std_recall", "std_accuracy", and "std_precision".
    # The values are the standard deviation of the F1 score, recall, accuracy, and
    # precision from cross-validation.
    answers["std_metrics"] = std_metrics

    # Type: bool
    # The value is True if precision is higher than recall, False otherwise.
    answers["is_precision_higher_than_recall"] = mean_metrics["mean_precision"] >mean_metrics["mean_recall"]

    # The value is a string the explains why precision is higher or lower than recall.
    answers["is_precision_higher_than_recall_explain"] = "Precision is higher than recall, indicating the model is more conservative in making positive predictions. This could suggest that false positives are more costly in this scenario, leading the classifier to minimize them. if is_precision_higher else Recall is higher than precision, suggesting that the model is more lenient in classifying positives, potentially minimizing false negatives."


    # The value is a confusion matrix.
    answers["confusion_matrix"] = conf_mat_c

    return answers


# --------------------------------------------------------------------------


def part_3d(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Train and evaluate a Support Vector Classifier (SVC) with class weights.

    Parameters
    ----------
    x_train : NDArray[np.floating]
        The feature matrix for the training data.
    y_train : NDArray[np.int32]
        The labels corresponding to the training data.
    x_test : NDArray[np.floating]
        The feature matrix for the testing data.
    y_test : NDArray[np.int32]
        The labels corresponding to the testing data.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - cv: The cross-validation object used.
        - clf: The trained classifier.
        - mean_F1: The mean F1 score from cross-validation.
        - mean_recall: The mean recall from cross-validation.
        - mean_accuracy: The mean accuracy from cross-validation.
        - mean_precision: The mean precision from cross-validation.
        - std_F1: The standard deviation of the F1 score.
        - std_recall: The standard deviation of the recall.
        - std_accuracy: The standard deviation of the accuracy.
        - std_precision: The standard deviation of the precision.
        - is_precision_higher_than_recall: Boolean indicating if precision is higher
            than recall.
        - is_precision_higher_than_recall_explain: Explanation of the precision vs
            recall comparison.
        - performance_difference_explain: Explanation of the performance difference
            due to class weights.
        - confusion_matrix: The confusion matrix for the predictions on the test set.
        - weight_dict: A dictionary of class weights used in training.

    Task
    ----
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the
        class_weights parameter).  Print out the class weights, and comment on the
        performance difference. Use compute_class_weight to compute the class weights.

    """
    global x_train, y_train, x_test, y_test  # noqa: PLW0603

    print("----------------------------------------------------------------")
    print("Enter part_3d", flush=True)

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
    # Keep two classes and remove 90% of one of the two classes
    
    binary_classes = np.array([0, 1]) 
    class_weights = compute_class_weight(class_weight="balanced", classes=binary_classes, y=y_train)
    weight_dict = dict(zip(binary_classes, class_weights))
    
    clf = SVC(random_state=seed, class_weight=weight_dict)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="macro"),
    "recall": make_scorer(recall_score, average="macro"),
    "f1": make_scorer(f1_score, average="macro")
    }

    # Perform cross-validation
    scores = cross_validate(
        clf, x_train, y_train, cv=cv,
        scoring=scoring,
        return_train_score=True
    )
    
    mean_metrics = {
    "mean_F1": float(np.nan_to_num(scores["test_f1"].mean())),
    "mean_recall": float(np.nan_to_num(scores["test_recall"].mean())),
    "mean_accuracy": float(np.nan_to_num(scores["test_accuracy"].mean())),
    "mean_precision": float(np.nan_to_num(scores["test_precision"].mean()))
    }
    
    std_metrics = {
    "std_F1": float(np.nan_to_num(scores["test_f1"].std())),
    "std_recall": float(np.nan_to_num(scores["test_recall"].std())),
    "std_accuracy": float(np.nan_to_num(scores["test_accuracy"].std())),
    "std_precision": float(np.nan_to_num(scores["test_precision"].std()))
    }
    
    # Train on the full dataset
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    clf.fit(x, y)
    y_pred = cross_val_predict(clf, x, y, cv=5)
    conf_mat_d = confusion_matrix(y, y_pred, labels=[0, 1])
    
    answers = {}

    # The value is a StratifiedKFold object
    answers["cv"] = cv

    # The value is a SVC classifier
    answers["clf"] = clf

    # The value is a dict[str, float].
    # The keys are "mean_F1", "mean_recall", "mean_accuracy", and "mean_precision".
    # The values are the mean of the F1 score, recall, accuracy, and precision from
    # cross-validation.
    answers["mean_metrics"] = mean_metrics

    # The value is a dict[str, float].
    # The keys are "std_F1", "std_recall", "std_accuracy", and "std_precision".
    # The values are the standard deviation of the F1 score, recall, accuracy, and
    # precision from cross-validation.
    answers["std_metrics"] = std_metrics

    # Type: bool
    answers["is_precision_higher_than_recall"] = mean_metrics["mean_precision"] > mean_metrics["mean_recall"]
    answers["is_precision_higher_than_recall_explain"] = "Precision is higher than recall, indicating the model is more conservative in making positive predictions.This could suggest that false positives are more costly in this scenario, leading the classifier to minimize them. if is_precision_higher else Recall is higher than precision, suggesting that the model is more lenient in classifying positives, potentially minimizing false negatives."

    # Type: 2x2 NDArray (np.array)
    answers["confusion_matrix"] = conf_mat_d

    # Type: dict[str, float]
    answers["weight_dict"] = weight_dict

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run your code and produce all your results for your report. We will spot check the
    reports, and grade your code with automatic tools.
    """

    ################################################
    # In real code, read MNIST files and define Xtrain and xtest appropriately
    rng = np.random.default_rng(seed)
    n_images = 1200
    n_features = 784
    x = rng.random((n_images, n_features))  # 100 samples, 100 features
    # Fill labels with 0 and 1 (mimic 7 and 9s)
    y = (x[:, :5].sum(axis=1) > 2.5).astype(int)
    n_train = 100
    x_train = x[0:n_train, :]
    x_test = x[n_train:, :]
    y_train = y[0:n_train]
    y_test = y[n_train:]

    # synthetic data with classes 0/1.
    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.create_data(
        n_rows=1200,
        n_features=784,
        frac_train=0.8,
    )

    # Replace 0s with 7s and 1s with 9s
    y_train[y_train == 0] = 7
    y_train[y_train == 1] = 9
    y_test[y_test == 0] = 7
    y_test[y_test == 1] = 9

    x = x_train
    y = y_train

    x_train, y_train, x_test, y_test = u.prepare_data()

    print("\nbefore part_3a")
    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")
    print(f"{x.shape=}, {y.shape=}")

    all_answers = {}
    all_answers["part_3a"] = part_3a()
    print()
    print(f"after part_3a, {x_train.shape=}, {y_train.shape=}")
    print(f"after part_3a, {x_test.shape=}, {y_test.shape=}")
    print(f"after part_3a, {x.shape=}, {y.shape=}")

    # The data is the full MNIST dataset
    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.prepare_data()

    print("before part_3b")
    print(f"before part_3b, {x_train.shape=}, {y_train.shape=}")
    print(f"before part_3b, {x_test.shape=}, {y_test.shape=}")
    print("============================================")
    all_answers["part_3b"] = part_3b()
    print()
    print(f"after part_3b, {x_train.shape=}, {y_train.shape=}")
    print(f"after part_3b, {x_test.shape=}, {y_test.shape=}")

    print("============================================")
    all_answers["part_3c"] = part_3c()
    print()
    print(f"after part_3c, {x_train.shape=}, {y_train.shape=}")
    print(f"after part_3c, {x_test.shape=}, {y_test.shape=}")
    print(f"after part_3c, {x.shape=}, {y.shape=}")
    quit()

    print("============================================")
    all_answers["part_3d"] = part_3d()
    print()
    print(f"after part_3d, {x_train.shape=}, {y_train.shape=}")
    print(f"after part_3d, {x_test.shape=}, {y_test.shape=}")
    print(f"after part_3d, {x.shape=}, {y.shape=}")

    u.save_dict("section3.pkl", dct=all_answers)
