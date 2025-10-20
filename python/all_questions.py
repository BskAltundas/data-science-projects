# Add import files
import pickle
from typing import Any

# Notes:
# - The calculations should be included in the answer
# - The answer type is specified in the question
# - Alert me to all issues.


# -----------------------------------------------------------
def question1() -> dict[str, Any]:
    """Answers to question 1."""
    answers: dict[str, Any] = {}

    # answer type: str ("Yes"/"No")
    answers["(a)"] = "No"

    # answer type: explanatory string (at least four words)
    answers["(a) explain"] = "The rules are not mutually exclusive because there are cases where multiple rules could apply to the same individual, leading to potential conflicts."

    # answer type: str ("Yes"/"No")
    answers["(b)"] = "No"

    # answer type: explanatory string (at least four words)
    answers["(b) explain"] = "The rule set is not exhaustive because there exist some combinations of attributes that do not fit into any rule."

    # answer type: str ("Yes"/"No")
    answers["(c)"] = "Yes"

    # answer type: explanatory string (at least four words)
    answers["(c) explain"] = "Ordering is needed because multiple rules apply to the same cases with conflicting outcomes."

    # answer type: str ("Yes"/"No")
    answers["(d)"] = "Yes"
    
    # answer type: explanatory string (at least four words)
    answers["(d) explain"] = "A default class is needed to handle cases that do not match any rule."

    return answers


# -----------------------------------------------------------
def question2() -> dict[str, Any]:
    """Answers to question 2."""
    answers: dict[str, Any] = {}

    # answer type: str ("Yes", "No")
    answers["(a)"] = None

    # answer type: explanatory string (at least four words)
    answers["(a) explain"] = None
    # answer type: str ("Yes", "No")
    answers["(b)"] = None

    # answer type: explanatory string (at least four words)
    answers["(b) explain"] = None

    # answer type: str ("Yes", "No")
    answers["(c)"] = None

    # answer type: explanatory string (at least four words)
    answers["(c) explain"] = None

    # answer type: str ("Yes", "No")
    answers["(d)"] = None

    # answer type: explanatory string (at least four words)
    answers["(d) explain"] = None

    return answers


# -----------------------------------------------------------
def question3() -> dict[str, Any]:
    """Answers to question 3."""
    answers: dict[str, Any] = {}

    # answer type: str ("Yes", "No")
    answers["(a)"] = "No"
    answers["(b)"] = "No"
    answers["(c)"] = "Yes"

    # answer type: explanatory string (at least four words)
    answers["(a) example"] = "The rules can lead to conflicting classifications, making them not mutually exclusive."
    answers["(b) example"] = "Since all animals in the dataset can be classified using the provided rules, the rule set is exhaustive"
    answers["(c) example"] = "Ordering is needed to prioritize more specific rules (R2) over general rules (R4) to prevent misclassifications."

    return answers


# -----------------------------------------------------------
def question4() -> dict[str, Any]:
    """Answers to question 4."""
    answers: dict[str, Any] = {}

    # If the requested boolean expression is true,
    # then the answer is 1, otherwise it is 0

    # answer type: int
    answers["(a)"] = 0
    answers["(b)"] = 0
    answers["(c)"] = 1
    answers["(d)"] = 0
    answers["(e)"] = 0
    answers["(f)"] = 0

    return answers


# -----------------------------------------------------------
def question5() -> dict[str, Any]:
    """Answers to question 5."""
    answers: dict[str, Any] = {}

    # answer type: string: ("Model 1", "Model 2")
    answers["(a)"] = "Model 2"

    # answer type: explanatory string (at least four words)
    answers["(a) explain"] = "Model 2 (Pruned Decision Tree) has lower training accuracy (82%) but higher test accuracy (80%), indicating that it generalizes better and is less reliant on the specific patterns in the training data."

    # answer type: string: ("Model 1", "Model 2")
    answers["(b)"] = "Model 2"

    # answer type: explanatory string (at least four words)
    answers["(b) explain"] = "Even though Model 1 has slightly higher accuracy (85% vs. 81%) on the full dataset, Model 2 is the better choice because it generalizes better and is less prone to overfitting, making it more reliable for future classification tasks."

    return answers


# -----------------------------------------------------------
# REMOVE?
def question6() -> dict[str, Any]:
    """Answers to question 6."""
    answers: dict[str, Any] = {}

    # The calculations should be included in the answer

    # answer type: float
    answers["(a) P(Buy|Ad)"] = 0.20
    answers["(a) P(Buy|No Ad)"] = 0.06
    answers["(a) P(No Buy|Ad)"] = 0.80
    answers["(a) P(No Buy|No Ad)"] = 0.94
    answers["(a) Probability of purchase at website"] = 0.088

    answers["(b) P(No Ad|Buy)/P(Ad|Buy)"] = 1.20

    # Answer type: string ("yes", "no")
    answers["(b) Is advertisement campaign successful?"] = "no"

    return answers


# -----------------------------------------------------------
def question7() -> dict[str, Any]:
    """Answers to question 7."""
    answers: dict[str, Any] = {}

    # answer type: bool (True, False)
    answers["(a)"] = False
    answers["(b)"] = True
    answers["(c)"] = False
    answers["(d)"] = True

    # answer type: explanatory string (at least four words)
    answers["(a) explain"] = "The gradients of weights at layer k+1 depend on the errors from layer k+2, not on the gradients of weights at layer k"
    answers["(b) explain"] = "When an ANN is applied to a test instance, it follows the same forward pass computation as in training, but without weight updates."
    answers["(c) explain"] = "The vanishing gradient problem occurs when the gradients of the loss function become extremely small during backpropagation, causing the weights in earlier layers to update very slowly or not at all."
    answers["(d) explain"] = "Since the loss is minimized and all training instances are correctly classified, the gradients of loss with respect to all weights at all layers will be 0"

    return answers


# -----------------------------------------------------------
def question8() -> dict[str, Any]:
    """Answers to question 8."""
    answers: dict[str, Any] = {}

    # Answer type: float
    answers["(c) P(X1=1 | +)"] = 0.8
    answers["(c) P(X1=1 | -)"] = 0.5
    answers["(c) P(X2=1 | +)"] = 0.5
    answers["(c) P(X2=1 | -)"] = 0.32
    answers["(c) P(X3=1 | +)"] = 0.40
    answers["(c) P(X3=1 | -)"] = 0.16
    answers["(a) Matrix 1"] = "True"

    # Answer type: float ????
    answers["(b) Conditional Independence"] = "Yes"

    # Answer type: float
    answers["(a) P(X1=1)"] = 0.65
    answers["(a) P(X2=1)"] = 0.41
    answers["(a) P(X1=1,X2=1)"] = 0.28

    # Answer type: str
    answers["(a) Relationship between X_1 and X_2"] = "Dependent"

    # Answer type: float
    answers["(d) P(A=1)"] = 0.5
    answers["(d) P(X1=1, X2=1|Class=+)"] = 0.4
    answers["(d) P(X1=1|Class=+)"] = 0.8
    answers["(d) P(X2=1|Class=+)"] = 0.5

    # Answer type: string ("yes", "no")
    answers["(d) A and B conditionally independent"] = "Yes"

    # Answer type: float
    answers["(d) Training error rate"] = 0.38

    return answers


# -----------------------------------------------------------
def question9() -> dict[str, Any]:
    """Answers to question 9."""
    answers: dict[str, Any] = {}

    # Answer type: int
    answers["(a) K"] = 1

    # Answer type: explanatory string
    answers["(a) explain"] = "In highly clustered regions, K=1 is the best choice because it ensures that the nearest neighbor is always from the same class, minimizing classification errors in dense areas."

    # Answer type: int
    answers["(b) K"] = 5

    # Answer type: explanatory string
    answers["(b) explain"] = "In the second image, where the classes are highly mixed, using K=5 helps smooth out noise and creates a more stable decision boundary, reducing misclassifications."

    return answers


# -----------------------------------------------------------
def question10() -> dict[str, Any]:
    """Answers to question 10."""
    answers: dict[str, Any] = {}

    # Answer type: float
    answers["(a) P(A=1|+)"] = 0.6
    answers["(a) P(B=1|+)"] = 0.4
    answers["(a) P(C=1|+)"] = 0.8
    answers["(a) P(A=1|-)"] = 0.4
    answers["(a) P(B=1|-)"] = 0.4
    answers["(a) P(C=1|-)"] = 0.2

    # Answer type: explanatory string
    answers["(a) P(A=1|+) explain your answer"] = "60% of positive class instances have A=1."

    # note: R is the sample (A=1,B=1,C=1)
    # Answer type: float
    answers["(b) P(+|R)"] = 0.096
    answers["(b) P(R|+)"] = 0.192
    answers["(b) P(R|-)"] = 0.032

    # Answer type: string, '+' or '-'
    answers["(b) class label"] = "+"

    # Answer type: explanatory string
    answers["(b) Explain your reasoning"] = "Since P(+|R) > P(-|R), the sample R is more likely to belong to the positive class."

    # Answer type: float
    answers["(c) P(A=1)"] = 0.5
    answers["(c) P(B=1)"] = 0.4
    answers["(c) P(A=1,B=1)"] = 0.2

    # Answer type: string, 'yes' or 'no'
    answers["(c) A independent of B?"] = "yes"

    # Answer type: float
    answers["(d) P(A=1)"] = 0.5
    answers["(d) P(B=0)"] = 0.6
    answers["(d) P(A=1,B=0)"] = 0.3

    # Answer type: string: 'yes' or 'no'
    answers["(d) A independent of B?"] = "yes"

    # Answer type: float
    answers["(e) P(A=1,B=1|+)"] = 0.2
    answers["(e) P(A=1|+)"] = 0.6
    answers["(e) P(B=1|+)"] = 0.4

    # Answer type: string: 'yes' or 'no'
    answers["(e) A independent of B given class +?"] = "no"

    # Answer type: explanatory string
    answers["(e) A and B conditionally independent given class +, explain"] = "Since the probabilities are not equal, Aand B are not independent given the class +."

    return answers


# --------------------------------------------------------
if __name__ == "__main__":
    answers_dict = {}
    answers_dict["question1"] = question1()
    answers_dict["question2"] = question2()
    answers_dict["question3"] = question3()
    answers_dict["question4"] = question4()
    answers_dict["question5"] = question5()
    answers_dict["question6"] = question6()
    answers_dict["question7"] = question7()
    answers_dict["question8"] = question8()
    answers_dict["question9"] = question9()
    answers_dict["question10"] = question10()
    print("end code")

    with open("answers.pkl", "wb") as f:
        pickle.dump(answers_dict, f)
