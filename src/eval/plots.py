import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay

def plot_roc(y_true, y_prob, ax=None, title="ROC"):
    disp = RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    if ax is None:
        plt.title(title); plt.show()
    return disp

def plot_pr(y_true, y_prob, ax=None, title="Precision-Recall"):
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    if ax is None:
        plt.title(title); plt.show()
    return disp

def plot_confusion(y_true, y_pred, ax=None, title="Confusion Matrix"):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    if ax is None:
        plt.title(title); plt.show()
    return disp
