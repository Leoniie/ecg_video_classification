from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import keras.backend as K
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score


def final_metric(y_true, y_pred):
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    return roc_auc


# confusion metric
def confusion_metric_vis(y_true, y_pred):
    y_actu = pd.Series(np.ravel(y_true), name='Actual')
    y_pred = pd.Series(np.ravel(y_pred), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)


# Keras metric f1
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
