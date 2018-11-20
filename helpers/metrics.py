from sklearn.metrics import roc_auc_score


def final_metric(y_true, y_pred):
    roc_auc = roc_auc_score( y_true=y_true, y_score=y_pred )
    return roc_auc
