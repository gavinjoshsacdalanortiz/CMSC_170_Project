#Evaluation metrics (confusion matrix, accuracy, precision, recall and F1)

#input checking
def _as_label(value):
    #return 0 or 1 for a label given as 0, 1, '0', '1' (or a numpy int)
    if isinstance(value, str):
        value = value.strip()
    if value in (0, 1, "0", "1"):
        return int(value)
    raise ValueError(f"labels must be 0 or 1, got {value!r}")

#convert both label lists to lists of 0/1 ints and check they line up
def _check_pair(y_true, y_pred):
    true = [_as_label(v) for v in y_true]
    pred = [_as_label(v) for v in y_pred]
    if len(true) != len(pred):
        raise ValueError(f"y_true has {len(true)} labels but y_pred has {len(pred)}")
    if len(true) == 0:
        raise ValueError("cannot score an empty list of predictions")
    return true, pred


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0

#confusion matrix (2x2)
def confusion_matrix(y_true, y_pred):
    true, pred = _check_pair(y_true, y_pred)
    tn = fp = fn = tp = 0
    for t, p in zip(true, pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:  #t == 1 and p == 0
            fn += 1
    return [[tn, fp], [fn, tp]]


def _unpack(cm):
    (tn, fp), (fn, tp) = cm
    return tn, fp, fn, tp



#Scores
#-----------------------------------------------------------------------------

#(TP + TN) / all samples
def accuracy(y_true, y_pred):
    tn, fp, fn, tp = _unpack(confusion_matrix(y_true, y_pred))
    return (tp + tn) / (tp + tn + fp + fn)

#TP / (TP + FP)
def precision(y_true, y_pred):
    """TP / (TP + FP): of the samples predicted as class 1, how many really are."""
    tn, fp, fn, tp = _unpack(confusion_matrix(y_true, y_pred))
    return _safe_div(tp, tp + fp)

#TP / (TP + FN)
def recall(y_true, y_pred):
    tn, fp, fn, tp = _unpack(confusion_matrix(y_true, y_pred))
    return _safe_div(tp, tp + fn)

#2PR / (P + R)
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return _safe_div(2 * p * r, p + r)

#all evaluation metrics
def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = _unpack(cm)
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return {
        "confusion_matrix": cm,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": p,
        "recall": r,
        "f1": _safe_div(2 * p * r, p + r),
    }



#helper function for printing confusion matrix
def format_confusion_matrix(cm):
    tn, fp, fn, tp = _unpack(cm)
    lines = [
        f"{'':<10}{'Predicted 0':>13}{'Predicted 1':>13}",
        f"{'Actual 0':<10}{tn:>13}{fp:>13}",
        f"{'Actual 1':<10}{fn:>13}{tp:>13}",
    ]
    return "\n".join(lines)
