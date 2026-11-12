# Gaussian Naive Bayes classifier

import numpy as np


class GaussianNaiveBayes:

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    # ----------------------------------------------------------------- fit --
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]

        self.class_prior_ = np.zeros(n_classes)
        self.theta_ = np.zeros((n_classes, n_features))  # per-class feature means
        self.var_ = np.zeros((n_classes, n_features))  # per-class feature variances

        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_prior_[k] = len(Xc) / len(X)
            self.theta_[k] = Xc.mean(axis=0)
            self.var_[k] = Xc.var(axis=0)  # MLE variance (divides by n)

        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        self.var_ += self.epsilon_
        return self

    # ----------------------------------------------------------- inference --
    def _joint_log_likelihood(self, X):
        """log P(c) + log P(x | c) for every sample and class, shape (n, n_classes)."""
        X = np.asarray(X, dtype=float)
        jll = np.zeros((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            log_prior = np.log(self.class_prior_[k])
            log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[k]))
            log_exp = -0.5 * np.sum((X - self.theta_[k]) ** 2 / self.var_[k], axis=1)
            jll[:, k] = log_prior + log_norm + log_exp
        return jll

    def predict_log_proba(self, X):
        """log P(c | x), normalised with the log-sum-exp trick."""
        jll = self._joint_log_likelihood(X)
        m = jll.max(axis=1, keepdims=True)
        log_evidence = m + np.log(np.sum(np.exp(jll - m), axis=1, keepdims=True))
        return jll - log_evidence

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


# --------------------------------------------------------------------------- #
# Evaluation metrics (confusion matrix, accuracy, precision, recall, F1)
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def _safe_div(num, den):
    return num / den if den > 0 else 0.0


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true, y_pred):
    (_, fp), (_, tp) = confusion_matrix(y_true, y_pred)
    return _safe_div(tp, tp + fp)


def recall(y_true, y_pred):
    (_, _), (fn, tp) = confusion_matrix(y_true, y_pred)
    return _safe_div(tp, tp + fn)


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return _safe_div(2 * p * r, p + r)


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
