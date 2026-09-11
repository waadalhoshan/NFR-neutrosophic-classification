"""
fuzzy_baseline.py — Fuzzy nearest-prototype classifier (Bezdek, 1981-style
FCM membership formula) used as the single-valued fuzzy baseline against
which the neutrosophic framework is compared.

For each category, a prototype vector is computed as the mean TF-IDF vector
of its positive training examples within each cross-validation fold. Fuzzy
membership degrees are then computed via the standard FCM update formula.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

from classifiers import DEFAULT_TFIDF_KWARGS


def fuzzy_membership(X_test: np.ndarray, centroids: np.ndarray, m: float = 2.0, eps: float = 1e-9):
    """
    Standard FCM membership formula:
      u_ic = 1 / sum_k( (d_ic / d_kc) ** (2/(m-1)) )

    Returns a membership matrix of shape (n_samples, n_categories).
    """
    n_samples = X_test.shape[0]
    n_cats = centroids.shape[0]
    dists = np.zeros((n_samples, n_cats))
    for c in range(n_cats):
        diff = X_test - centroids[c]
        dists[:, c] = np.sqrt((diff ** 2).sum(axis=1)) + eps

    power = 2.0 / (m - 1.0)
    membership = np.zeros((n_samples, n_cats))
    for i in range(n_samples):
        d = dists[i]
        ratio = d[:, None] / d[None, :]
        denom = (ratio ** power).sum(axis=1)
        membership[i] = 1.0 / denom
    return membership


def run_fuzzy_baseline(texts, Y, n_splits=5, random_state=42, tfidf_kwargs=None):
    tfidf_kwargs = tfidf_kwargs or DEFAULT_TFIDF_KWARGS
    vectorizer = TfidfVectorizer(**tfidf_kwargs)
    X = vectorizer.fit_transform(texts).toarray()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_samples, n_cats = Y.shape
    membership_all = np.zeros((n_samples, n_cats))

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]

        centroids = np.zeros((n_cats, X.shape[1]))
        for c in range(n_cats):
            pos_mask = Y_train[:, c] == 1
            centroids[c] = X_train[pos_mask].mean(axis=0) if pos_mask.sum() > 0 else X_train.mean(axis=0)

        membership_all[test_idx] = fuzzy_membership(X_test, centroids)

    return membership_all


if __name__ == "__main__":
    from data import load_nice_dataset

    df, texts, Y = load_nice_dataset()
    membership = run_fuzzy_baseline(texts, Y)
    np.save("results/membership_fuzzy.npy", membership)
    print("Saved membership_fuzzy.npy to results/")
