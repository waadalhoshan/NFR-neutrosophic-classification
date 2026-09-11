"""
classifiers.py — Probabilistic multi-label classifiers used as the source of
truth (T) and falsity (F) components in the neutrosophic framework.

Both classifiers are wrapped in a One-vs-Rest scheme over TF-IDF features and
evaluated via K-fold cross-validation, with out-of-fold predicted
probabilities returned so that no requirement's neutrosophic representation
is influenced by its own presence in a training fold.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold

DEFAULT_TFIDF_KWARGS = dict(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
    max_features=3000,
    min_df=2,
)


def _cross_val_probabilities(build_classifier_fn, X, Y, n_splits=5, random_state=42):
    """Generic K-fold cross-validation loop returning out-of-fold probabilities."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_samples, n_labels = Y.shape
    probs = np.zeros((n_samples, n_labels))

    for train_idx, test_idx in kf.split(X):
        clf = build_classifier_fn()
        clf.fit(X[train_idx], Y[train_idx])
        probs[test_idx] = clf.predict_proba(X[test_idx])

    return probs


def get_svm_probabilities(texts, Y, n_splits=5, random_state=42, tfidf_kwargs=None):
    """TF-IDF + linear SVM (One-vs-Rest, Platt-scaled probabilities)."""
    tfidf_kwargs = tfidf_kwargs or DEFAULT_TFIDF_KWARGS
    vectorizer = TfidfVectorizer(**tfidf_kwargs)
    X = vectorizer.fit_transform(texts)

    build_fn = lambda: OneVsRestClassifier(
        SVC(kernel='linear', probability=True, random_state=random_state)
    )
    probs = _cross_val_probabilities(build_fn, X, Y, n_splits, random_state)
    return probs, vectorizer


def get_naive_bayes_probabilities(texts, Y, n_splits=5, random_state=42, tfidf_kwargs=None):
    """TF-IDF + Multinomial Naive Bayes (One-vs-Rest)."""
    tfidf_kwargs = tfidf_kwargs or DEFAULT_TFIDF_KWARGS
    vectorizer = TfidfVectorizer(**tfidf_kwargs)
    X = vectorizer.fit_transform(texts)

    build_fn = lambda: OneVsRestClassifier(MultinomialNB())
    probs = _cross_val_probabilities(build_fn, X, Y, n_splits, random_state)
    return probs, vectorizer


if __name__ == "__main__":
    from data import load_nice_dataset

    df, texts, Y = load_nice_dataset()

    probs_svm, _ = get_svm_probabilities(texts, Y)
    probs_nb, _ = get_naive_bayes_probabilities(texts, Y)

    np.save("results/probs_svm.npy", probs_svm)
    np.save("results/probs_nb.npy", probs_nb)
    np.save("results/Y.npy", Y)
    print("Saved probs_svm.npy, probs_nb.npy, and Y.npy to results/")
