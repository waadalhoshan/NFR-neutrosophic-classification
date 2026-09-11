"""
neutrosophic.py — Core neutrosophic framework: derivation of truth (T),
indeterminacy (I), and falsity (F) from classifier probabilities, and the
weighted distance and entropy measures defined in the paper (Section 3).
"""
import numpy as np


def normalized_entropy(p_vec: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability-like vector, normalized to [0, 1]."""
    p = np.clip(p_vec, eps, None)
    p = p / p.sum()
    m = len(p)
    ent = -np.sum(p * np.log(p))
    return ent / np.log(m)


def derive_TIF(probs: np.ndarray):
    """
    Derive T, I, F matrices from a classifier's predicted probability matrix.

    T_c(r) = probs[r, c]                       (own predicted probability)
    F_c(r) = max probability among OTHER categories (strongest competitor)
    I_c(r) = normalized Shannon entropy of the full probability row
             (same value across all categories for a given requirement)

    Parameters
    ----------
    probs : np.ndarray of shape (n_samples, n_categories)

    Returns
    -------
    T, I, F : np.ndarray, each of shape (n_samples, n_categories)
    """
    n_samples, n_cats = probs.shape
    T = probs.copy()
    F = np.zeros_like(probs)
    I = np.zeros_like(probs)

    for i in range(n_samples):
        row = probs[i]
        row_entropy = normalized_entropy(row)
        for c in range(n_cats):
            other_probs = np.delete(row, c)
            F[i, c] = other_probs.max()
            I[i, c] = row_entropy

    return T, I, F


def neutrosophic_distance(T, I, F, w1=0.34, w2=0.33, w3=0.33):
    """
    Weighted neutrosophic distance to the ideal prototype <1, 0, 0>.
    d(r,c) = sqrt(w1*(1-T)^2 + w2*I^2 + w3*F^2)
    """
    return np.sqrt(w1 * (1 - T) ** 2 + w2 * I ** 2 + w3 * F ** 2)


def neutrosophic_entropy(T, I, F, eps=1e-9):
    """
    Neutrosophic entropy (ambiguity measure).
    E(r,c) = 1 - |T - F| / (1 + I)
    Higher values indicate greater classification ambiguity.
    """
    return 1 - np.abs(T - F) / (1 + I + eps)


def top_category_scores(probs: np.ndarray, T, I, F, D, E):
    """
    Convenience function: for each requirement, extract T/I/F/D/E at its
    top-predicted category (argmax of probs), returning one value per
    requirement rather than per (requirement, category) pair.
    """
    best_idx = probs.argmax(axis=1)
    idx = np.arange(len(probs))
    return {
        'top_category_idx': best_idx,
        'T': T[idx, best_idx],
        'I': I[idx, best_idx],
        'F': F[idx, best_idx],
        'D': D[idx, best_idx],
        'E': E[idx, best_idx],
    }


if __name__ == "__main__":
    import pandas as pd
    from data import load_nice_dataset, LABEL_COLS

    df, texts, Y = load_nice_dataset()
    probs = np.load("results/probs_svm.npy")

    T, I, F = derive_TIF(probs)
    D = neutrosophic_distance(T, I, F)
    E = neutrosophic_entropy(T, I, F)

    top = top_category_scores(probs, T, I, F, D, E)
    df['num_labels'] = Y.sum(axis=1)
    df['pred_top_category'] = [LABEL_COLS[i] for i in top['top_category_idx']]
    df['I_score'] = top['I']
    df['E_score'] = top['E']
    df['D_score'] = top['D']

    df.to_csv("results/nice_with_neutrosophic_scores.csv", index=False)
    np.save("results/T.npy", T)
    np.save("results/I.npy", I)
    np.save("results/F.npy", F)
    np.save("results/D.npy", D)
    np.save("results/E.npy", E)
    print("Saved neutrosophic score matrices and summary CSV to results/")
