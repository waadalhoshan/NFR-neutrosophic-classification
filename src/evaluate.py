"""
evaluate.py — Classification performance metrics and the statistical
validation tests reported in Section 5 of the paper:

  1. Multi-label classification performance (precision/recall/F1, Hamming loss)
  2. Ambiguity validation: does the entropy measure distinguish single-label
     from multi-label (ground-truth) requirements?
  3. Diagnostic value: does the entropy measure correlate with classifier
     misclassification?
"""
import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, hamming_loss, f1_score

from data import LABEL_COLS


def classification_performance(probs: np.ndarray, Y: np.ndarray, threshold: float = 0.3):
    """
    Convert probabilities to binary multi-label predictions via a fixed
    threshold (falling back to the top-1 category if no label clears the
    threshold), then compute macro/micro precision, recall, F1, and
    Hamming loss.
    """
    Y_pred = (probs >= threshold).astype(int)
    no_pred_rows = Y_pred.sum(axis=1) == 0
    if no_pred_rows.any():
        top1 = probs.argmax(axis=1)
        for i in np.where(no_pred_rows)[0]:
            Y_pred[i, top1[i]] = 1

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        Y, Y_pred, average='macro', zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        Y, Y_pred, average='micro', zero_division=0)
    h_loss = hamming_loss(Y, Y_pred)

    p_cat, r_cat, f1_cat, support_cat = precision_recall_fscore_support(
        Y, Y_pred, average=None, zero_division=0)

    return {
        'precision_macro': p_macro, 'recall_macro': r_macro, 'f1_macro': f1_macro,
        'precision_micro': p_micro, 'recall_micro': r_micro, 'f1_micro': f1_micro,
        'hamming_loss': h_loss,
        'per_category': {
            LABEL_COLS[i]: {'precision': p_cat[i], 'recall': r_cat[i],
                             'f1': f1_cat[i], 'support': int(support_cat[i])}
            for i in range(len(LABEL_COLS))
        }
    }


def validate_ambiguity_detection(score: np.ndarray, num_labels: np.ndarray):
    """
    Test whether an ambiguity score (e.g., neutrosophic E, or fuzzy
    membership entropy) differs between single-label and multi-label
    ground-truth requirements. Returns Mann-Whitney U, point-biserial
    correlation, and Spearman correlation with exact label count.
    """
    single = score[num_labels == 1]
    multi = score[num_labels >= 2]
    is_multi = (num_labels >= 2).astype(int)

    u_stat, mw_p = stats.mannwhitneyu(single, multi, alternative='less')
    pb_corr, pb_p = stats.pointbiserialr(is_multi, score)
    rho, rho_p = stats.spearmanr(num_labels, score)

    return {
        'single_mean': float(single.mean()), 'single_std': float(single.std()),
        'multi_mean': float(multi.mean()), 'multi_std': float(multi.std()),
        'mannwhitney_u': float(u_stat), 'mannwhitney_p': float(mw_p),
        'point_biserial_r': float(pb_corr), 'point_biserial_p': float(pb_p),
        'spearman_rho': float(rho), 'spearman_p': float(rho_p),
    }


def validate_error_diagnostic(score: np.ndarray, probs: np.ndarray, Y: np.ndarray):
    """
    Test whether an ambiguity score is higher for requirements where the
    classifier's top-1 prediction is NOT among the true labels.
    """
    top_pred = probs.argmax(axis=1)
    is_correct = np.array([Y[i, top_pred[i]] == 1 for i in range(len(Y))]).astype(int)

    correct_scores = score[is_correct == 1]
    error_scores = score[is_correct == 0]

    u_stat, mw_p = stats.mannwhitneyu(correct_scores, error_scores, alternative='less')
    pb_corr, pb_p = stats.pointbiserialr(1 - is_correct, score)

    return {
        'top1_accuracy': float(is_correct.mean()),
        'correct_mean': float(correct_scores.mean()),
        'error_mean': float(error_scores.mean()),
        'mannwhitney_p': float(mw_p),
        'point_biserial_r': float(pb_corr), 'point_biserial_p': float(pb_p),
    }


if __name__ == "__main__":
    import json
    import pandas as pd

    Y = np.load("results/Y.npy")
    probs_svm = np.load("results/probs_svm.npy")
    probs_nb = np.load("results/probs_nb.npy")
    membership_fuzzy = np.load("results/membership_fuzzy.npy")
    df_scores = pd.read_csv("results/nice_with_neutrosophic_scores.csv")

    num_labels = Y.sum(axis=1)

    report = {}
    report['classification_svm'] = classification_performance(probs_svm, Y)
    report['classification_nb'] = classification_performance(probs_nb, Y)
    report['classification_fuzzy'] = classification_performance(membership_fuzzy, Y, threshold=0.15)

    report['ambiguity_svm_E'] = validate_ambiguity_detection(df_scores['E_score'].values, num_labels)

    fuzzy_entropy = np.array([
        -np.sum(np.clip(row, 1e-12, None) / row.sum() * np.log(np.clip(row, 1e-12, None) / row.sum()))
        / np.log(len(row))
        for row in membership_fuzzy
    ])
    report['ambiguity_fuzzy'] = validate_ambiguity_detection(fuzzy_entropy, num_labels)

    report['error_diagnostic_svm_E'] = validate_error_diagnostic(
        df_scores['E_score'].values, probs_svm, Y)

    with open("results/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\nSaved full evaluation report to results/evaluation_report.json")
