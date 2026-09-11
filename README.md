# Neutrosophic Distance and Entropy-Based Classification of Non-Functional Software Requirements

This repository contains the code and data used to produce the results reported in the paper *"Neutrosophic Distance and Entropy-Based Classification of Non-Functional Software Requirements."*

## Overview

Non-functional requirements (NFRs) frequently belong to more than one quality category simultaneously, yet most classifiers represent classification confidence as a single scalar value per category. This repository implements a single-valued neutrosophic set (SVNS) framework — representing each classification decision through independent truth (T), indeterminacy (I), and falsity (F) components — and empirically validates whether the resulting entropy measure distinguishes genuinely ambiguous, multi-label requirements from unambiguous ones, using real human-annotated ground truth.

## Repository Structure

```
.
├── data/
│   └── nice_nfr.csv              # Cleaned NICE dataset (NFR-only subset, 11 categories)
├── src/
│   ├── data.py                   # Dataset loading utilities
│   ├── classifiers.py            # SVM and Naive Bayes probability derivation
│   ├── neutrosophic.py           # T/I/F derivation, distance and entropy measures
│   ├── fuzzy_baseline.py         # Fuzzy c-means nearest-prototype baseline
│   └── evaluate.py               # Classification metrics + statistical validation
├── results/                      # Output directory (populated by running the pipeline)
├── run_all.py                    # Runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Dataset

The dataset used in this study, `data/nice_nfr.csv`, is a filtered subset of the **NICE dataset** (Rejithkumar & Anish, 2025), itself a multi-label re-annotation of the augmented PROMISE non-functional requirements corpus (Cleland-Huang et al., 2007; Dalpiaz et al., 2019). The original NICE dataset is publicly available at:

> https://zenodo.org/records/14590935

The filtered subset used here retains only requirements confirmed as non-functional (`IsQuality == 1`), drops the empty "Other" category, and contains 381 requirements across 11 NFR categories: Availability, Fault Tolerance, Legal, Look & Feel, Maintainability, Operability, Performance, Portability, Scalability, Security, and Usability. Each requirement may have one, two, or three true category labels.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended. All dependencies are standard scientific Python packages (scikit-learn, pandas, numpy, scipy, scikit-fuzzy).

## Reproducing the Results

Run the full pipeline from the repository root:

```bash
python run_all.py
```

This runs, in order:

1. **`src/data.py`** — loads and summarizes the dataset.
2. **`src/classifiers.py`** — trains the SVM and Naive Bayes classifiers under 5-fold cross-validation and saves out-of-fold predicted probabilities to `results/probs_svm.npy` and `results/probs_nb.npy`.
3. **`src/neutrosophic.py`** — derives T/I/F from the SVM probabilities and computes the neutrosophic distance (`D`) and entropy (`E`) measures, saving per-requirement scores to `results/nice_with_neutrosophic_scores.csv`.
4. **`src/fuzzy_baseline.py`** — runs the fuzzy c-means nearest-prototype baseline and saves membership degrees to `results/membership_fuzzy.npy`.
5. **`src/evaluate.py`** — computes classification performance (precision/recall/F1, Hamming loss) and the statistical validation tests reported in the paper (Mann–Whitney U, point-biserial correlation, Spearman correlation), saving a full report to `results/evaluation_report.json`.

Each module can also be run individually (from the repository root, so that relative paths resolve correctly), e.g.:

```bash
python src/classifiers.py
python src/neutrosophic.py
```

## Key Results

Running the pipeline reproduces the following findings reported in the paper:

| Test | Statistic | p-value |
|---|---|---|
| Neutrosophic entropy (SVM) vs. multi-label ground truth | Mann–Whitney U | p ≈ 3.1 × 10⁻¹⁰ |
| Neutrosophic entropy vs. classifier misclassification | Mann–Whitney U | p ≈ 8.8 × 10⁻¹⁵ |
| Fuzzy membership entropy vs. multi-label ground truth | Mann–Whitney U | p ≈ 0.95 (not significant) |

The neutrosophic entropy measure significantly distinguishes genuinely multi-label requirements from single-label ones and correlates with classifier misclassification, while the analogous entropy computed from a conventional fuzzy c-means membership vector does not.

## Reproducibility Notes

- All classifier probabilities are obtained via 5-fold cross-validation with out-of-fold prediction, so no requirement's score is influenced by its own presence in a training fold.
- Random seeds are fixed (`random_state=42`) throughout for reproducibility; minor numerical variation may still occur across scikit-learn versions.
- The exact package versions used to produce the results reported in the paper are listed as comments in `requirements.txt`.

## Citation

If you use this code or the accompanying dataset subset, please cite the paper and the original NICE dataset:

```bibtex
@article{yourname2026neutrosophic,
  title   = {Neutrosophic Distance and Entropy-Based Classification of Non-Functional Software Requirements},
  author  = {Your Name},
  journal = {AIMS Mathematics},
  year    = {2026}
}

@inproceedings{rejithkumar2025nice,
  title     = {NICE: Non-Functional Requirements Identification, Classification, and Explanation Using Small Language Models},
  author    = {Rejithkumar, G. and Anish, P. R.},
  booktitle = {2025 IEEE/ACM 47th International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP)},
  pages     = {284--295},
  year      = {2025}
}
```

## License

Code in this repository is released under the MIT License (see `LICENSE`). The included dataset subset is derived from the NICE dataset, released under CC-BY-4.0 by its original authors.
