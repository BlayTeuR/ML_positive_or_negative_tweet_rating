# Tweet sentiment classification with hand-crafted features and tree ensembles

Binary sentiment classification (positive vs negative) of about 100k tweets. It uses **interpretable, lexicon-based features** instead of word embeddings, and compares five tree-based learners, each with and without grid search. Coursework for INF7370 (Machine Learning) at UQAM, Fall 2025.

> **Result:** the best model (Gradient Boosting, tuned, feature set v2) reaches **69.4 % accuracy** on a 20k-tweet validation split. The majority-class baseline is **56.5 %**. Feature engineering in v2 raised **precision from 0.685 to 0.703**, which means fewer false positives, the main failure mode found in v1.

## Results

Hold-out validation: 20 % of 99,989 labelled tweets (about 20k). Class balance is 56.5 % positive and 43.5 % negative.

| Model | v1 (6 features) Acc / F1 | **v2 (+15 features) Acc / F1** | v2 Precision | v2 Recall |
|---|---|---|---|---|
| AdaBoost (tuned) | 0.676 / 0.743 | 0.688 / 0.743 | 0.693 | 0.801 |
| Bagging (tuned) | 0.680 / 0.738 | 0.690 / 0.745 | 0.693 | 0.805 |
| Decision tree (depth 5) | 0.673 / 0.736 | 0.683 / **0.746** | 0.677 | 0.830 |
| **Gradient Boosting (tuned)** | 0.679 / 0.735 | **0.694** / 0.743 | **0.703** | 0.790 |
| Random Forest (tuned) | 0.680 / 0.737 | 0.691 / 0.742 | 0.698 | 0.792 |
| *Majority class* | *0.565* | *0.565* | – | – |

**Findings**
- **The features are the bottleneck, not the models.** All five learners land within about 1 point of each other, and `GridSearchCV` rarely moves accuracy by more than 0.2 points. Tree depth is the only hyperparameter that matters: accuracy is best at depth 5 for single trees and depth 10 for random forests, and drops at depth 20 or more.
- **v1 failure mode:** high recall (about 0.80–0.83) but many false positives. Negative tweets without explicit negative words were labelled positive by default.
- **v2 fix:** 15 new features target that failure mode: negation scope (`pos_near_negation`, `neg_near_negation`), intensifier scope (`pos_after_intensifier`), elongations ("soooo"), question marks, ellipses, punctuation runs, mentions and hashtag polarity. Every model gained accuracy and precision.
- **Ceiling of binary labels:** a qualitative review of 40 test predictions shows that mixed emotions ("great, but need a scolding") cannot be captured by a binary target.

## Pipeline

1. **Cleaning:** strip URLs and mentions, keep hashtag words (`#happy` → `happy`), expand slang (`lol` → `laughing`, `omg` → `oh my god`), collapse letter repetitions.
2. **Feature extraction:** lexicon-based scores. There are 6 features in v1 (net word polarity, net emoji polarity, exclamations plus intensifiers, negations, all-caps words, mean word length). v2 adds the 15 features listed above. The lexicons are enriched from public opinion-word lists.
3. **Models:** Decision Tree, AdaBoost, Bagging, Random Forest and Gradient Boosting (scikit-learn). Each is trained once with default parameters and once with `GridSearchCV`.
4. **Evaluation:** accuracy, precision, recall, F1 and confusion matrices, plus a qualitative error analysis.

## Next steps
- Compare against a TF-IDF + logistic regression baseline and a fine-tuned DistilBERT or RoBERTa. Both are needed to place the ~69 % in context.
- Report k-fold CV means ± standard deviation instead of a single split.
- Move to a 3- or 4-class target (positive / negative / mixed / neutral) or a probabilistic score.

## Reproduce

```bash
pip install -r requirements.txt
cd src
python training.py   # menu: (1) build features, (2) train, (3) evaluate, (4) compare models
python test.py       # predictions on the unlabeled test set -> test_predictions.csv
```
Base and grid-searched models are saved in `src/models/base` and `src/models/opt_results`. The v2 training and evaluation scripts live in `version_2/`. *(The v2 feature-extraction script still needs to be committed so that v2 can be reproduced.)*

Full report (FR): [`rapport_Jallais_Bastien_TP1.pdf`](./rapport_Jallais_Bastien_TP1.pdf)

**Stack:** Python · scikit-learn · pandas · Matplotlib / Seaborn
