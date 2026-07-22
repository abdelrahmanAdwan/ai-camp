# Model Comparison Report
## Breast Cancer Wisconsin (Diagnostic) — Stage 4

Positive class = **malignant (0)**. Precision / Recall / F1 measure how well each model detects malignant tumours.

## Performance Comparison Table

| Model         |   Accuracy |   Precision |   Recall |   F1-score |
|:--------------|-----------:|------------:|---------:|-----------:|
| SVM           |     0.9649 |      0.9318 |   0.9762 |     0.9535 |
| XGBoost       |     0.9561 |      0.9744 |   0.9048 |     0.9383 |
| Random Forest |     0.9474 |      0.9286 |   0.9286 |     0.9286 |
| Decision Tree |     0.9298 |      0.8864 |   0.9286 |     0.907  |
| Naive Bayes   |     0.9298 |      0.9048 |   0.9048 |     0.9048 |

## Best Model

**SVM** is the best-performing baseline model.

- Accuracy: **0.9649**
- Precision (malignant): **0.9318**
- Recall (malignant): **0.9762**
- F1-score (malignant): **0.9535**

## Analysis — which model performed best and why

Selection criterion: highest **F1-score** on the malignant class (the balance of precision and recall), with **recall** and then **accuracy** as tie-breakers. Recall is emphasised because in a cancer screening context a false negative — labelling a malignant tumour as benign — is the most harmful error.

**SVM** achieved the strongest overall balance. Ensemble and margin-based methods (Random Forest, SVM, XGBoost) generally outperform the simpler baselines here: the features are numerous, continuous and highly correlated, which suits margin-maximising and ensemble models. **Naive Bayes**, which assumes feature independence, is handicapped by the strong multicollinearity found in Stage 1, while a single **Decision Tree** tends to overfit and is less stable than the ensembles.

The two strongest models — **SVM** and **XGBoost** — are carried forward to Stage 5 for hyperparameter optimization.

## Charts

- `figures/10_metric_comparison.png`
- `figures/11_accuracy.png`
- `figures/11_precision.png`
- `figures/11_recall.png`
- `figures/11_f1_score.png`
- `figures/12_grouped_comparison.png`