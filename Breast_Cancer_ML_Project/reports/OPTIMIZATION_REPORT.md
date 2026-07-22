# Model Optimization — Performance Report
## Breast Cancer Wisconsin (Diagnostic) — Stage 5

Tuning via GridSearchCV / RandomizedSearchCV with 5-fold StratifiedKFold cross-validation. Scoring = **F1 on malignant (0)**.

Optimized models: **SVM** and **XGBoost**.

## Best Hyperparameters

### SVM
- Best CV F1: **0.9704**
- Best params: `{'C': 5, 'gamma': 0.01, 'kernel': 'rbf'}`

### XGBoost
- Best CV F1: **0.9647**
- Best params: `{'subsample': 0.9, 'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.9}`

## Before vs After (test set)

### SVM
| Metric    |   Before |   After |   Delta |
|:----------|---------:|--------:|--------:|
| Accuracy  |   0.9649 |  0.9561 | -0.0088 |
| Precision |   0.9318 |  0.9111 | -0.0207 |
| Recall    |   0.9762 |  0.9762 | -0      |
| F1-score  |   0.9535 |  0.9425 | -0.011  |

### XGBoost
| Metric    |   Before |   After |   Delta |
|:----------|---------:|--------:|--------:|
| Accuracy  |   0.9561 |  0.9649 |  0.0088 |
| Precision |   0.9744 |  0.975  |  0.0006 |
| Recall    |   0.9048 |  0.9286 |  0.0238 |
| F1-score  |   0.9383 |  0.9512 |  0.0129 |

## Final Model

**XGBoost** (optimized) was saved as `models/final_model.joblib`.

- Accuracy: **0.9649**
- Precision (malignant): **0.9750**
- Recall (malignant): **0.9286**
- F1-score (malignant): **0.9512**
- Best params: `{'subsample': 0.9, 'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.9}`
