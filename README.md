# UFC Fighters Win Count Prediction

A regression-based machine learning pipeline that predicts the seasonal win count of UFC fighters based on physical attributes and fight statistics. Trained on 4,100+ fighter records with outlier detection, hyperparameter tuning, and model comparison across KNN, Random Forest, and XGBoost.

## How It Works

```
Fighter Data → Preprocessing → Outlier Detection → Feature Selection → Model Training → Win Count Prediction
```

## Dataset

- **Size:** 4,111 fighter records
- **Features:** 18 attributes (physical stats, fight performance metrics)
- **Target:** Win count (regression)
- **Notable entries:** Jeremy Horn (91 wins, most in dataset), Shannon Ritch (most losses)

## Features Used for Prediction

| Fight Performance | Physical |
|-------------------|----------|
| Significant strike defence | Reach (cm) |
| Takedown defense | Age |
| Significant striking accuracy | |
| Takedown accuracy | |
| Significant strikes landed per minute | |
| Average takedowns landed per 15 minutes | |

## Key Preprocessing Steps

- **Date of birth → Age:** Converted birth dates to age in years
- **Outlier detection:** Local Outlier Factor (LOF) with 20 neighbors, removed top 10 anomalous records
- **Missing data handling:** Dropped rows with null values in key numeric columns
- **No duplicates found** after outlier removal
- **Train/test split:** 80/20

## Top Correlated Features with Wins

| Feature | Correlation |
|---------|-------------|
| Losses | 0.628 |
| Significant strike defence | 0.339 |
| Draws | 0.293 |
| Age | 0.286 |
| Takedown defense | 0.276 |

## Models & Results (RMSE — lower is better)

| Model | Naive RMSE | Tuned RMSE | Best Parameters |
|-------|-----------|------------|-----------------|
| **XGBoost** | 8.03 | **7.42** | colsample=0.9, lr=0.01, depth=3, n_est=500 |
| Random Forest | 7.57 | 7.45 | max_depth=7, max_features=3, n_est=100 |
| KNN | 8.20 | 7.46 | n_neighbors=29 |

All models tuned with **10-fold cross-validation GridSearchCV**.

## Sample Prediction

Input: A 32-year-old fighter with 60% strike defence, 70% takedown defence, 33% striking accuracy, 70% takedown accuracy, 2.24 strikes/min, 2.0 takedowns/15min, 181.8cm reach.

| Model | Predicted Wins |
|-------|---------------|
| KNN | 15.7 |
| Random Forest | 12.3 |
| XGBoost | 12.6 |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| ML Models | Scikit-learn (KNN, RandomForest), XGBoost |
| Outlier Detection | LocalOutlierFactor |
| Visualization | Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy |

## Project Structure

```
├── UFC_Fighters_Wincount_Prediction.ipynb    # Full pipeline notebook
├── ufc_dataset.csv                           # Dataset (4,111 fighters)
└── README.md
```

## Key Findings

- XGBoost achieved the lowest RMSE (7.42) after tuning — best overall model
- All three models converged to similar error (~7.4-7.5), suggesting a natural performance ceiling with available features
- Defensive metrics (strike defence, takedown defence) are stronger predictors of win count than offensive stats
- Losses positively correlates with wins (0.628) — fighters with more fights tend to have both more wins and losses

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
jupyter notebook UFC_Fighters_Wincount_Prediction.ipynb
```

## License

Apache 2.0
