# 💎 Diamond Price Prediction

A machine learning project that predicts diamond prices based on physical and quality attributes. Built using Python and Scikit-Learn, this project demonstrates a full ML pipeline from data cleaning and feature engineering to model comparison and evaluation.

## Problem Statement

Diamonds can look identical to the naked eye, yet vary dramatically in price. The challenge is identifying which attributes — cut, color, clarity, carat, and dimensions — most influence price, and building a model that predicts it accurately.

## Dataset

- **Source:** Diamonds dataset (53,940 records)
- **Features:** carat, cut, color, clarity, depth, table, x, y, z dimensions, price
- **Target variable:** `price` (USD)

## Approach

### Data Preprocessing
- Removed records where any dimension (x, y, z) was 0 — physically impossible diamonds that would skew results
- Engineered a new **`size`** feature by multiplying x × y × z, which captures volumetric size and correlates more strongly with price than individual dimensions
- Dropped raw dimension columns to eliminate multicollinearity
- Applied **LabelEncoder** to `cut` (ordinal — quality scale) to preserve rank ordering
- Applied **One-Hot Encoding** to `color` and `clarity` (nominal categories)

### Models Compared
All models evaluated using **10-fold cross-validation** and **RMSE (Root Mean Squared Error)**:

| Model | Type |
|-------|------|
| Linear Regression | Baseline |
| Lasso Regression | L1 regularization |
| ElasticNet | L1 + L2 regularization |
| Ridge Regression | L2 regularization |
| Decision Tree Regressor | Non-linear |

### Data Split
- 80% training / 20% test
- Training set further split: 75% train / 25% validation

## Project Structure

```
DiamondPricePrediction/
├── Diamond.py                  # Full ML pipeline
├── Diamonds Presentation.pptx  # Project presentation slides
└── README.md
```

## Programming Language and Libraries

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## Key Takeaways

- Feature engineering (composite `size` attribute) improved model performance over using raw dimensions
- Ordinal encoding outperformed one-hot encoding for the `cut` attribute due to its rank-based nature
- Decision Tree captured non-linear relationships that linear models missed
- Cross-validation provided more reliable performance estimates than a single train/test split

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Update the dataset path in Diamond.py
# Run the script
python Diamond.py
```
