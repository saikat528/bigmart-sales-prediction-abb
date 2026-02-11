# bigmart-sales-prediction-abb
End-to-end ML pipeline for BigMart sales prediction with advanced feature engineering and stage-wise tuning

# BigMart Sales Prediction – ABB Evaluation

## Objective
Predict Item_Outlet_Sales for unseen item–outlet pairs using historical sales data.
Evaluation metric: Root Mean Squared Error (RMSE).

---

## Repository Structure
- `notebooks/` – EDA, feature engineering, and model experiments
- `src/` – Final reproducible training and prediction script
- `submission/` – Final submission CSV
- `docs/` – 1-page approach note

---

## Feature Engineering Highlights
- Outlet maturity & visibility normalization
- Relative pricing and density-based demand features
- Item–Outlet interaction features
- MRP segmentation

---

## Model
- HistGradientBoostingRegressor
- Stage-wise hyperparameter tuning
- 5-fold cross-validated RMSE optimization

---

## How to Run
1. Place `train.csv` and `test.csv` inside a local `data/` directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

