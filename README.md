# 🏡 Melbourne Housing Price Prediction

This project applies a machine learning pipeline to predict house prices in Melbourne, Australia. The dataset is cleaned, preprocessed, and used to train a `RandomForestRegressor` model wrapped inside a `scikit-learn` pipeline for efficient and reproducible modeling.

---

## 📊 Dataset

The dataset used is `melb_data.csv`, which contains various features such as:
- Location information (`Suburb`, `Address`, `Regionname`, etc.)
- Property characteristics (`Rooms`, `Type`, `Landsize`, etc.)
- Sale and seller details (`Date`, `SellerG`, etc.)

---

## 🧠 Model Overview

The project uses the following modeling pipeline:

- **Numerical Features**:
  - Imputation using the **mean**
  - Scaling using `StandardScaler` *(optional for trees but included for extensibility)*

- **Categorical Features**:
  - Imputation using the **most frequent value**
  - Encoding using `OneHotEncoder` (with `handle_unknown='ignore'`)

- **Model**:
  - `RandomForestRegressor` with tuned parameters:
    ```python
    RandomForestRegressor(
      n_estimators=200,
      max_depth=15,
      min_samples_split=10,
      min_samples_leaf=4,
      max_features='sqrt',
      random_state=42,
      n_jobs=-1
    )
    ```

---

## 🧪 Model Evaluation

The model is evaluated using **5-fold cross-validation** with the R² metric.

```text
Average R² Score: 0.786
