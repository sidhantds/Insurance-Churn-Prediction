# 🛡️ Insurance Customer Churn Prediction

Predict which insurance customers are likely to **churn** (cancel or not renew their policy), so the company can step in early with retention offers.

> **Result:** Random Forest picked over Logistic Regression and Decision Tree with the best F1 (0.466) and precision (0.473) on a 12% churn rate. Found and removed a leaking column (`acct_suspd_date`) that would have given a fake perfect score.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Colab](https://img.shields.io/badge/Notebook-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

---

## 📌 Problem Statement

Keeping an existing customer costs far less than winning a new one. This project builds a **binary classification** model that predicts whether a customer will churn:

- `Churn = 1` → customer left
- `Churn = 0` → customer stayed

---

## 📂 Dataset

| Item | Details |
|---|---|
| Source | [Kaggle – Auto Insurance Churn Analysis Dataset](https://www.kaggle.com/datasets/merishnasuwal/auto-insurance-churn-analysis-dataset) |
| File | `autoinsurance_churn.csv` |
| Rows used | 37,791 customers (subset), 37,790 after removing 1 incomplete row |
| Columns | 22 |
| Target | `Churn` |
| Type | Synthetic but realistic insurance customer data |

### Churn distribution

```mermaid
pie showData title Customer Churn (Imbalanced Data)
    "Stayed (0) - 33,234" : 33234
    "Churned (1) - 4,556" : 4556
```

> Only **12.06%** of customers churned, so the data is **imbalanced**. A model that always predicts "stay" would already get 87.9% accuracy, so accuracy alone is misleading.

---

## 🏗️ System design

### Current design: what the notebook does

```mermaid
flowchart LR
    RAW[("autoinsurance_churn.csv<br/>37,791 rows × 22 columns")] --> AUD["Column audit<br/>drop IDs, raw dates, location,<br/>and the leaking acct_suspd_date"]
    AUD --> SPLIT{{"Stratified 80 / 20 split<br/>happens BEFORE any fitting"}}

    SPLIT --> TRAIN["Training set"]
    SPLIT --> TEST["Test set"]

    TRAIN --> FE["Feature engineering<br/>tenure_years, is_new_customer"]
    TEST --> FE2["Same feature steps,<br/>repeated by hand"]

    FE --> MAP["marital_status mapped to 0 / 1"]
    FE2 --> MAP
    MAP --> ENC["get_dummies<br/>test columns aligned to train"]
    ENC --> SCL["StandardScaler<br/>fit on train only"]

    SCL --> LR["Logistic Regression"]
    ENC --> DT["Decision Tree"]
    MAP --> RF["Random Forest"]
    LR --> EV["Compare on test set,<br/>pick best F1"]
    DT --> EV
    RF --> EV
```

The key protection is **the order**: the split happens before any scaler or encoder is fitted, so nothing learned from the test rows leaks into training. One hot encoding is aligned to the training columns, so a category that only appears in the test set cannot add a new column.

### Design decisions and trade offs

| Decision | Why | Cost |
|---|---|---|
| Drop `acct_suspd_date` | It is filled for 100% of churners and 0% of non churners. It records the outcome rather than predicting it | None; keeping it would make the model useless in practice |
| Drop raw dates, keep tenure | Tenure is what carries the signal; a raw date also encodes when the data was extracted | The exact start month is lost |
| Scaler only for Logistic Regression | Trees split on thresholds and do not care about scale | Two slightly different input paths to maintain |
| F1 and precision over accuracy | With 12% churn, "nobody churns" already scores 87.9% accuracy | F1 weights precision and recall equally, which may not match the real cost of a missed churner |

### Gaps in the current design

* **Model chosen on the test set.** All three models are scored on the test set and the best is picked there, so the reported 0.466 F1 is slightly optimistic. Cross validation on the training set should choose the model, and the test set should be used once at the end.
* **Preprocessing is done by hand.** Feature steps are written twice, once for train and once for test, and each model gets a slightly different input: scaled dummies, unscaled dummies, or the mapped columns. Nothing produces a single saved object that can score a new customer.
* **Default 0.5 threshold.** With imbalanced classes, choosing the threshold from validation data would usually give a better precision and recall balance.

### How I would ship it

```mermaid
flowchart LR
    NEW[("New customer records")] --> P
    subgraph P["One saved sklearn Pipeline (joblib)"]
        F["FunctionTransformer<br/>tenure features"] --> C["ColumnTransformer<br/>OneHotEncoder(handle_unknown='ignore')"]
        C --> RF["Random Forest,<br/>chosen by cross validation"]
    end
    P --> TH["Threshold tuned<br/>on validation data"]
    TH --> L[("Monthly ranked<br/>retention list")]
```

Wrapping every step in one Pipeline means training and scoring run the exact same code, the artifact can be saved and loaded in one line, and the manual train and test duplication disappears.

---

## 🔍 Exploratory Data Analysis – Key Findings

| # | Finding |
|---|---|
| 1 | **Tenure is the strongest churn driver**: new customers churn the most |
| 2 | Marital status has almost no effect (single 12.3% vs married 11.9%) |
| 3 | `acct_suspd_date` is filled for **100% of churners and 0% of non-churners** → **data leakage** |
| 4 | `state` has only one value (TX), so it carries no information |
| 5 | County churn rates such as 46% come from very small groups, so they aren't reliable |
| 6 | Data-quality issues: age spike at ~55, tenure capped at ~17 years, income recorded in brackets |
| 7 | `individual_id` and `address_id` are almost identical and unrelated to churn |

---

## 🧹 Data Preprocessing

Steps, in order: drop 10 columns (IDs, raw dates, location, and the leaking `acct_suspd_date`), check duplicates (0 found), drop 1 row with a missing label, split 80 / 20 with stratification, build tenure features, drop the text ranges in `home_market_value`, encode `marital_status`, and scale for Logistic Regression only.

**Why the scaler is fitted after the split:** it learns its mean and standard deviation from the **training data only**, so no information leaks from the test set.

| Set | Rows | Churn rate |
|---|---|---|
| Train | 30,232 | 12.06% |
| Test | 7,558 | 12.05% |

---

## 🛠️ Feature Engineering

| New feature | How it is made | Why |
|---|---|---|
| `tenure_years` | `days_tenure / 365` | Easier to understand than days |
| `is_new_customer` | 1 if tenure < 365 days | New customers churn the most (from EDA) |

---

## 🤖 Models

```mermaid
flowchart LR
    D[Training Data] --> LR[Logistic Regression<br/>scaled data]
    D --> DT[Decision Tree<br/>max_depth = 10]
    D --> RF[Random Forest<br/>100 trees, max_depth = 10]
    LR --> E[Evaluate on Test Data]
    DT --> E
    RF --> E
    E --> S[Select Best Model]
```

| Model | Simple idea |
|---|---|
| Logistic Regression | Weighted sum of the features → probability (linear) |
| Decision Tree | A chain of yes/no questions |
| Random Forest | 100 decision trees vote together |

All models use `class_weight='balanced'`, so mistakes on churners count more and the class imbalance is handled.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.840 | 0.378 | **0.507** | 0.433 |
| Decision Tree | 0.840 | 0.372 | 0.480 | 0.419 |
| **Random Forest** ⭐ | **0.873** | **0.473** | 0.460 | **0.466** |

```mermaid
xychart-beta
    title "F1-score by Model"
    x-axis ["Logistic Regression", "Decision Tree", "Random Forest"]
    y-axis "F1-score" 0 --> 0.6
    bar [0.433, 0.419, 0.466]
```

### Random Forest – classification report (test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 – Stayed | 0.93 | 0.93 | 0.93 | 6,647 |
| 1 – Churned | 0.47 | 0.46 | 0.47 | 911 |

The model catches about **46% of the customers who churn**, while keeping false alarms lower than the other models.

### Why not accuracy?

```mermaid
flowchart LR
    A["Lazy model:<br/>always predicts 'Stay'"] --> B["Accuracy 87.9%<br/>Recall 0%<br/>❌ Useless"]
    C["Random Forest"] --> D["Accuracy 87.3%<br/>Recall 46%<br/>✅ Useful"]
```

On imbalanced data, **recall** and **F1** matter more than accuracy. Missing a churner (false negative) costs more than a false alarm (false positive).

---

## 🏆 Model Selection

**Selected model: Random Forest**

1. Best F1-score (0.466), precision (0.473) and accuracy (0.873)
2. Captures non-linear patterns
3. More stable than a single Decision Tree because 100 trees vote together
4. Logistic Regression has slightly higher recall but far more false alarms (precision 0.378)

---

## 🎯 Prediction on Unseen Data

The final Random Forest outputs a **churn probability** for each customer with `predict_proba`. For example, the first 10 test customers got probabilities between 0.32 and 0.36. All were below the 0.5 threshold, so all were predicted to stay, and all 10 really stayed.

```mermaid
flowchart LR
    C[Customer details] --> M[Random Forest]
    M --> P[Churn probability]
    P -->|">= 0.5"| H[High risk: contact customer]
    P -->|"< 0.5"| L[Low risk]
```

---

## 💡 Business Recommendations

- **Focus retention on new customers** (first year), the highest-risk group
- Score customers regularly and contact the highest-probability ones first
- Collect richer data (claims, complaints, payment delays) to improve predictions

---

## ⚠️ Limitations & Future Work

- Synthetic data, so real-world patterns may differ
- A 37.8k-row subset was used; training on the full dataset could change results slightly
- Features are only weakly related to churn, which limits performance
- Next steps: hyperparameter tuning, cross-validation, Gradient Boosting / XGBoost, threshold tuning, more behavioural features

---

## 🗂️ Repository Structure

```
├── Insurance_Churn_Prediction_clean.ipynb   # Full notebook: data understanding → EDA → models → prediction
├── requirements.txt
└── README.md
```

> The dataset is not included in this repo. Download `autoinsurance_churn.csv` from the Kaggle link above.

---

## ▶️ How to Run

1. Clone the repo
   ```bash
   git clone https://github.com/sidhantds/insurance-churn-prediction.git
   ```
2. Install the libraries
   ```bash
   pip install -r requirements.txt
   ```
3. Download `autoinsurance_churn.csv` from Kaggle and put it next to the notebook
4. Open `Insurance_Churn_Prediction_clean.ipynb` in Jupyter or Google Colab and run all cells

---

## 🧰 Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Google Colab`

---

## 👤 Author

**Sidhant Pujara** · [GitHub](https://github.com/sidhantds)
