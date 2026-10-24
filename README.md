# 🛡️ Insurance Customer Churn Prediction

Predict which insurance customers are likely to **churn** (cancel or not renew their policy), so the company can step in early with retention offers.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter%20%2F%20Colab-F37626?logo=jupyter&logoColor=white)

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
| File used | `autoinsurance_churn.csv` |
| Size | 1,680,909 customers × 22 columns |
| Target | `Churn` |
| Type | Synthetic but realistic insurance customer data |

### Churn distribution

```mermaid
pie showData title Customer Churn (Imbalanced Data)
    "Stayed (0)" : 88.5
    "Churned (1)" : 11.5
```

> Only **11.5%** of customers churned, so the data is **imbalanced**. Accuracy alone would be misleading.

---

## 🔄 Project Workflow

```mermaid
flowchart LR
    A[Kaggle Dataset] --> B[Data Understanding]
    B --> C[EDA]
    C --> D[Preprocessing]
    D --> E[Feature Engineering]
    E --> F[Model Building]
    F --> G[Model Evaluation]
    G --> H[Model Selection]
    H --> I[Prediction on Unseen Data]
    I --> J[Report]
```

---

## 🔍 Exploratory Data Analysis – Key Findings

| # | Finding |
|---|---|
| 1 | **Tenure is the strongest churn driver** (correlation **-0.21**): new customers churn more |
| 2 | Age has a weak effect (-0.06) |
| 3 | Income and marital status have almost no effect (single 11.7% vs married 11.4% churn) |
| 4 | `acct_suspd_date` is filled only for churned customers → **data leakage** |
| 5 | Data-quality issues: age spike at ~55 (filled values), tenure capped at ~17 years, income recorded in brackets |
| 6 | `individual_id` and `address_id` are 0.99 correlated with each other and unrelated to churn |

---

## 🧹 Data Preprocessing

```mermaid
flowchart TD
    A[Raw Data<br/>22 columns] --> B[Drop columns]
    B --> B1[IDs: no pattern]
    B --> B2[Dates: covered by age and tenure]
    B --> B3[acct_suspd_date: data leakage]
    B --> B4[Location: no relation to churn]
    B --> B5[home_market_value: text ranges]
    B1 & B2 & B3 & B4 & B5 --> C[Remove duplicates]
    C --> D[Encode marital_status<br/>Married = 0, Single = 1]
    D --> E[Stratified 80/20 train-test split]
    E --> F[Feature engineering]
    F --> G[StandardScaler<br/>for Logistic Regression only]
```

**Why the split comes before scaling:** the scaler learns its mean and standard deviation from the **training data only**, so no information leaks from the test set.

| Set | Rows | Churn rate |
|---|---|---|
| Train | 1,344,727 | 11.51% |
| Test | 336,182 | 11.51% |

---

## 🛠️ Feature Engineering

| New feature | How it is made | Why |
|---|---|---|
| `tenure_years` | `days_tenure / 365` | Easier to read than days |
| `is_new_customer` | 1 if tenure < 365 days | New customers churn more (from EDA) |

Feature importance was measured with the Random Forest's `feature_importances_`.

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

All models use `class_weight='balanced'`, so mistakes on churners count more and the class imbalance is handled.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.851 | 0.380 | **0.465** | 0.418 |
| Decision Tree | 0.876 | 0.460 | 0.454 | 0.457 |
| **Random Forest** ⭐ | **0.876** | **0.461** | 0.456 | **0.458** |

```mermaid
xychart-beta
    title "F1-score by Model"
    x-axis ["Logistic Regression", "Decision Tree", "Random Forest"]
    y-axis "F1-score" 0 --> 0.6
    bar [0.418, 0.457, 0.458]
```

### Random Forest – Confusion Matrix (test set)

| | Predicted: Stay | Predicted: Churn |
|---|---|---|
| **Actual: Stay** | ✅ 276,878 (TN) | ❌ 20,613 (FP) |
| **Actual: Churn** | ❌ 21,067 (FN) | ✅ 17,624 (TP) |

The model caught **17,624 of 38,691 churners (≈46%)**.

### Why not accuracy?

```mermaid
flowchart LR
    A["Lazy model:<br/>always predicts 'No Churn'"] --> B["Accuracy 88.5%<br/>Recall 0%<br/>❌ Useless"]
    C["Random Forest"] --> D["Accuracy 87.6%<br/>Recall 46%<br/>✅ Useful"]
```

On imbalanced data, **recall** and **F1** matter more than accuracy. Missing a churner (FN) costs more than a false alarm (FP).

---

## 🏆 Model Selection

**Selected model: Random Forest (threshold = 0.5)**

1. Best F1-score and precision
2. Captures non-linear patterns, which is why the tree models beat Logistic Regression
3. More stable than a single Decision Tree because 100 trees vote together
4. Gives feature importance, so its predictions can be explained

### Threshold tuning

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| **0.50** ⭐ | 0.455 | 0.462 | **0.459** |
| 0.45 | 0.460 | 0.447 | 0.453 |
| 0.40 | 0.469 | 0.408 | 0.436 |
| 0.35 | 0.990 | 0.116 | 0.207 |
| 0.30 | 1.000 | 0.115 | 0.206 |

Lowering the threshold added only ~1% recall before the model started flagging almost every customer (precision dropped to the churn rate). The default **0.5** was kept.

---

## 🎯 Prediction Demo

The final model was tested on unseen customers. Changing **only the tenure** of the same customer shows the effect the model learned:

```mermaid
flowchart LR
    C[Same customer] --> N[Tenure = 100 days] --> HP[Higher churn probability]
    C --> O[Tenure = 5000 days] --> LP[Lower churn probability]
```

---

## 💡 Business Recommendations

- **Focus retention on new customers** (first year), because they are the highest-risk group
- Score all customers regularly and contact the high-probability ones with offers or calls
- Collect richer data (claims, complaints, payment delays) to improve predictions

---

## ⚠️ Limitations & Future Work

- The dataset is synthetic, so real-world patterns may differ
- Features are only weakly related to churn, which limits performance
- Next steps: Gradient Boosting / XGBoost, cross-validation, hyperparameter tuning, more behavioural features

---

## 🗂️ Repository Structure

```
├── Insurance_Churn_Prediction.ipynb   # Full notebook: EDA → models → report
└── README.md                          # Project overview (this file)
```

> The dataset is not included because of its size. Download `autoinsurance_churn.csv` from the Kaggle link above.

---

## ▶️ How to Run

1. Clone the repo
   ```bash
   git clone https://github.com/sidhantds/insurance-churn-prediction.git
   ```
2. Install the libraries
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Download `autoinsurance_churn.csv` from Kaggle and put it next to the notebook
4. Open `Insurance_Churn_Prediction.ipynb` (Jupyter or Google Colab) and run all cells

---

## 🧰 Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Jupyter / Google Colab`

---

## 👤 Author

**Sidhant Pujara** · [GitHub](https://github.com/sidhantds)
