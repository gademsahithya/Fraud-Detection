# 🛡️ Fraud Detection using XGBoost

An end-to-end machine learning project for detecting fraudulent
financial transactions using the **PaySim synthetic transaction
dataset**.

The project focuses not only on training an XGBoost classifier, but on
the complete ML journey: **baseline modeling → feature engineering →
hyperparameter tuning → threshold tuning → evaluation → live
deployment**.

🌐 **Live Demo:** https://fraud-detection-14zq.onrender.com/

------------------------------------------------------------------------

## 📌 Project Overview

Financial fraud detection is a highly imbalanced classification problem.
In a real transaction dataset, legitimate transactions greatly outnumber
fraudulent ones.

This makes **accuracy alone a poor evaluation metric**.

For example, a model can achieve around 99% accuracy simply by
predicting most transactions as legitimate while still missing a large
number of fraudulent transactions.

Therefore, this project focuses on:

-   Precision
-   Recall
-   F1-score
-   Confusion matrix
-   Threshold tuning
-   Class imbalance
-   Historical customer behavior
-   Transaction and balance-based features

The final model is an **XGBoost classifier** that produces a fraud
probability. A transaction is flagged when its predicted probability
reaches the selected threshold.

------------------------------------------------------------------------

## 🎯 Objective

The objective was to build a fraud detection model that can identify as
many fraudulent transactions as possible while keeping false fraud
alerts under control.

The project started with a simple baseline and progressively improved
through feature engineering and model optimization.

The most important improvement came from giving the model information
about:

> **Previous account behavior + current transaction/balance context**

------------------------------------------------------------------------

# 🔄 My Machine Learning Journey

This project was built progressively rather than jumping directly to the
final model.

## 1. Logistic Regression Baseline

I first created a Logistic Regression baseline to establish a reference
point.

### Result

``` text
Accuracy: 99.87%

Confusion Matrix:

[[1270891      13]
 [   1620       0]]
```

Fraud-class performance:

``` text
Precision: 0.00
Recall:    0.00
F1-score:  0.00
```

Although the accuracy looked extremely high, the model detected:

``` text
True Positives = 0
```

This demonstrated an important lesson:

> **High accuracy does not mean a fraud detection model is performing
> well.**

Because fraud is rare, accuracy can be misleading.

------------------------------------------------------------------------

# 2. Initial XGBoost Model

I then moved to **XGBoost**, a powerful gradient boosting algorithm
suitable for tabular classification problems.

The initial XGBoost model improved fraud detection substantially.

### Result

``` text
Accuracy: 99.90%

Confusion Matrix:

[[1270805      99]
 [   1110     510]]
```

Fraud performance:

``` text
Precision: ~84%
Recall:    ~31%
F1-score:  ~46%
```

The model was now detecting fraudulent transactions, but it was still
missing many fraud cases.

------------------------------------------------------------------------

# 3. Hyperparameter Tuning

Instead of accepting the default XGBoost configuration, I used
**RandomizedSearchCV** to search for better hyperparameters.

The parameters explored included:

-   `n_estimators`
-   `learning_rate`
-   `max_depth`
-   `min_child_weight`

The best configuration from the search was:

``` python
{
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.20,
    "min_child_weight": 1
}
```

The best cross-validation recall was approximately:

``` text
35.6%
```

This showed that model configuration could improve fraud detection, but
there was still significant room for improvement.

------------------------------------------------------------------------

# 4. Tuned XGBoost

Using the tuned model, the test-set result improved to:

``` text
Confusion Matrix:

[[1270729     175]
 [   1026     594]]
```

Fraud performance:

``` text
Precision: ~77%
Recall:    ~37%
F1-score:  ~50%
```

The model now correctly identified:

``` text
594 fraudulent transactions
```

But I wanted to improve the model further.

------------------------------------------------------------------------

# 5. Feature Engineering 🚀

This became the biggest improvement in the project.

Instead of relying mainly on the raw transaction values, I engineered
features representing **customer behavior and transaction context**.

## Historical sender behavior

### Previous transaction count

``` text
origin_transaction_count
```

Number of transactions previously made by the sender.

### Historical average transaction amount

``` text
origin_avg_amount
```

Average amount of the sender's previous transactions.

### Historical maximum transaction amount

``` text
origin_max_amount
```

Largest amount previously sent by the sender.

### Transaction amount vs historical average

``` text
amount_vs_origin_avg
```

This compares the current transaction with the sender's previous average
transaction amount.

For example, if a customer usually sends around ₹1,500 and suddenly
sends ₹12,000, this feature becomes much larger.

------------------------------------------------------------------------

# 6. Destination Account Behavior

I also created features describing the receiving account.

``` text
destination_transaction_count
destination_unique_origins
```

These capture information such as:

-   How many transactions the destination account has received
-   How many different senders have interacted with the destination

This provides additional behavioral context to the model.

------------------------------------------------------------------------

# 7. Balance-Based Features

Raw balances were transformed into ratios that provide more meaningful
transaction context.

### Origin amount ratio

``` text
origin_amount_ratio = amount / oldbalanceOrg
```

This represents how much of the sender's available balance is involved
in the transaction.

### Origin remaining ratio

``` text
origin_remaining_ratio = newbalanceOrig / oldbalanceOrg
```

This represents the proportion of the sender's balance remaining after
the transaction.

### Destination amount ratio

``` text
destination_amount_ratio = amount / oldbalanceDest
```

This provides context about the transaction amount relative to the
destination's previous balance.

### Destination balance ratio

``` text
destination_balance_ratio = newbalanceDest / oldbalanceDest
```

This captures how the destination balance changes relative to its
previous balance.

Zero-denominator cases were handled explicitly rather than allowing
infinite values.

------------------------------------------------------------------------

# 8. Final Feature Set

The final model uses features from three main categories.

### Transaction information

``` text
step
amount
type_CASH_IN
type_CASH_OUT
type_DEBIT
type_PAYMENT
type_TRANSFER
```

### Historical account behavior

``` text
origin_transaction_count
origin_avg_amount
origin_max_amount
amount_vs_origin_avg
destination_unique_origins
destination_transaction_count
```

### Balance context

``` text
origin_amount_ratio
origin_remaining_ratio
destination_amount_ratio
destination_balance_ratio
```

The target variable is:

``` text
isFraud
```

------------------------------------------------------------------------

# 9. Final Feature-Engineered XGBoost Model

After adding the behavioral and balance-based features, the model
improved dramatically.

The resulting confusion matrix was:

``` text
[[1270619     285]
 [    558    1062]]
```

At the default threshold of `0.50`:

``` text
Precision: 78.84%
Recall:    65.56%
F1-score:  71.59%
```

Most importantly, the number of correctly detected fraudulent
transactions increased to:

``` text
1,062 True Positives
```

This is the key improvement in the project.

------------------------------------------------------------------------

# 📈 From 0 to \~1,100 Correctly Detected Fraud Cases

The progression demonstrates why model development is more than simply
selecting an algorithm.

  Stage                          True Positive Fraud Cases   Fraud Recall   Fraud F1
  ---------------------------- --------------------------- -------------- ----------
  Logistic Regression                                    0          0.00%       0.00
  Initial XGBoost                                      510          \~31%     \~0.46
  Tuned XGBoost                                        594          \~37%     \~0.50
  Threshold tuning                                     755          \~47%     \~0.53
  Feature-engineered XGBoost                     **1,062**     **65.56%**   **0.72**

So the project progressed from:

**0 detected fraud cases → 510 → 594 → 755 → 1,062 correctly detected
fraud cases**

That is approximately **1,100 correctly detected fraud cases**, without
simply relying on accuracy.

------------------------------------------------------------------------

# 🎚️ Threshold Tuning

XGBoost produces a probability rather than an automatic business
decision.

For example:

``` text
Fraud probability = 0.73
```

The classification threshold determines whether that probability becomes
a fraud prediction.

I tested multiple thresholds:

    Threshold    Precision       Recall           F1
  ----------- ------------ ------------ ------------
         0.50       78.84%       65.56%       71.59%
     **0.40**   **75.22%**   **68.58%**   **71.75%**
         0.30       69.60%       72.22%       70.89%
         0.20       61.55%       76.98%       68.40%
         0.10       54.55%       79.57%       64.73%

The deployed application uses a **0.40 flagging threshold**.

At this threshold:

``` text
Probability >= 0.40 → Flag as fraud
Probability < 0.40  → Treat as non-fraud
```

The reason for lowering the threshold from 0.50 is to increase fraud
recall while maintaining reasonably high precision.

The live application explicitly displays the threshold used by the
deployed model.

------------------------------------------------------------------------

# 🧠 Why Accuracy Was Not My Main Metric

The dataset is highly imbalanced.

There are many more legitimate transactions than fraudulent
transactions.

Therefore:

``` text
Accuracy ≠ complete picture
```

A fraud detection system that predicts almost everything as legitimate
can achieve very high accuracy while failing to catch fraud.

Instead, this project focuses on:

### Precision

Of all transactions predicted as fraud, how many were actually fraud?

### Recall

Of all actual fraudulent transactions, how many did the model detect?

### F1-score

A combined measure of precision and recall.

For fraud detection, recall is particularly important because missing a
fraudulent transaction can be costly. However, precision also matters
because flagging too many legitimate transactions creates unnecessary
investigations.

------------------------------------------------------------------------

# 🏗️ Project Architecture

``` text
PaySim Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Time-Based Data Split
      ↓
Feature Engineering
      ↓
Historical Account Features
      ↓
Balance-Based Features
      ↓
XGBoost
      ↓
Hyperparameter Tuning
      ↓
Probability Prediction
      ↓
Threshold = 0.40
      ↓
Fraud / Non-Fraud
      ↓
Live Web Application
```

------------------------------------------------------------------------

# 🌐 Live Application

The project is deployed as a live transaction screening application:

**https://fraud-detection-14zq.onrender.com/**

The application allows a user to enter transaction information
including:

-   Transaction type
-   Amount
-   Simulation hour
-   Sender balance before transaction
-   Sender balance after transaction
-   Sender's past transaction count
-   Sender's average past amount
-   Sender's largest past amount
-   Receiver balance before transaction
-   Receiver balance after transaction
-   Receiver transaction count
-   Number of different senders

The application then sends these features through the trained XGBoost
model and returns a fraud risk estimate.

The live interface uses a **0.40 flagging threshold**.

------------------------------------------------------------------------

# ⚠️ Important Note About the Prediction

The model output is a **risk estimate**, not a definitive statement that
a transaction is fraudulent.

A real financial system would normally use the model as part of a larger
fraud investigation workflow.

For example:

``` text
Transaction
     ↓
ML Risk Score
     ↓
Threshold
     ↓
Low Risk → Continue
High Risk → Human Review
```

The deployed application therefore treats the model prediction as a
screening signal rather than a final financial decision.

------------------------------------------------------------------------

# 🛠️ Tech Stack

-   **Python**
-   **Pandas**
-   **NumPy**
-   **Scikit-learn**
-   **XGBoost**
-   **Matplotlib / Seaborn** for analysis and visualization
-   **Streamlit / web application layer**
-   **Render** for deployment

------------------------------------------------------------------------

# 📊 Dataset

This project uses the **PaySim synthetic financial transaction
dataset**.

Important columns include:

``` text
step
type
amount
nameOrig
oldbalanceOrg
newbalanceOrig
nameDest
oldbalanceDest
newbalanceDest
isFraud
isFlaggedFraud
```

The raw account identifiers were not directly one-hot encoded because
they have extremely high cardinality.

Instead, they were used to construct behavioral features and then
excluded from the final model feature matrix.

------------------------------------------------------------------------

# 🔐 Handling Historical Information

Historical account features were constructed using transactions that
occurred previously for the relevant account.

For example:

``` text
Current transaction
       ↑
Previous transactions of same sender
       ↓
count / average / maximum
```

This allows the model to identify unusual behavior relative to the
account's own history.

The transaction data was ordered by the PaySim simulation step so that
historical features represented information available before the current
transaction.

------------------------------------------------------------------------

# 💡 Key Lessons From the Project

### 1. Accuracy can be misleading

A highly imbalanced dataset requires metrics such as precision, recall,
and F1.

### 2. Feature engineering can matter more than changing algorithms

The biggest performance improvement came from creating meaningful
behavioral and balance-based features.

### 3. XGBoost is powerful for tabular data

The model was able to learn nonlinear relationships between transaction,
account-history, and balance features.

### 4. Threshold selection matters

The model's probability output does not automatically determine the
business decision.

Changing the threshold changes the balance between false positives and
false negatives.

### 5. Domain knowledge improves ML

Understanding what sender balances, receiver balances, transaction
history, and transaction types mean helped create more informative
features.

------------------------------------------------------------------------

# 🚀 Future Improvements

Possible future improvements include:

-   Precision-Recall AUC (PR-AUC) analysis
-   SHAP-based model explainability
-   More time-aware validation
-   Cost-sensitive threshold optimization
-   Model monitoring
-   Drift detection
-   More sophisticated account-level behavioral features
-   Real-time transaction scoring
-   Human-review workflow
-   Production database integration
-   Model versioning and experiment tracking

------------------------------------------------------------------------

# 👩‍💻 What This Project Demonstrates

This project demonstrates practical experience with:

-   Supervised machine learning
-   Binary classification
-   Highly imbalanced datasets
-   Exploratory data analysis
-   Feature engineering
-   Historical behavioral features
-   Ratio-based feature engineering
-   XGBoost
-   Hyperparameter tuning
-   Cross-validation
-   Threshold tuning
-   Precision/recall tradeoffs
-   Confusion matrix analysis
-   Model evaluation
-   Model deployment
-   Building a usable ML application

------------------------------------------------------------------------

## ⭐ Final Takeaway

This project was not built as a simple:

``` text
Dataset → Model → Accuracy
```

Instead, the development process was:

``` text
Baseline
   ↓
Identify the weakness
   ↓
Try XGBoost
   ↓
Tune hyperparameters
   ↓
Engineer account behavior features
   ↓
Engineer balance features
   ↓
Evaluate precision / recall / F1
   ↓
Tune classification threshold
   ↓
Deploy the model
```

The result was an improvement from **0 fraud cases correctly detected by
the initial Logistic Regression baseline to 1,062 correctly detected
fraudulent transactions with the final feature-engineered XGBoost
model** on the reported test evaluation.

> **From 0 → 1,062 correctly detected fraud cases through iterative ML
> development.**

------------------------------------------------------------------------

## 📬 Project

**Live Demo:** https://fraud-detection-14zq.onrender.com/

**Model:** XGBoost

**Problem:** Financial transaction fraud detection

**Dataset:** PaySim

**Final deployed threshold:** 0.40
