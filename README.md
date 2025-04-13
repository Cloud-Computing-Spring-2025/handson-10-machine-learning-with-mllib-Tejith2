# 📊 Hands-on #10: Customer Churn Prediction using Spark MLlib

## 🔍 Objective

This project applies Apache Spark MLlib to build and evaluate machine learning models for customer churn prediction. Tasks include data preprocessing, model training, feature selection, and hyperparameter tuning — all running in a containerized Spark environment via Docker.

---

## 🗃 Dataset

**File:** `customer_churn.csv`  
**Source:** Synthetic or generated via `dataset-generator.py`

**Features used:**
- `gender`, `PhoneService`, `InternetService` (categorical)
- `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` (numeric)
- `Churn` (label)

---

## 🧪 Tasks Completed

### ✅ Task 1: Data Preprocessing and Feature Engineering
- Handled missing values in `TotalCharges`
- Encoded categorical features using `StringIndexer` + `OneHotEncoder`
- Assembled feature vector using `VectorAssembler`

### ✅ Task 2: Train and Evaluate a Logistic Regression Model
- Split data into 80% train / 20% test
- Trained `LogisticRegression` model
- Evaluated using AUC (Area Under the ROC Curve)

### ✅ Task 3: Feature Selection using Chi-Square Test
- Selected top 5 features based on chi-square test
- Displayed selected feature vectors and churn labels

### ✅ Task 4: Hyperparameter Tuning & Model Comparison
- Compared Logistic Regression, Decision Tree, Random Forest, and GBT
- Used 5-fold CrossValidator with parameter grids
- Reported best AUC for each model

---

## 🐳 Run Instructions (Mac + Docker)

### Clone the Repo

```bash
git clone https://github.com/Cloud-Computing-Spring-2025/handson-10-machine-learning-with-mllib-Tejith2.git
cd handson-10-machine-learning-with-mllib-Tejith2


#output
INFO RandomForest:   init: 1.0959E-5
  total: 0.061604083
  findBestSplits: 0.061474041
  chooseSplits: 0.061402375
   INFO GradientBoostedTrees:   building tree 4: 0.062196833
  buildMetadata: 0.029402
  init: 1.5833E-5
  total: 0.707777667
  building tree 3: 0.062508834
  building tree 6: 0.064000709
  building tree 0: 0.119181958
  building tree 9: 0.068224667
  building tree 8: 0.066792292
  building tree 2: 0.062448541
  building tree 5: 0.063938875
  findSplits: 0.019418709
  building tree 7: 0.064843916
  building tree 1: 0.061432792

