# 🧠 Customer Churn Prediction using Machine Learning

A complete end-to-end **Customer Churn Prediction** project that analyzes customer data, identifies churn patterns, and builds machine learning models to predict whether a customer will leave the service.  

The project compares multiple ML algorithms — **Decision Tree, Random Forest, XGBoost, and CatBoost** — and fine-tunes them to find the best-performing model for churn prediction.

---

## 🚀 Project Overview

Customer churn is a critical problem for subscription-based businesses.  
This project uses real-world customer data to:
- Understand why customers churn.  
- Visualize key business insights.  
- Build predictive models that accurately classify churn vs. non-churn customers.  
- Recommend strategies to improve customer retention.

---

## 🧩 Project Workflow

### **Step 1: Data Loading**
- Imported the dataset (`customerchurndataset.csv`) using pandas.  
- Previewed and understood structure, shape, and basic statistics.

### **Step 2: Data Cleaning**
- Removed unnecessary columns like `customerID`.  
- Fixed missing or blank values in `TotalCharges`.  
- Converted datatypes to numerical where required.

### **Step 3: Feature Encoding**
- Converted categorical variables into numeric form using `LabelEncoder`.  
- Saved encoders in a separate file (`encoders.pkl`) for reuse.

### **Step 4: Balancing the Data**
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the churn classes.  
- Ensured equal distribution of churn (1) and non-churn (0) samples.

### **Step 5: Model Training & Comparison**
- Trained and compared three models:
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- Evaluated each using **cross-validation accuracy**.

### **Step 6: Hyperparameter Tuning**
- Optimized Random Forest using **GridSearchCV** for better accuracy and generalization.

### **Step 7: CatBoost Final Model & ROC-AUC**
- Trained a **CatBoost Classifier**, which achieved the best overall performance.  
- Calculated ROC-AUC to measure the model’s discriminative ability.

### **Step 8: Save the Final Model**
- Exported the final CatBoost model and feature names using **pickle** (`customer_churn_catboost.pkl`).  
- Ready for deployment and prediction on new data.

---

## 📊 Results

| Model | Accuracy | ROC-AUC | Remark |
|--------|-----------|----------|--------|
| Decision Tree | 0.78 | 0.82 | Baseline Model |
| Random Forest | 0.78 | 0.83 | Stable and Robust |
| XGBoost | 0.78 | 0.83 | Slightly Better but Complex |
| **CatBoost** | **0.79** | **0.84** | ✅ Best Model |

> CatBoost achieved the highest **accuracy (0.79)** and **ROC-AUC (0.84)**, making it the final chosen model.

---

## 📦 Files and Structure

