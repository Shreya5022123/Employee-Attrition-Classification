 **Employee Attrition Prediction**

This project builds a powerful, production-ready Employee Attrition Prediction System using advanced machine learning techniques.
It uses SMOTE, feature engineering, cross-validation, and a stacked ensemble model to accurately classify whether an employee is likely to leave the company.

**Features of This Project**
 1. HR Attrition Dataset (IBM HR Analytics Dataset)
 2. Data preprocessing & categorical encoding
 3. Feature selection + domain-based engineered features
 4. Handling imbalanced data using SMOTE
 5. Standardization with StandardScaler
 6. Multiple ML Models:

    GradientBoostingClassifier

    XGBoostClassifier

    Stacking Ensemble (RF + SVC + LR → XGBoost Meta Learner)
    
  8.  5-Fold Cross-Validation using F1-Score
  9.  Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
  10.  Saves the best model + scaler as .pkl
  11. Ready for deployment (Streamlit)

**Dataset Source**

This project uses the IBM HR Analytics Employee Attrition Dataset, a widely used open-source dataset for HR analytics and attrition prediction tasks.
Dataset Name:IBM HR Analytics Employee Attrition & Performance Dataset

This dataset was originally published by IBM as part of an HR analytics case study and later made publicly available on:
**Kaggle:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
It is also referenced across multiple academic HR research papers and machine learning tutorials.
**Dataset Description**

Total records: 1,470 employees (your processed version ~1,400 after cleaning)

Target variable: Attrition (Yes/No)

Contains demographic, behavioral, performance, and job-related fields.

_This dataset is completely open-source and can be freely used for educational and research purposes._

**Model Performance Summary**

Gradient Boosting Classifier achieved an accuracy of 0.9089, precision of 0.9243, recall of 0.8906, F1-score of 0.9072, and ROC-AUC of 0.9089.
XGBoost Classifier delivered the strongest results, with an accuracy of 0.9271, precision of 0.9451, recall of 0.9068, F1-score of 0.9256, and ROC-AUC of 0.9271.
Stacking Classifier performed competitively with an accuracy of 0.9392, precision of 0.9428, recall of 0.9352, F1-score of 0.9390, and ROC-AUC of 0.9392.

**Gui Results**

<img width="940" height="448" alt="image" src="https://github.com/user-attachments/assets/9eb348ab-1d11-4a15-9fcc-efbb2e147aaa" />

                                                         User interface

<img width="940" height="496" alt="image" src="https://github.com/user-attachments/assets/e1231a04-18d6-40c8-b3cb-c00a770118de" />

                                                        Predicting attrition



