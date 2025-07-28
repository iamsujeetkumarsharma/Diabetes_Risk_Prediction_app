# Diabetes Risk Prediction

## Problem Statement
Early detection of diabetes risk is crucial for timely intervention and improved patient outcomes. This project builds a robust, explainable machine learning pipeline to predict diabetes risk using the PIMA Indian Diabetes Dataset.

## Dataset
- **Source:** [PIMA Indian Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)
- **Description:** Medical diagnostic data for female patients of Pima Indian heritage, aged 21 and above.

## Project Features
- End-to-end pipeline: data loading, cleaning, EDA, model training, evaluation, explainability, and deployment.
- Models: Logistic Regression and Random Forest.
- Explainability: SHAP visualizations for feature impact.
- Interactive Streamlit app for real-time risk prediction.

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the main pipeline:**
   ```bash
   python diabetes_risk_prediction.py
   ```
4. **Launch the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Model Insights
- **Best Model:** Random Forest (selected based on F1-score)
- **Key Features:** Glucose, BMI, Age (as per SHAP analysis)
- **Performance:**
  - Accuracy, Precision, Recall, F1-Score reported for both models
  - Confusion matrix and metric comparison visualized

## SHAP & Evaluation Visuals
- ![SHAP Summary Plot](screenshots/shap_summary.png)
- ![Confusion Matrix](screenshots/confusion_matrix.png)
- ![Metrics Comparison](screenshots/metrics_comparison.png)

## Business/Healthcare Relevance
Early risk prediction enables:
- Proactive lifestyle interventions
- Reduced healthcare costs
- Improved patient quality of life

## Usage
- Use the Streamlit app to input patient data and receive instant risk predictions and probability.
- Model and scaler are saved as `diabetes_model.pkl` and `scaler.pkl` for deployment.

## Authors
- [Sujeet Kumar]
