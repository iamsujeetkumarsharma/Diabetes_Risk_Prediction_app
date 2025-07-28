"""
Diabetes Risk Prediction using PIMA Indian Diabetes Dataset
Author: [Your Name]
Date: [Today's Date]
Description: End-to-end machine learning pipeline for early diabetes risk prediction.
"""

# 1. Project Setup: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# 2. Dataset Loading
def load_data(url: str) -> pd.DataFrame:
    """Load the PIMA Indian Diabetes Dataset from a URL."""
    df = pd.read_csv(url)
    return df

def explore_data(df: pd.DataFrame):
    """Perform data exploration and visualization."""
    print("\nDataset Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nNull Values:\n", df.isnull().sum())
    
    # Histograms
    df.hist(bins=20, figsize=(15, 10))
    plt.suptitle('Feature Distributions', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Pair Plot
    sns.pairplot(df, hue='Outcome', diag_kind='hist')
    plt.suptitle('Pair Plot by Outcome', y=1.02)
    plt.show()
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
    
    # Class Imbalance Visualization
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Outcome', data=df)
    plt.title('Diabetes Outcome Count')
    plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
    plt.ylabel('Count')
    plt.show()
    print("\nClass Distribution:\n", df['Outcome'].value_counts())

def clean_and_preprocess(df: pd.DataFrame):
    """Clean and preprocess the dataset: impute invalid zeroes, normalize, split X/y."""
    # Columns where zero is invalid
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_clean = df.copy()
    df_clean[zero_invalid_cols] = df_clean[zero_invalid_cols].replace(0, np.nan)
    print("\nMissing values after replacing zeros:")
    print(df_clean.isnull().sum())
    # Impute missing values with column mean
    df_clean.fillna(df_clean.mean(), inplace=True)
    print("\nMissing values after imputation:")
    print(df_clean.isnull().sum())
    # Split features and label
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\nFeature means after scaling (should be ~0):\n", X_scaled.mean(axis=0))
    return X_scaled, y, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train, X_test):
    """Train Logistic Regression and predict on test set."""
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    return lr, y_pred, y_proba

def train_random_forest(X_train, y_train, X_test):
    """Train Random Forest and predict on test set."""
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    return rf, y_pred, y_proba

def evaluate_model(y_test, y_pred, y_proba, model_name):
    """Evaluate model and return metrics and report."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"\n{model_name} Classification Report:\n", report)
    return {'model': model_name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'cm': cm, 'report': report}

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_metrics_comparison(metrics_list):
    labels = [m['model'] for m in metrics_list]
    accuracy = [m['accuracy'] for m in metrics_list]
    precision = [m['precision'] for m in metrics_list]
    recall = [m['recall'] for m in metrics_list]
    f1 = [m['f1'] for m in metrics_list]
    x = np.arange(len(labels))
    width = 0.2
    plt.figure(figsize=(10,6))
    plt.bar(x - 1.5*width, accuracy, width, label='Accuracy')
    plt.bar(x - 0.5*width, precision, width, label='Precision')
    plt.bar(x + 0.5*width, recall, width, label='Recall')
    plt.bar(x + 1.5*width, f1, width, label='F1-Score')
    plt.xticks(x, labels)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('Model Metrics Comparison')
    plt.legend()
    plt.show()

def compare_models(metrics_list):
    """Create a summary table comparing models and select the better one."""
    summary_df = pd.DataFrame(metrics_list)[['model', 'accuracy', 'precision', 'recall', 'f1']]
    print("\nModel Comparison Summary:")
    print(summary_df.to_string(index=False))
    # Select better model by F1-score (or accuracy if tie)
    best_idx = summary_df['f1'].idxmax()
    best_model = summary_df.loc[best_idx, 'model']
    print(f"\nSelected model for deployment: {best_model}")
    return summary_df, best_model

def plot_feature_importance(rf_model, feature_names):
    """Plot feature importances from Random Forest model."""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10,6))
    plt.barh(range(len(importances)), importances[indices], align='center')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importances')
    plt.tight_layout()
    plt.show()

def shap_explainability(rf_model, X_train, X_test, feature_names):
    """Explain Random Forest predictions using SHAP."""
    print("\nCalculating SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    # SHAP Summary Plot
    shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=True)
    # SHAP Bar Plot
    shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, plot_type='bar', show=True)
    # Comment on most influential features
    mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
    top_features = np.array(feature_names)[np.argsort(mean_abs_shap)[::-1][:3]]
    print(f"\nTop 3 most influential features (SHAP): {', '.join(top_features)}")

def export_model_and_scaler(model, scaler, model_path='diabetes_model.pkl', scaler_path='scaler.pkl'):
    """Save the trained model and scaler to disk."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def main():
    # Dataset URL
    DATA_URL = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
    
    # Load dataset
    df = load_data(DATA_URL)
    
    # Display first 5 rows and summary
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    # Data Exploration & Visualization
    explore_data(df)
    # Data Cleaning & Preprocessing
    X_scaled, y, scaler = clean_and_preprocess(df)
    # Train-Test Split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    # Model Building
    print("\nTraining Logistic Regression...")
    lr_model, lr_pred, lr_proba = train_logistic_regression(X_train, y_train, X_test)
    print("Training Random Forest...")
    rf_model, rf_pred, rf_proba = train_random_forest(X_train, y_train, X_test)
    # Model Evaluation
    lr_metrics = evaluate_model(y_test, lr_pred, lr_proba, 'Logistic Regression')
    rf_metrics = evaluate_model(y_test, rf_pred, rf_proba, 'Random Forest')
    plot_confusion_matrix(lr_metrics['cm'], 'Logistic Regression')
    plot_confusion_matrix(rf_metrics['cm'], 'Random Forest')
    plot_metrics_comparison([lr_metrics, rf_metrics])
    # Model Comparison
    summary_df, best_model = compare_models([lr_metrics, rf_metrics])
    # Feature Importance (Random Forest)
    feature_names = df.drop('Outcome', axis=1).columns.tolist()
    plot_feature_importance(rf_model, feature_names)
    # SHAP Explainability
    shap_explainability(rf_model, X_train, X_test, feature_names)
    # Export Trained Model and Scaler
    export_model_and_scaler(rf_model, scaler)
    # For next steps: deployment or Streamlit app

if __name__ == "__main__":
    main() 