import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
data_url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
df = pd.read_csv(data_url)

# Step 2: Preprocess data
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].mean())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 3: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_scaled, y)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_scaled, y)

# Step 5: Evaluate models
def print_metrics(name, y_true, y_pred):
    print(f"\n{name} Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred):.3f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))

print_metrics("Random Forest", y, rf_model.predict(X_scaled))
print_metrics("Logistic Regression", y, lr_model.predict(X_scaled))

# Step 6: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, (model, name) in enumerate(zip([rf_model, lr_model], ["Random Forest", "Logistic Regression"])):
    cm = confusion_matrix(y, model.predict(X_scaled))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{name} Confusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('model_confusion_matrices.png')
plt.close()

# Step 7: Save models and scaler
joblib.dump(rf_model, 'diabetes_model.pkl')
joblib.dump(lr_model, 'logreg_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print('Models and scaler saved successfully. Confusion matrices saved as model_confusion_matrices.png.')
