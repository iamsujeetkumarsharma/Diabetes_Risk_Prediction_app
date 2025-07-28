

# === Imports ===
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objs as go
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# === Custom CSS for premium look ===
st.markdown('''
<style>
.main {
    background: linear-gradient(135deg, #232b5c 0%, #3a4668 100%) !important;
}
.stApp {
    background: #f7fafd;
}
.stButton>button {
    background: linear-gradient(90deg, #232b5c 0%, #3a4668 100%);
    color: #fff;
    font-size: 22px;
    font-weight: bold;
    border-radius: 12px;
    padding: 0.7em 2.5em;
    margin: 0.5em auto 0.5em auto;
    display: block;
    transition: 0.2s;
    box-shadow: 0 4px 16px rgba(44,62,80,0.08);
}
.stButton>button:hover {
    background: linear-gradient(90deg, #3a4668 0%, #232b5c 100%);
    color: #ffd700;
}
.stNumberInput>div>input, .stSlider>div {
    border-radius: 8px;
}
.result-low {
    color: #2e7d32;
    font-weight: bold;
    font-size: 1.25em;
    display: inline-block;
    background: #e8f5e9;
    border-radius: 8px;
    padding: 0.3em 1em;
    margin: 0.5em 0;
    box-shadow: 0 2px 8px rgba(46,125,50,0.08);
    letter-spacing: 0.5px;
}
.result-high {
    color: #c62828;
    font-weight: bold;
    font-size: 1.25em;
    display: inline-block;
    background: #ffebee;
    border-radius: 8px;
    padding: 0.3em 1em;
    margin: 0.5em 0;
    box-shadow: 0 2px 8px rgba(198,40,40,0.08);
    letter-spacing: 0.5px;
}
.subtitle {
    font-size: 1.2em;
    color: #3a4668;
    margin-bottom: 0.5em;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 0.5rem !important;
}
</style>
''', unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966484.png", width=80)
st.sidebar.title("🧠 AI-Powered Diabetes Risk Predictor")
st.sidebar.markdown("""
**Project Description**

This app predicts diabetes risk using a Random Forest model trained on the PIMA dataset. Enter your health data to get a personalized risk assessment.

**Data Source**
- [PIMA Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)

**Developer**
- Sujeet Kumar  
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/iamsujeetsharma)
""")




# === Model Loading and Caching ===
@st.cache_resource
def load_models_and_scaler():
    rf_model = joblib.load('diabetes_model.pkl')
    lr_model = joblib.load('logreg_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf_model, lr_model, scaler

rf_model, lr_model, scaler = load_models_and_scaler()
model = rf_model  # Default model for prediction UI

@st.cache_resource
def get_shap_explainer_and_values(_model, feature_names):
    explainer = shap.TreeExplainer(_model)
    X_bg = np.zeros((5, len(feature_names)))  # smaller sample for speed
    shap_values = explainer.shap_values(X_bg)
    return explainer, shap_values, X_bg


# === Feature Metadata ===
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
defaults = [1, 120, 70, 20, 80, 25.0, 0.5, 33]
mins = [0, 0, 0, 0, 0, 10.0, 0.1, 10]
maxs = [20, 200, 140, 99, 846, 67.1, 2.5, 100]


# === Main App Layout ===
with st.container():
    st.markdown('<h1 style="text-align:center;font-size:2.8em;font-weight:900;color:#232b5c;">🩺 AI-Powered Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="text-align:center;">Powered by Machine Learning & Real Medical Data</div>', unsafe_allow_html=True)


tabs = st.tabs(["📊 Prediction Form", "📈 Model Comparison", "🧬 Model Explanation", "ℹ️ About"])
with tabs[1]:
    st.markdown('### 📈 Model Comparison')
    # --- Load and preprocess data for metrics ---
    data_url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
    df = pd.read_csv(data_url)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, df[col].mean())
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    from sklearn.preprocessing import StandardScaler
    scaler_eval = StandardScaler()
    X_scaled = scaler_eval.fit_transform(X)

    # --- Model metrics ---
    models = [(rf_model, "Random Forest"), (lr_model, "Logistic Regression")]
    metrics = []
    for m, name in models:
        y_pred = m.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        metrics.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec})
    st.dataframe(pd.DataFrame(metrics).set_index("Model"), use_container_width=True)

    # --- Confusion Matrices ---
    st.markdown('#### Confusion Matrices')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (m, name) in enumerate(models):
        cm = confusion_matrix(y, m.predict(X_scaled))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)

    # --- Classification Reports ---
    st.markdown('#### Classification Reports')
    for m, name in models:
        st.markdown(f'**{name}**')
        report = classification_report(y, m.predict(X_scaled), output_dict=False)
        st.code(report)

with tabs[0]:
    # --- Input Form ---
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input('Pregnancies', min_value=mins[0], max_value=maxs[0], value=defaults[0], step=1, key='preg')
        glucose = st.slider('Glucose Level', min_value=mins[1], max_value=maxs[1], value=defaults[1], key='gluc')
        bp = st.slider('Blood Pressure', min_value=mins[2], max_value=maxs[2], value=defaults[2], key='bp')
        skin = st.number_input('Skin Thickness', min_value=mins[3], max_value=maxs[3], value=defaults[3], step=1, key='skin')
    with col2:
        insulin = st.number_input('Insulin Level', min_value=mins[4], max_value=maxs[4], value=defaults[4], step=1, key='insulin')
        bmi = st.number_input('BMI', min_value=mins[5], max_value=maxs[5], value=defaults[5], step=0.1, key='bmi')
        dpf = st.number_input('Diabetes Pedigree Function', min_value=mins[6], max_value=maxs[6], value=defaults[6], step=0.01, key='dpf')
        age = st.slider('Age', min_value=mins[7], max_value=maxs[7], value=defaults[7], key='age')

    user_input = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]

    # --- Prediction Button and Result ---
    predict_btn = st.button('🧠 Predict Risk', use_container_width=True)
    if predict_btn:
        with st.spinner('Analyzing your data...'):
            X_df = pd.DataFrame([user_input], columns=feature_names)
            X_scaled = scaler.transform(X_df)
            proba = model.predict_proba(X_scaled)[0, 1]
            pred = model.predict(X_scaled)[0]
            st.markdown('<div>', unsafe_allow_html=True)
            if pred == 1:
                st.markdown('<div class="result-high">🔴 High Diabetes Risk Detected!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-low">🟢 Low Diabetes Risk Detected!</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:1.3em;">Probability: <b>{proba*100:.1f}%</b></div>', unsafe_allow_html=True)
            st.progress(proba)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Feature Importance Plot ---
        st.markdown('#### 📊 Feature Importance')
        importances = model.feature_importances_
        fig = go.Figure([go.Bar(x=feature_names, y=importances, marker_color='#232b5c')])
        fig.update_layout(title='Feature Importance', xaxis_title='Feature', yaxis_title='Importance', plot_bgcolor='#fff')
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    # --- Model Explanation (SHAP) ---
    st.markdown('### 🧬 Model Explanation (SHAP)')
    with st.expander('Show SHAP Summary Plot'):
        try:
            explainer, shap_values, X_bg = get_shap_explainer_and_values(model, feature_names)
            fig, ax = plt.subplots(figsize=(8, 3))
            shap.summary_plot(shap_values, X_bg, feature_names=feature_names, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP plot unavailable: {e}")

with tabs[3]:
    # --- About Tab ---
    st.markdown('''
    ### ℹ️ About This App
    - **Purpose:** Predict diabetes risk using AI and medical data.
    - **Model:** Random Forest Classifier
    - **Data:** PIMA Indian Diabetes Dataset
    - **Developer:** Sujeet Kumar
    - **Contact:** [LinkedIn](https://www.linkedin.com/in/your-linkedin-profile)
    ''')