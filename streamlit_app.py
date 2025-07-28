
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objs as go
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Custom CSS for premium look
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
    margin: 1em auto;
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
.card {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(44,62,80,0.10);
    padding: 2em;
    margin: 2em auto;
    max-width: 500px;
    text-align: center;
}
.result-low {color: #2e7d32; font-weight: bold;}
.result-high {color: #c62828; font-weight: bold;}
.subtitle {font-size: 1.2em; color: #3a4668; margin-bottom: 1em;}
</style>
''', unsafe_allow_html=True)

# Sidebar
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



# Load models and scaler
@st.cache_resource
def load_models_and_scaler():
    rf_model = joblib.load('diabetes_model.pkl')
    lr_model = joblib.load('logreg_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf_model, lr_model, scaler
rf_model, lr_model, scaler = load_models_and_scaler()

# Use Random Forest as default for prediction UI
model = rf_model

# Cache SHAP explainer and values
@st.cache_resource
def get_shap_explainer_and_values(model, feature_names):
    explainer = shap.TreeExplainer(model)
    X_bg = np.zeros((5, len(feature_names)))  # smaller sample for speed
    shap_values = explainer.shap_values(X_bg)
    return explainer, shap_values, X_bg

feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
defaults = [1, 120, 70, 20, 80, 25.0, 0.5, 33]
mins = [0, 0, 0, 0, 0, 10.0, 0.1, 10]
maxs = [20, 200, 140, 99, 846, 67.1, 2.5, 100]

with st.container():
    st.markdown('<h1 style="text-align:center;font-size:2.8em;font-weight:900;color:#232b5c;">🩺 AI-Powered Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="text-align:center;">Powered by Machine Learning & Real Medical Data</div>', unsafe_allow_html=True)

tabs = st.tabs(["📊 Prediction Form", "📈 Model Comparison", "🧬 Model Explanation", "ℹ️ About"])
with tabs[1]:
    st.markdown('### 📈 Model Comparison')
    # Load data for metrics
    import pandas as pd
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

    models = [(rf_model, "Random Forest"), (lr_model, "Logistic Regression")]
    metrics = []
    for m, name in models:
        y_pred = m.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        metrics.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec})

    st.dataframe(pd.DataFrame(metrics).set_index("Model"), use_container_width=True)

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

    st.markdown('#### Classification Reports')
    for m, name in models:
        st.markdown(f'**{name}**')
        report = classification_report(y, m.predict(X_scaled), output_dict=False)
        st.code(report)

with tabs[0]:
    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input('Pregnancies', min_value=mins[0], max_value=maxs[0], value=defaults[0], step=1)
        glucose = st.slider('Glucose Level', min_value=mins[1], max_value=maxs[1], value=defaults[1])
        bp = st.slider('Blood Pressure', min_value=mins[2], max_value=maxs[2], value=defaults[2])
        skin = st.number_input('Skin Thickness', min_value=mins[3], max_value=maxs[3], value=defaults[3], step=1)
    with col2:
        insulin = st.number_input('Insulin Level', min_value=mins[4], max_value=maxs[4], value=defaults[4], step=1)
        bmi = st.number_input('BMI', min_value=mins[5], max_value=maxs[5], value=defaults[5], step=0.1)
        dpf = st.number_input('Diabetes Pedigree Function', min_value=mins[6], max_value=maxs[6], value=defaults[6], step=0.01)
        age = st.slider('Age', min_value=mins[7], max_value=maxs[7], value=defaults[7])

    user_input = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]

    st.markdown('---')
    predict_btn = st.button('🧠 Predict Risk', use_container_width=True)
    if predict_btn:
        with st.spinner('Analyzing your data...'):
            import pandas as pd
            X_df = pd.DataFrame([user_input], columns=feature_names)
            X_scaled = scaler.transform(X_df)
            proba = model.predict_proba(X_scaled)[0, 1]
            pred = model.predict(X_scaled)[0]
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if pred == 1:
                st.markdown('<span class="result-high">🔴 High Diabetes Risk Detected!</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="result-low">🟢 Low Diabetes Risk Detected!</span>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:1.3em;">Probability: <b>{proba*100:.1f}%</b></div>', unsafe_allow_html=True)
            st.progress(proba)
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature importance plot (move outside spinner)
        st.markdown('#### 📊 Feature Importance')
        importances = model.feature_importances_
        fig = go.Figure([go.Bar(x=feature_names, y=importances, marker_color='#232b5c')])
        fig.update_layout(title='Feature Importance', xaxis_title='Feature', yaxis_title='Importance', plot_bgcolor='#fff')
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.markdown('### 🧬 Model Explanation (SHAP)')
    with st.expander('Show SHAP Summary Plot'):
        try:
            explainer, shap_values, X_bg = get_shap_explainer_and_values(model, feature_names)
            fig, ax = plt.subplots(figsize=(8, 3))
            shap.summary_plot(shap_values, X_bg, feature_names=feature_names, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP plot unavailable: {e}")

with tabs[2]:
    st.markdown('''
    ### ℹ️ About This App
    - **Purpose:** Predict diabetes risk using AI and medical data.
    - **Model:** Random Forest Classifier
    - **Data:** PIMA Indian Diabetes Dataset
    - **Developer:** Sujeet Kumar
    - **Contact:** [LinkedIn](https://www.linkedin.com/in/your-linkedin-profile)
    ''')