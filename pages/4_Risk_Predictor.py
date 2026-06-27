import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Risk Predictor", page_icon="🤖", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/brfss_ace_clean_2023.csv")

@st.cache_resource
def train_model(df):
    # Features and target
    features = ["ACE_SCORE", "SEXVAR", "_AGE_G", "_BMI5CAT", 
                "SMOKE100", "DRINKS_ANY", "INCOME3", "EDUCA"]
    target = "ADDEPEV3"

    # Drop rows with missing values in relevant columns
    model_df = df[features + [target]].dropna()

    # Encode categorical variables
    le = LabelEncoder()
    for col in features:
        if model_df[col].dtype == "object":
            model_df[col] = le.fit_transform(model_df[col].astype(str))
    model_df[target] = le.fit_transform(model_df[target].astype(str))

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    accuracy = rf.score(X_test, y_test)
    y_pred = rf.predict(X_test)

    return rf, accuracy, X_test, y_test, y_pred, features, model_df

df = load_data()

st.markdown("<h2 style='color:#4A0E3A;'>🤖 Depression Risk Predictor</h2>", unsafe_allow_html=True)
st.markdown("This Random Forest model predicts depression risk based on ACE score and demographic/lifestyle factors.")
st.markdown("---")

# Train model
with st.spinner("Training model..."):
    rf, accuracy, X_test, y_test, y_pred, features, model_df = train_model(df)

# Model metrics
st.markdown("### Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "Random Forest")
col2.metric("Accuracy", f"{accuracy*100:.1f}%")
col3.metric("Features", str(len(features)))
col4.metric("Training Samples", f"{len(model_df):,}")

st.markdown("---")

# Row 1: Feature importance + Confusion matrix
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": ["ACE Score", "Gender", "Age Group", "BMI",
                   "Smoking", "Alcohol Use", "Income", "Education"],
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig1 = px.bar(
        importance_df, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#F2C4D0", "#9B2D6F", "#4A0E3A"]
    )
    fig1.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2 = px.imshow(
        cm,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        x=["No Depression", "Depression"],
        y=["No Depression", "Depression"],
        color_continuous_scale=["#F2C4D0", "#4A0E3A"],
        text_auto=True
    )
    fig2.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Row 2: Individual predictor
st.markdown("### 🔍 Predict Individual Depression Risk")
st.markdown("Fill in the profile below to predict depression risk:")

col1, col2, col3 = st.columns(3)

with col1:
    ace_score = st.slider("ACE Score (0-13)", 0, 13, 2)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age_group = st.selectbox("Age Group", ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])

with col2:
    bmi = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    smoking = st.selectbox("Ever Smoked 100 Cigarettes?", ["Yes", "No"])
    alcohol = st.selectbox("Drinks Alcohol?", ["Yes", "No"])

with col3:
    income = st.selectbox("Income Level", ["<$15k", "$15-25k", "$25-35k", "$35-50k", 
                                            "$50-100k", "$75-100k", "$100-200k", 
                                            "$150-200k", "$200k+"])
    education = st.selectbox("Education Level", ["No School", "Elementary", 
                                                   "Some High School", 
                                                   "High School Graduate",
                                                   "Some College", 
                                                   "College Graduate"])

if st.button("🔮 Predict Depression Risk", use_container_width=True):
    # Encode inputs the same way as training
    input_df = pd.DataFrame([{
        "ACE_SCORE": ace_score,
        "SEXVAR": gender,
        "_AGE_G": age_group,
        "_BMI5CAT": bmi,
        "SMOKE100": smoking,
        "DRINKS_ANY": alcohol,
        "INCOME3": income,
        "EDUCA": education
    }])

    # Encode categoricals
    le = LabelEncoder()
    for col in input_df.columns:
        if input_df[col].dtype == "object":
            # Fit on training data values
            all_vals = model_df[col].astype(str).unique()
            le.fit(all_vals)
            input_df[col] = le.transform(input_df[col].astype(str))

    # Predict
    pred = rf.predict(input_df)[0]
    prob = rf.predict_proba(input_df)[0]
    depression_prob = prob[1] * 100

    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if pred == 1:
            st.error(f"⚠️ **High Depression Risk Detected**")
            st.markdown(f"This profile has a **{depression_prob:.1f}%** predicted probability of depression.")
        else:
            st.success(f"✅ **Low Depression Risk**")
            st.markdown(f"This profile has a **{depression_prob:.1f}%** predicted probability of depression.")

        # Risk gauge
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=depression_prob,
            title={"text": "Depression Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4A0E3A"},
                "steps": [
                    {"range": [0, 30], "color": "#F2C4D0"},
                    {"range": [30, 60], "color": "#D4739A"},
                    {"range": [60, 100], "color": "#4A0E3A"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": depression_prob
                }
            }
        ))
        st.plotly_chart(fig3, use_container_width=True)
