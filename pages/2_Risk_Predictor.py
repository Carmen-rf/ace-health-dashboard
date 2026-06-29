import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Risk Predictor", page_icon="🤖", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/brfss_ace_clean_2023.csv")

@st.cache_resource
def train_model(_df):
    features = ["ACE_SCORE","SEXVAR","_AGE_G","_BMI5CAT",
                "SMOKE100","DRINKS_ANY","INCOME3","EDUCA"]
    target = "ADDEPEV3"
    model_df = _df[features+[target]].copy().dropna()
    model_df_raw = model_df.copy()
    encoders = {}
    for col in features+[target]:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le
    X = np.array(model_df[features].values.tolist(), dtype=float)
    y = np.array(model_df[target].values.tolist(), dtype=int)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    rf = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
    rf.fit(X_train,y_train)
    accuracy = rf.score(X_test,y_test)
    y_pred = rf.predict(X_test)
    return rf,accuracy,X_test,y_test,y_pred,features,encoders,model_df_raw

df = load_data()
st.markdown("<h2 style='color:#5B2C6F;'>🤖 Depression Risk Predictor</h2>", unsafe_allow_html=True)
st.markdown("This Random Forest model predicts depression risk based on ACE score and demographic/lifestyle factors.")
st.markdown("---")

with st.spinner("Training model..."):
    rf,accuracy,X_test,y_test,y_pred,features,encoders,model_df_raw = train_model(df)

st.markdown("### Model Performance")
col1,col2,col3,col4 = st.columns(4)
col1.metric("Model","Random Forest")
col2.metric("Accuracy",f"{accuracy*100:.1f}%")
col3.metric("Features",str(len(features)))
col4.metric("Training Samples",f"{len(model_df_raw):,}")
st.markdown("---")

col1,col2 = st.columns(2)
with col1:
    st.markdown("#### Feature Importance")
    importance_df = pd.DataFrame({
        "Feature":["ACE Score","Gender","Age Group","BMI","Smoking","Alcohol Use","Income","Education"],
        "Importance":rf.feature_importances_
    }).sort_values("Importance",ascending=True)
    fig1 = px.bar(importance_df,x="Importance",y="Feature",orientation="h",
        color="Feature",
        color_discrete_sequence=["#5B2C6F","#1ABC9C","#E74C3C","#2E86C1","#E67E22","#27AE60","#F1C40F","#8E44AD"])
    fig1.update_layout(plot_bgcolor="white",showlegend=False)
    st.plotly_chart(fig1,use_container_width=True)

with col2:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test,y_pred)
    fig2 = px.imshow(cm,
        labels={"x":"Predicted","y":"Actual","color":"Count"},
        x=["No Depression","Depression"],
        y=["No Depression","Depression"],
        color_continuous_scale=["#AED6F1","#1A5276"],
        text_auto=True)
    fig2.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig2,use_container_width=True)

st.markdown("---")
st.markdown("### 🔍 Predict Individual Depression Risk")
col1,col2,col3 = st.columns(3)

with col1:
    ace_score = st.slider("ACE Score (0-13)",0,13,2)
    gender = st.selectbox("Gender",sorted(model_df_raw["SEXVAR"].unique()))
    age_group = st.selectbox("Age Group",["18-24","25-34","35-44","45-54","55-64","65+"])

with col2:
    bmi = st.selectbox("BMI Category",["Underweight","Normal","Overweight","Obese"])
    smoking = st.selectbox("Ever Smoked?",sorted(model_df_raw["SMOKE100"].unique()))
    alcohol = st.selectbox("Drinks Alcohol?",sorted(model_df_raw["DRINKS_ANY"].unique()))

with col3:
    income = st.selectbox("Income Level",sorted(model_df_raw["INCOME3"].unique()))
    education = st.selectbox("Education Level",["No School","Elementary",
                             "Some High School","High School Graduate",
                             "Some College","College Graduate"])

if st.button("🔮 Predict Depression Risk",use_container_width=True):
    input_dict = {"ACE_SCORE":ace_score,"SEXVAR":gender,"_AGE_G":age_group,
                  "_BMI5CAT":bmi,"SMOKE100":smoking,"DRINKS_ANY":alcohol,
                  "INCOME3":income,"EDUCA":education}
    input_encoded = []
    for col in features:
        val = input_dict[col]
        if col in encoders:
            try: val = encoders[col].transform([str(val)])[0]
            except: val = 0
        input_encoded.append(float(val))
    input_array = np.array(input_encoded,dtype=float).reshape(1,-1)
    pred = rf.predict(input_array)[0]
    prob = rf.predict_proba(input_array)[0]
    depression_prob = prob[1]*100
    st.markdown("---")
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        if pred==1:
            st.error("⚠️ **High Depression Risk Detected**")
        else:
            st.success("✅ **Low Depression Risk**")
        st.markdown(f"Predicted probability of depression: **{depression_prob:.1f}%**")
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=depression_prob,
            title={"text":"Depression Risk %"},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#5B2C6F"},
                   "steps":[
                       {"range":[0,30],"color":"#A9DFBF"},
                       {"range":[30,60],"color":"#F9E79F"},
                       {"range":[60,100],"color":"#F1948A"}]}))
        st.plotly_chart(fig3,use_container_width=True)
