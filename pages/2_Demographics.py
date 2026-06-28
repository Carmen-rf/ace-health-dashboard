import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Demographics", page_icon="👤", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/brfss_ace_clean_2023.csv")

df = load_data()

st.markdown("<h2 style='color:#4A0E3A;'>👤 Demographic Profile of ACE Exposure</h2>", unsafe_allow_html=True)
st.markdown("Explore how ACE exposure varies across gender, age, income, education and BMI.")
st.markdown("---")

st.sidebar.header("Filters")
gender = st.sidebar.multiselect("Gender", options=df["SEXVAR"].dropna().unique(), default=df["SEXVAR"].dropna().unique())
age = st.sidebar.multiselect("Age Group", options=df["_AGE_G"].dropna().unique(), default=df["_AGE_G"].dropna().unique())
ace_cat = st.sidebar.multiselect("ACE Category", options=df["ACE_CATEGORY"].dropna().unique(), default=df["ACE_CATEGORY"].dropna().unique())

filtered = df[
    df["SEXVAR"].isin(gender) &
    df["_AGE_G"].isin(age) &
    df["ACE_CATEGORY"].isin(ace_cat)
]

st.markdown("### Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Respondents", f"{len(filtered):,}")
col2.metric("% Female", f"{(filtered['SEXVAR'] == 'Female').mean()*100:.1f}%")
col3.metric("Mean ACE Score", f"{filtered['ACE_SCORE'].mean():.2f}")
col4.metric("Most Common Age Group", filtered["_AGE_G"].mode()[0])

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Mean ACE Score by Gender")
    gender_ace = filtered.groupby("SEXVAR")["ACE_SCORE"].mean().reset_index()
    gender_ace.columns = ["Gender", "Mean ACE Score"]
    fig1 = px.bar(gender_ace, x="Gender", y="Mean ACE Score",
        color="Gender",
        color_discrete_sequence=["#4A0E3A", "#D4739A"],
        text="Mean ACE Score")
    fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig1.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### Mean ACE Score by Age Group")
    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_ace = filtered.groupby("_AGE_G")["ACE_SCORE"].mean().reindex(age_order).reset_index()
    age_ace.columns = ["Age Group", "Mean ACE Score"]
    fig2 = px.line(age_ace, x="Age Group", y="Mean ACE Score",
        markers=True, color_discrete_sequence=["#9B2D6F"])
    fig2.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### % High ACE Exposure (4+) by Income Level")
    income_order = ["<$15k", "$15-25k", "$25-35k", "$35-50k",
                    "$50-100k", "$75-100k", "$100-200k", "$150-200k", "$200k+"]
    income_pct = filtered[filtered["INCOME3"].isin(income_order)].groupby("INCOME3").apply(
        lambda x: (x["ACE_CATEGORY"] == "High (4+)").mean() * 100
    ).reindex(income_order).reset_index()
    income_pct.columns = ["Income Level", "% High ACE (4+)"]
    fig3 = px.bar(income_pct, x="Income Level", y="% High ACE (4+)",
        color="% High ACE (4+)",
        color_continuous_scale=["#F2C4D0", "#9B2D6F", "#4A0E3A"],
        text="% High ACE (4+)")
    fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig3.update_layout(plot_bgcolor="white", xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("#### % High ACE Exposure (4+) by Education Level")
    edu_order = ["No School", "Elementary", "Some High School",
                 "High School Graduate", "Some College", "College Graduate"]
    edu_pct = filtered[filtered["EDUCA"].isin(edu_order)].groupby("EDUCA").apply(
        lambda x: (x["ACE_CATEGORY"] == "High (4+)").mean() * 100
    ).reindex(edu_order).reset_index()
    edu_pct.columns = ["Education Level", "% High ACE (4+)"]
    fig4 = px.bar(edu_pct, x="Education Level", y="% High ACE (4+)",
        color="% High ACE (4+)",
        color_continuous_scale=["#F2C4D0", "#9B2D6F", "#4A0E3A"],
        text="% High ACE (4+)")
    fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig4.update_layout(plot_bgcolor="white", xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### BMI Category by ACE Exposure")
    bmi_ace = filtered.groupby(["_BMI5CAT", "ACE_CATEGORY"]).size().reset_index(name="Count")
    bmi_order = ["Underweight", "Normal", "Overweight", "Obese"]
    fig5 = px.bar(bmi_ace, x="_BMI5CAT", y="Count", color="ACE_CATEGORY",
        barmode="group",
        category_orders={"_BMI5CAT": bmi_order},
        color_discrete_sequence=["#4A0E3A", "#9B2D6F", "#D4739A", "#F2C4D0"],
        labels={"_BMI5CAT": "BMI Category", "ACE_CATEGORY": "ACE Category"})
    fig5.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    st.markdown("#### Mean ACE Score by Age & Gender")
    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    heatmap_data = filtered.groupby(["_AGE_G", "SEXVAR"])["ACE_SCORE"].mean().unstack()
    heatmap_data = heatmap_data.reindex(age_order)
    fig6 = px.imshow(heatmap_data,
        color_continuous_scale=["#F2C4D0", "#9B2D6F", "#4A0E3A"],
        labels={"color": "Mean ACE Score"},
        aspect="auto")
    fig6.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig6, use_container_width=True)
