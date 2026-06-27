import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Health Outcomes", page_icon="🏥", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/brfss_ace_clean_2023.csv")

df = load_data()

st.markdown("<h2 style='color:#4A0E3A;'>🏥 ACEs & Health Outcomes</h2>", unsafe_allow_html=True)
st.markdown("Explore how adverse childhood experiences relate to chronic disease, mental health, and risky behaviors in adulthood.")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")
gender = st.sidebar.multiselect("Gender", options=df["SEXVAR"].dropna().unique(), default=df["SEXVAR"].dropna().unique())
age = st.sidebar.multiselect("Age Group", options=df["_AGE_G"].dropna().unique(), default=df["_AGE_G"].dropna().unique())

filtered = df[
    df["SEXVAR"].isin(gender) &
    df["_AGE_G"].isin(age)
]

# ACE category order
cat_order = ["None (0)", "Low (1)", "Moderate (2-3)", "High (4+)"]

st.markdown("### Key Metrics")
col1, col2, col3, col4 = st.columns(4)

high_ace = filtered[filtered["ACE_CATEGORY"] == "High (4+)"]
no_ace = filtered[filtered["ACE_CATEGORY"] == "None (0)"]

dep_high = (high_ace["ADDEPEV3"] == "Yes").mean() * 100
dep_none = (no_ace["ADDEPEV3"] == "Yes").mean() * 100
diab_high = (high_ace["DIABETE4"] == "Yes").mean() * 100
diab_none = (no_ace["DIABETE4"] == "Yes").mean() * 100

col1.metric("Depression - High ACE", f"{dep_high:.1f}%", f"+{dep_high - dep_none:.1f}% vs No ACE")
col2.metric("Depression - No ACE", f"{dep_none:.1f}%")
col3.metric("Diabetes - High ACE", f"{diab_high:.1f}%", f"+{diab_high - diab_none:.1f}% vs No ACE")
col4.metric("Diabetes - No ACE", f"{diab_none:.1f}%")

st.markdown("---")

# Row 1: Depression + Diabetes by ACE category
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Depression Rate by ACE Category")
    dep = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["ADDEPEV3"] == "Yes").mean() * 100
    ).reindex(cat_order).reset_index()
    dep.columns = ["ACE Category", "Depression Rate (%)"]
    fig1 = px.bar(
        dep, x="ACE Category", y="Depression Rate (%)",
        color="ACE Category",
        color_discrete_sequence=["#F2C4D0", "#D4739A", "#9B2D6F", "#4A0E3A"],
        text="Depression Rate (%)"
    )
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig1.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### Diabetes Rate by ACE Category")
    diab = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["DIABETE4"] == "Yes").mean() * 100
    ).reindex(cat_order).reset_index()
    diab.columns = ["ACE Category", "Diabetes Rate (%)"]
    fig2 = px.bar(
        diab, x="ACE Category", y="Diabetes Rate (%)",
        color="ACE Category",
        color_discrete_sequence=["#F2C4D0", "#D4739A", "#9B2D6F", "#4A0E3A"],
        text="Diabetes Rate (%)"
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Row 2: Heart disease + High BP by ACE category
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Heart Disease Rate by ACE Category")
    heart = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["CVDCRHD4"] == "Yes").mean() * 100
    ).reindex(cat_order).reset_index()
    heart.columns = ["ACE Category", "Heart Disease Rate (%)"]
    fig3 = px.bar(
        heart, x="ACE Category", y="Heart Disease Rate (%)",
        color="ACE Category",
        color_discrete_sequence=["#F2C4D0", "#D4739A", "#9B2D6F", "#4A0E3A"],
        text="Heart Disease Rate (%)"
    )
    fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig3.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("#### High Blood Pressure Rate by ACE Category")
    bp = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["BPHIGH6"] == "Yes").mean() * 100
    ).reindex(cat_order).reset_index()
    bp.columns = ["ACE Category", "High BP Rate (%)"]
    fig4 = px.bar(
        bp, x="ACE Category", y="High BP Rate (%)",
        color="ACE Category",
        color_discrete_sequence=["#F2C4D0", "#D4739A", "#9B2D6F", "#4A0E3A"],
        text="High BP Rate (%)"
    )
    fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig4.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Row 3: Mental health days + Physical health days vs ACE score
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Avg Poor Mental Health Days by ACE Score")
    mental = filtered.groupby("ACE_SCORE")["MENTHLTH"].mean().reset_index()
    mental.columns = ["ACE Score", "Avg Poor Mental Health Days"]
    fig5 = px.line(
        mental, x="ACE Score", y="Avg Poor Mental Health Days",
        markers=True,
        color_discrete_sequence=["#9B2D6F"]
    )
    fig5.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    st.markdown("#### Avg Poor Physical Health Days by ACE Score")
    physical = filtered.groupby("ACE_SCORE")["PHYSHLTH"].mean().reset_index()
    physical.columns = ["ACE Score", "Avg Poor Physical Health Days"]
    fig6 = px.line(
        physical, x="ACE Score", y="Avg Poor Physical Health Days",
        markers=True,
        color_discrete_sequence=["#4A0E3A"]
    )
    fig6.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# Row 4: Risky behaviors — smoking + alcohol
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Smoking Rate by ACE Category")
    smoke = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["SMOKE100"] == "Yes").mean() * 100
    ).reindex(cat_order).reset_index()
    smoke.columns = ["ACE Category", "Smoking Rate (%)"]
    fig7 = px.bar(
        smoke, x="ACE Category", y="Smoking Rate (%)",
        color="ACE Category",
        color_discrete_sequence=["#F2C4D0", "#D4739A", "#9B2D6F", "#4A0E3A"],
        text="Smoking Rate (%)"
    )
    fig7.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig7.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    st.markdown("#### Alcohol Use Rate by ACE Category")
    alc = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["DRINKS_ANY"] == "Yes").mean() * 100
    ).reindex(cat_order).reset_index()
    alc.columns = ["ACE Category", "Alcohol Use Rate (%)"]
    fig8 = px.bar(
        alc, x="ACE Category", y="Alcohol Use Rate (%)",
        color="ACE Category",
        color_discrete_sequence=["#F2C4D0", "#D4739A", "#9B2D6F", "#4A0E3A"],
        text="Alcohol Use Rate (%)"
    )
    fig8.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig8.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)
