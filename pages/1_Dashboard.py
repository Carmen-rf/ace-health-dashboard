import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/brfss_ace_clean_2023.csv")

df = load_data()

st.markdown("<h2 style='color:#5B2C6F;'>📊 ACE Health Impact Dashboard</h2>", unsafe_allow_html=True)
st.markdown("Analyzing Adverse Childhood Experiences and their impact on adult health outcomes among 56,041 U.S. adults.")
st.markdown("---")

st.sidebar.header("Filters")
gender = st.sidebar.multiselect("Gender", options=df["SEXVAR"].dropna().unique(), default=df["SEXVAR"].dropna().unique())
age = st.sidebar.multiselect("Age Group", options=df["_AGE_G"].dropna().unique(), default=df["_AGE_G"].dropna().unique())
filtered = df[df["SEXVAR"].isin(gender) & df["_AGE_G"].isin(age)]
cat_order = ["None (0)", "Low (1)", "Moderate (2-3)", "High (4+)"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Respondents", f"{len(filtered):,}")
col2.metric("Mean ACE Score", f"{filtered['ACE_SCORE'].mean():.2f}")
col3.metric("High ACE Exposure (4+)", f"{(filtered['ACE_CATEGORY'] == 'High (4+)').mean()*100:.1f}%")
col4.metric("Depression Rate (High ACE)", f"{(filtered[filtered['ACE_CATEGORY']=='High (4+)']['ADDEPEV3']=='Yes').mean()*100:.1f}%")

st.markdown("---")
st.markdown("### 🔍 ACE Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ACE Category Breakdown")
    cat_counts = filtered["ACE_CATEGORY"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig1 = px.pie(cat_counts, names="Category", values="Count", hole=0.45,
        color_discrete_sequence=["#5B2C6F","#1ABC9C","#E74C3C","#2E86C1"])
    fig1.update_traces(textposition="outside", textinfo="percent+label")
    fig1.update_layout(showlegend=False, margin=dict(t=30,b=30))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### Prevalence of Each ACE Type")
    ace_vars = ["ACEDEPRS","ACEDRINK","ACEDRUGS","ACEPRISN","ACEDIVRC",
                "ACEPUNCH","ACEHURT1","ACESWEAR","ACETOUCH","ACETTHEM",
                "ACEHVSEX","ACEADSAF","ACEADNED"]
    ace_labels = {
        "ACEDEPRS":"Parent Mental Illness","ACEDRINK":"Parent Alcohol Abuse",
        "ACEDRUGS":"Parent Drug Abuse","ACEPRISN":"Parent Incarcerated",
        "ACEDIVRC":"Parental Divorce","ACEPUNCH":"Physical Abuse",
        "ACEHURT1":"Physical Hurt","ACESWEAR":"Verbal Abuse",
        "ACETOUCH":"Sexual Touch","ACETTHEM":"Sexual Force",
        "ACEHVSEX":"Forced Sex","ACEADSAF":"Unsafe Home",
        "ACEADNED":"Basic Needs Unmet"
    }
    ace_pct = pd.DataFrame({
        "ACE Type":[ace_labels[v] for v in ace_vars],
        "% Experienced":[filtered[v].mean()*100 for v in ace_vars]
    }).sort_values("% Experienced", ascending=True)
    fig2 = px.bar(ace_pct, x="% Experienced", y="ACE Type", orientation="h",
        color="% Experienced",
        color_continuous_scale=["#AED6F1","#2E86C1","#1A5276"])
    fig2.update_layout(plot_bgcolor="white", showlegend=False, margin=dict(t=30,b=30))
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.markdown("#### ACE Categories by Type")
    ace_groups = {
        "Household Dysfunction":["ACEDEPRS","ACEDRINK","ACEDRUGS","ACEPRISN","ACEDIVRC"],
        "Physical Abuse":["ACEPUNCH","ACEHURT1"],
        "Verbal Abuse":["ACESWEAR"],
        "Sexual Abuse":["ACETOUCH","ACETTHEM","ACEHVSEX"],
        "Neglect & Safety":["ACEADSAF","ACEADNED"]
    }
    group_pct = pd.DataFrame({
        "Group":list(ace_groups.keys()),
        "% Experienced":[filtered[list(v)].max(axis=1).mean()*100 for v in ace_groups.values()]
    }).sort_values("% Experienced", ascending=True)
    fig3 = px.bar(group_pct, x="% Experienced", y="Group", orientation="h",
        color="Group",
        color_discrete_sequence=["#E74C3C","#E67E22","#F1C40F","#1ABC9C","#5B2C6F"])
    fig3.update_layout(plot_bgcolor="white", showlegend=False, margin=dict(t=30,b=30))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("### 👤 Demographics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### Mean ACE Score by Gender")
    gender_ace = filtered.groupby("SEXVAR")["ACE_SCORE"].mean().reset_index()
    gender_ace.columns = ["Gender","Mean ACE Score"]
    fig4 = px.bar(gender_ace, x="Gender", y="Mean ACE Score",
        color="Gender", color_discrete_sequence=["#5B2C6F","#1ABC9C"], text="Mean ACE Score")
    fig4.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig4.update_layout(plot_bgcolor="white", showlegend=False, margin=dict(t=30,b=30))
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    st.markdown("#### Mean ACE Score by Age Group")
    age_order = ["18-24","25-34","35-44","45-54","55-64","65+"]
    age_ace = filtered.groupby("_AGE_G")["ACE_SCORE"].mean().reindex(age_order).reset_index()
    age_ace.columns = ["Age Group","Mean ACE Score"]
    fig5 = px.line(age_ace, x="Age Group", y="Mean ACE Score",
        markers=True, color_discrete_sequence=["#E74C3C"])
    fig5.update_layout(plot_bgcolor="white", margin=dict(t=30,b=30))
    st.plotly_chart(fig5, use_container_width=True)

with col3:
    st.markdown("#### % High ACE by Income")
    income_order = ["<$15k","$15-25k","$25-35k","$35-50k","$50-100k","$75-100k","$100-200k","$150-200k","$200k+"]
    income_pct = filtered[filtered["INCOME3"].isin(income_order)].groupby("INCOME3").apply(
        lambda x: (x["ACE_CATEGORY"]=="High (4+)").mean()*100
    ).reindex(income_order).reset_index()
    income_pct.columns = ["Income Level","% High ACE (4+)"]
    fig6 = px.bar(income_pct, x="Income Level", y="% High ACE (4+)",
        color="% High ACE (4+)",
        color_continuous_scale=["#AED6F1","#2E86C1","#1A5276"],
        text="% High ACE (4+)")
    fig6.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig6.update_layout(plot_bgcolor="white", showlegend=False, xaxis_tickangle=-45, margin=dict(t=30,b=30))
    st.plotly_chart(fig6, use_container_width=True)

with col4:
    st.markdown("#### % High ACE by Education")
    edu_order = ["No School","Elementary","Some High School","High School Graduate","Some College","College Graduate"]
    edu_pct = filtered[filtered["EDUCA"].isin(edu_order)].groupby("EDUCA").apply(
        lambda x: (x["ACE_CATEGORY"]=="High (4+)").mean()*100
    ).reindex(edu_order).reset_index()
    edu_pct.columns = ["Education Level","% High ACE (4+)"]
    fig7 = px.bar(edu_pct, x="Education Level", y="% High ACE (4+)",
        color="% High ACE (4+)",
        color_continuous_scale=["#A9DFBF","#27AE60","#1E8449"],
        text="% High ACE (4+)")
    fig7.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig7.update_layout(plot_bgcolor="white", showlegend=False, xaxis_tickangle=-45, margin=dict(t=30,b=30))
    st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")
st.markdown("### 🏥 Health Outcomes")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Depression Rate by ACE Category")
    dep = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["ADDEPEV3"]=="Yes").mean()*100
    ).reindex(cat_order).reset_index()
    dep.columns = ["ACE Category","Depression Rate (%)"]
    fig8 = px.bar(dep, x="ACE Category", y="Depression Rate (%)",
        color="ACE Category",
        color_discrete_sequence=["#1ABC9C","#2E86C1","#E67E22","#E74C3C"],
        text="Depression Rate (%)")
    fig8.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig8.update_layout(plot_bgcolor="white", showlegend=False, margin=dict(t=30,b=30))
    st.plotly_chart(fig8, use_container_width=True)

with col2:
    st.markdown("#### Diabetes Rate by ACE Category")
    diab = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["DIABETE4"]=="Yes").mean()*100
    ).reindex(cat_order).reset_index()
    diab.columns = ["ACE Category","Diabetes Rate (%)"]
    fig9 = px.bar(diab, x="ACE Category", y="Diabetes Rate (%)",
        color="ACE Category",
        color_discrete_sequence=["#1ABC9C","#2E86C1","#E67E22","#E74C3C"],
        text="Diabetes Rate (%)")
    fig9.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig9.update_layout(plot_bgcolor="white", showlegend=False, margin=dict(t=30,b=30))
    st.plotly_chart(fig9, use_container_width=True)

with col3:
    st.markdown("#### Heart Disease & High BP by ACE Category")
    heart = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["CVDCRHD4"]=="Yes").mean()*100
    ).reindex(cat_order).reset_index()
    heart.columns = ["ACE Category","Rate"]
    heart["Condition"] = "Heart Disease"
    bp = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["BPHIGH6"]=="Yes").mean()*100
    ).reindex(cat_order).reset_index()
    bp.columns = ["ACE Category","Rate"]
    bp["Condition"] = "High BP"
    combined = pd.concat([heart, bp])
    fig10 = px.bar(combined, x="ACE Category", y="Rate", color="Condition",
        barmode="group",
        color_discrete_sequence=["#E74C3C","#E67E22"],
        text="Rate")
    fig10.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig10.update_layout(plot_bgcolor="white", margin=dict(t=30,b=30))
    st.plotly_chart(fig10, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Smoking & Alcohol Use by ACE Category")
    smoke = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["SMOKE100"]=="Yes").mean()*100
    ).reindex(cat_order).reset_index()
    smoke.columns = ["ACE Category","Rate"]
    smoke["Behavior"] = "Smoking"
    alc = filtered.groupby("ACE_CATEGORY").apply(
        lambda x: (x["DRINKS_ANY"]=="Yes").mean()*100
    ).reindex(cat_order).reset_index()
    alc.columns = ["ACE Category","Rate"]
    alc["Behavior"] = "Alcohol Use"
    combined2 = pd.concat([smoke, alc])
    fig11 = px.bar(combined2, x="ACE Category", y="Rate", color="Behavior",
        barmode="group",
        color_discrete_sequence=["#5B2C6F","#1ABC9C"],
        text="Rate")
    fig11.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig11.update_layout(plot_bgcolor="white", margin=dict(t=30,b=30))
    st.plotly_chart(fig11, use_container_width=True)

with col2:
    st.markdown("#### Avg Poor Mental & Physical Health Days by ACE Score")
    mental = filtered.groupby("ACE_SCORE")["MENTHLTH"].mean().reset_index()
    mental.columns = ["ACE Score","Days"]
    mental["Type"] = "Mental Health Days"
    physical = filtered.groupby("ACE_SCORE")["PHYSHLTH"].mean().reset_index()
    physical.columns = ["ACE Score","Days"]
    physical["Type"] = "Physical Health Days"
    combined3 = pd.concat([mental, physical])
    fig12 = px.line(combined3, x="ACE Score", y="Days", color="Type",
        markers=True,
        color_discrete_sequence=["#E74C3C","#2E86C1"])
    fig12.update_layout(plot_bgcolor="white", margin=dict(t=30,b=30))
    st.plotly_chart(fig12, use_container_width=True)
