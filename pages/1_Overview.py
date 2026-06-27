import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ACE Overview", page_icon="📊", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/brfss_ace_clean_2023.csv")

df = load_data()

st.markdown("<h2 style='color:#4A0E3A;'>📊 ACE Overview</h2>", unsafe_allow_html=True)
st.markdown("Explore the distribution and prevalence of Adverse Childhood Experiences among U.S. adults.")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")
gender = st.sidebar.multiselect("Gender", options=df["SEXVAR"].dropna().unique(), default=df["SEXVAR"].dropna().unique())
age = st.sidebar.multiselect("Age Group", options=df["_AGE_G"].dropna().unique(), default=df["_AGE_G"].dropna().unique())

# Apply filters
filtered = df[df["SEXVAR"].isin(gender) & df["_AGE_G"].isin(age)]

# Row 1: KPI metrics
st.markdown("### Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Respondents", f"{len(filtered):,}")
col2.metric("Mean ACE Score", f"{filtered['ACE_SCORE'].mean():.2f}")
col3.metric("High Exposure (4+)", f"{(filtered['ACE_CATEGORY'] == 'High (4+)').mean()*100:.1f}%")
col4.metric("No ACEs Reported", f"{(filtered['ACE_CATEGORY'] == 'None (0)').mean()*100:.1f}%")

st.markdown("---")

# Row 2: Score distribution + Category donut
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ACE Score Distribution")
    fig1 = px.histogram(
        filtered, x="ACE_SCORE", nbins=14,
        color_discrete_sequence=["#4A0E3A"],
        labels={"ACE_SCORE": "ACE Score", "count": "Count"}
    )
    fig1.update_layout(bargap=0.1, plot_bgcolor="white")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### ACE Category Breakdown")
    cat_counts = filtered["ACE_CATEGORY"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig2 = px.pie(
        cat_counts, names="Category", values="Count", hole=0.45,
        color_discrete_sequence=["#4A0E3A", "#9B2D6F", "#D4739A", "#F2C4D0"]
    )
    fig2.update_traces(textposition="outside", textinfo="percent+label")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Row 3: % who experienced each ACE type
st.markdown("#### Prevalence of Each ACE Type")

ace_vars = ["ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC",
            "ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM",
            "ACEHVSEX", "ACEADSAF", "ACEADNED"]

ace_labels = {
    "ACEDEPRS": "Parent Mental Illness",
    "ACEDRINK": "Parent Alcohol Abuse",
    "ACEDRUGS": "Parent Drug Abuse",
    "ACEPRISN": "Parent Incarcerated",
    "ACEDIVRC": "Parental Divorce",
    "ACEPUNCH": "Physical Abuse",
    "ACEHURT1": "Physical Hurt",
    "ACESWEAR": "Verbal Abuse",
    "ACETOUCH": "Sexual Touch",
    "ACETTHEM": "Sexual Force",
    "ACEHVSEX": "Forced Sex",
    "ACEADSAF": "Unsafe Home",
    "ACEADNED": "Basic Needs Unmet"
}

ace_pct = pd.DataFrame({
    "ACE Type": [ace_labels[v] for v in ace_vars],
    "% Experienced": [filtered[v].mean() * 100 for v in ace_vars]
}).sort_values("% Experienced", ascending=True)

fig3 = px.bar(
    ace_pct, x="% Experienced", y="ACE Type", orientation="h",
    color="% Experienced",
    color_continuous_scale=["#F2C4D0", "#9B2D6F", "#4A0E3A"],
    labels={"% Experienced": "% of Respondents"}
)
fig3.update_layout(plot_bgcolor="white", showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Row 4: ACE groups breakdown
st.markdown("#### ACE Categories by Type")

ace_groups = {
    "Household Dysfunction": ["ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC"],
    "Physical Abuse": ["ACEPUNCH", "ACEHURT1"],
    "Verbal Abuse": ["ACESWEAR"],
    "Sexual Abuse": ["ACETOUCH", "ACETTHEM", "ACEHVSEX"],
    "Neglect & Safety": ["ACEADSAF", "ACEADNED"]
}

group_pct = pd.DataFrame({
    "Group": list(ace_groups.keys()),
    "% Experienced (any)": [
        filtered[vars].max(axis=1).mean() * 100
        for vars in ace_groups.values()
    ]
}).sort_values("% Experienced (any)", ascending=True)

fig4 = px.bar(
    group_pct, x="% Experienced (any)", y="Group", orientation="h",
    color_discrete_sequence=["#9B2D6F"],
    labels={"% Experienced (any)": "% Who Experienced At Least One"}
)
fig4.update_layout(plot_bgcolor="white")
st.plotly_chart(fig4, use_container_width=True)
