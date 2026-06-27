import streamlit as st

st.set_page_config(
    page_title="ACE Health Dashboard",
    page_icon="🧠",
    layout="wide"
)

def check_password():
    def password_entered():
        if st.session_state["password"] == "ace2023":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h1 style='text-align:center; color:#4A0E3A;'>🧠 ACE Health Impact Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center; color:gray;'>Adverse Childhood Experiences & Adult Health Outcomes</h4>", unsafe_allow_html=True)
        st.markdown("---")
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
            st.error("❌ Incorrect password. Please try again.")
        return False
    else:
        return True

if check_password():
    st.markdown("<h1 style='text-align:center; color:#4A0E3A;'>🧠 ACE Health Impact Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:gray;'>Adverse Childhood Experiences & Adult Health Outcomes | CDC BRFSS 2023</h4>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Welcome")
    st.write("""
    This dashboard explores the relationship between **Adverse Childhood Experiences (ACEs)** 
    and adult health outcomes among **56,041 U.S. adults** surveyed by the CDC in 2023.
    
    ACEs are traumatic events occurring before age 18 — including abuse, neglect, and household dysfunction. 
    Research shows that the more ACEs a person experiences, the higher their risk of chronic disease, 
    mental illness, and risky behaviors in adulthood.
    """)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Respondents", "56,041")
    col2.metric("📋 ACE Variables", "13")
    col3.metric("⚠️ High ACE Exposure (4+)", "26%")
    col4.metric("📊 Data Source", "CDC BRFSS 2023")

    st.markdown("---")
    st.markdown("### Navigate using the sidebar to explore:")
    st.markdown("""
    - 📊 **ACE Overview** — Distribution and prevalence of ACEs
    - 👤 **Demographics** — ACE exposure by age, gender, income
    - 🏥 **Health Outcomes** — ACEs vs chronic disease and mental health
    - 🤖 **Risk Predictor** — Predict depression risk from ACE profile
    """)
