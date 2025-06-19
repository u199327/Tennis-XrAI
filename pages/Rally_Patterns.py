import streamlit as st

from src.utils.rally_analysis_utils import generate_rally_summary, load_rally_data_from_csv

def display_rally_analysis():
    """Streamlit UI for generating rally analysis summary via LLM."""

    st.set_page_config(page_title="Rally Analysis Summary", page_icon="🎾")
    st.title("🎾 Rally Analysis Summary")
    st.markdown("Get an AI-powered breakdown of your most frequent rally sequences.")

    player_name = st.text_input("Enter player's name:")
    role = st.selectbox("Select role:", ["Server", "Returner"])
    outcome = st.selectbox("Select outcome:", ["Win", "Loss"])

    if st.button("Generate Rally Summary"):
        if not player_name:
            st.warning("Please enter a player's name.")
            return

        try:
            data = load_rally_data_from_csv("data/databases_tenis/charting-m-points-2024s.csv")

        except Exception as e:
            st.error(f"Failed to fetch rally data: {e}")
            return

        if not data:
            st.warning("No rally data available.")
            return

        summary = generate_rally_summary(data, player_name, role, outcome)
        st.subheader("🧠 LLM Analysis Summary")
        st.markdown(summary)

display_rally_analysis()