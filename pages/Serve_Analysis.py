import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.serve_analysis_utils import get_serve_placement, plot_serve_heatmap_on_image, get_rally_win_percentage, generate_service_analysis_summary


def display_service_analysis():
    if "db" not in st.session_state:
        st.error("Database connection is not available. Please go back to the main page to initialize it.")
        return
    
    db = st.session_state.db

    st.set_page_config(page_title="Service Analysis", page_icon="🎾")
    st.title("🎾 Service Analysis")
    st.markdown("Analyze service performance based on different factors.")

    player_name = st.text_input("Enter player's name:")

    if player_name:
        serve_data = get_serve_placement(db, player_name)
        rally_data = get_rally_win_percentage(db, player_name)

        # Generate and display analysis summary
        analysis_summary = generate_service_analysis_summary(serve_data, rally_data)
        #analysis_summary = "BLA BLA BLA"
        st.subheader("📊 Service Analysis Summary")
        st.markdown(analysis_summary)

        if serve_data and not isinstance(serve_data[0], str):
            serve_dict = {
                "deuce_wide": serve_data[0][1],
                "deuce_middle": serve_data[0][2],
                "deuce_t": serve_data[0][3],
                "advantage_wide": serve_data[0][4],
                "advantage_middle": serve_data[0][5],
                "advantage_t": serve_data[0][6],
            }
            st.subheader("Serve Placement Heatmap")
            plot_serve_heatmap_on_image(serve_dict)
        else:
            st.warning("No serve placement data found.")

        if rally_data and not isinstance(rally_data[0], str):
            rally_lengths = ["1 Shot", "3 Shots", "5 Shots", "7 Shots", "9 Shots"]
            rally_percentages = [rally_data[0][0], rally_data[0][1], rally_data[0][2], rally_data[0][3], rally_data[0][4]]

            st.subheader("Rally Win Percentage by Rally Length")
            st.write(
            f"This chart shows the percentage of points won by {player_name} when serving in rallies of specific lengths. "
            "For example, '3 Shots' includes only rallies that lasted exactly 3 shots."
            )
            df = pd.DataFrame({"Rally Length": rally_lengths, "Win Percentage": rally_percentages})

            pastel_colors = ["#A7C7E7", "#FFDDC1", "#B5EAD7", "#CBAACB", "#FFB7B2"]

            fig, ax = plt.subplots()
            ax.bar(df["Rally Length"], df["Win Percentage"], color=pastel_colors, edgecolor="black")

            ax.set_ylabel("Win Percentage")
            ax.set_title(f"Rally Win Percentage for {player_name}")
            ax.set_ylim(0, 100)

            for i, v in enumerate(df["Win Percentage"]):
                ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
            st.pyplot(fig)
        else:
            st.warning("No rally win percentage data found.")
    else:
        st.warning("Please enter a player's name to analyze service performance.")


# Run the service analysis page
display_service_analysis()
