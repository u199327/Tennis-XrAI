import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.overview_utils import get_serve_performance, get_break_point_efficiency, get_return_performance, get_winners_and_errors,generate_full_performance_summary


def overview_analysis():
    if "db" not in st.session_state:
        st.error("Database connection is not available. Please go back to the main page to initialize it.")
        return
    
    db = st.session_state.db

    st.set_page_config(page_title="Player Performance Analysis", page_icon="🎾")
    st.title("🎾 Player Performance Analysis")
    st.markdown("Analyze and compare the performance of a player in various aspects including serve, return, break points, and winners/errors.")

    player_name = st.text_input("Enter player's name:")

    if player_name:
        serve_data = get_serve_performance(db, player_name) # Fetch serve performance data
        break_point_data = get_break_point_efficiency(db, player_name) # Fetch break point efficiency data
        return_data = get_return_performance(db, player_name) # Fetch return performance data
        winners_errors_data = get_winners_and_errors(db, player_name) # Fetch winners and errors data
        
        analysis_summary = generate_full_performance_summary(serve_data, break_point_data, return_data, winners_errors_data)
        #analysis_summary = "BLA BLA BLA"
        st.subheader("📊 Overview Summary")
        st.markdown(analysis_summary)

        if serve_data:
            st.subheader("📊 Serve Performance")
            st.markdown(f"**Average Aces:** {serve_data[0][0]} aces")
            st.markdown(f"**Average Double Faults:** {serve_data[0][1]} double faults")
            st.markdown(f"**Average First Serve In Percentage:** {serve_data[0][2]}%")
            st.markdown(f"**Average First Serve Won Percentage:** {serve_data[0][3]}%")
            st.markdown(f"**Average Second Serve Won Percentage:** {serve_data[0][4]}%")
        else:
            st.warning("No serve performance data found.")
        
        if break_point_data:
            st.subheader("📊 Break Point Efficiency")
            st.markdown(f"**{player_name}'s Break Point Saved Percentage:** {break_point_data[0][0]}%")
            st.markdown(f"**Global Average Break Point Saved Percentage:** {break_point_data[0][1]}%")

            fig, ax = plt.subplots()
            ax.bar([f"{player_name}", "Overall"], [break_point_data[0][0], break_point_data[0][1]], color=["#FF6347", "#4CAF50"], edgecolor="black")
            ax.set_ylabel("Break Points Saved Percentage")
            ax.set_title(f"Break Point Saving Performance for {player_name}")
            ax.set_ylim(0, 100)
            st.pyplot(fig)
        else:
            st.warning("No break point efficiency data found.")
        
        if return_data:
            player_return_avg = return_data[0][0]
            overall_return_avg = return_data[0][1]
            st.subheader("📊 Return Performance")
            st.markdown(f"**{player_name}'s Return Points Won Percentage:** {player_return_avg}%")
            st.markdown(f"**Overall Average Return Points Won Percentage:** {overall_return_avg}%")
            
            fig, ax = plt.subplots()
            ax.bar([f"{player_name}", "Overall"], [player_return_avg, overall_return_avg], color=["#FF6347", "#4CAF50"], edgecolor="black")
            ax.set_ylabel("Return Points Won Percentage")
            ax.set_title(f"Return Performance for {player_name}")
            ax.set_ylim(0, 100)
            st.pyplot(fig)
        else:
            st.warning("No return performance data found.")
        
        if winners_errors_data:
            st.subheader("📊 Winners and Unforced Errors per match")
            st.markdown(f"**Average Winners:** {winners_errors_data[0][0]}")
            st.markdown(f"**Average Unforced Errors:** {winners_errors_data[0][1]}")
            st.markdown(f"**Average Forehand Winners:** {winners_errors_data[0][2]}")
            st.markdown(f"**Average Backhand Winners:** {winners_errors_data[0][3]}")
            st.markdown(f"**Average Forehand Unforced Errors:** {winners_errors_data[0][4]}")
            st.markdown(f"**Average Backhand Unforced Errors:** {winners_errors_data[0][5]}")
        else:
            st.warning("No winners and unforced errors data found.")
    else:
        st.warning("Please enter a player's name to analyze performance.")


# Run the player performance analysis page
overview_analysis()