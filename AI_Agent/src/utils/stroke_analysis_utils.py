import plotly.graph_objects as go
from PIL import Image
import pandas as pd
import streamlit as st
from src.sql_react_agent import ReActSQLAgent


def generate_stroke_performance_summary(raw_stroke_data, stroke_prefix, player_name):
    agent = ReActSQLAgent(
        llm_model=None,  # Set your model here
        db_manager=st.session_state.db,
        table_provider=st.session_state.table_provider,
        sql_dialect='SQLite'
    )

    # Mapping of relevant strokes and their human-readable labels
    readable_strokes = {
        "F-XC": "Forehand cross court",
        "F-DTM": "Forehand down the middle",
        "F-DTL": "Forehand down the line",
        "B-XC": "Backhand cross court",
        "B-DTM": "Backhand down the middle",
        "B-DTL": "Backhand down the line"
    }

    # Filter strokes based on prefix
    valid_strokes = [f"{stroke_prefix}-XC", f"{stroke_prefix}-DTM", f"{stroke_prefix}-DTL"]

    # Build stroke data dict
    stroke_data = {
        row[0]: {
            "label": readable_strokes[row[0]],
            "winner_percentage": row[1],
            "induced_forced_percentage": row[2],
            "unforced_error_percentage": row[3]
        }
        for row in raw_stroke_data if row[0] in valid_strokes
    }

    # Convert to prompt lines
    stroke_lines = []
    for stroke_code, values in stroke_data.items():
        wp = values["winner_percentage"]
        ip = values["induced_forced_percentage"]
        up = values["unforced_error_percentage"]
        label = values["label"]
        stroke_lines.append(
            f"- {label}: Winner {wp:.1f}%, Induced Error {ip:.1f}%, Unforced Error {up:.1f}%"
        )

    prompt = f"""
    You are a tennis strategy analyst. You are analyzing stroke direction stats for **{player_name}**.

    The player's performance is broken down by direction for their **{stroke_prefix}-hand**. For each direction, you are given the percentage of:
    - winners
    - forced/induced errors
    - unforced errors

    Here is the data:

    {chr(10).join(stroke_lines)}

    Write a concise analysis (max 150 words) in bullet points. Identify:
    - Best and worst performing directions
    - High-risk patterns (e.g. high unforced error rates)
    - Tactical recommendations for the player
    - Any insightful observations or anomalies
    """

    response_messages = agent.invoke(prompt)
    return response_messages["messages"][-1].content


def get_stroke_performance(db, player_name, stroke_prefix):
    """Fetch stroke performance stats by direction for a given player and stroke type."""
    query = f"""
        SELECT 
            type_of_shot,
            ROUND(SUM(winners) * 100.0 / NULLIF(SUM(pt_ending), 0), 2) AS winner_percentage,
            ROUND(SUM(induced_forced) * 100.0 / NULLIF(SUM(pt_ending), 0), 2) AS induced_forced_percentage,
            ROUND(SUM(unforced) * 100.0 / NULLIF(SUM(pt_ending), 0), 2) AS unforced_error_percentage
        FROM charting_m_stats_ShotDirOutcomes_top25_2024
        WHERE player = '{player_name}'
          AND type_of_shot LIKE '{stroke_prefix}-%'
        GROUP BY type_of_shot
        ORDER BY type_of_shot;
    """
    return db._execute_query(query)

def plot_stroke_performance_on_image(stroke_data, stroke_prefix="F", title="Stroke Performance"):
    court_image = Image.open("tennis_court.png")
    width, height = court_image.size

    stroke_positions_pixels = {
        "F-XC": (67, 57),
        "F-DTM": (197, 57),
        "F-DTL": (332, 57),
        "B-XC": (332, 57),
        "B-DTM": (197, 57),
        "B-DTL": (67, 57),
    }

    if stroke_prefix == "F":
        player_position = (274, 576)
    else:
        player_position = (138, 576)

    fig = go.Figure()

    # Add background image
    fig.add_layout_image(
        dict(
            source=court_image,
            x=0,
            y=height,
            xref="x",
            yref="y",
            sizex=width,
            sizey=height,
            sizing="stretch",
            layer="below"
        )
    )

    # Utility for scaled line thickness
    def scaled_thickness(pct, max_pct=100, max_thick=6):
        return (pct / max_pct) * max_thick

    # Draw lines and markers
    for stroke, (x, y) in stroke_positions_pixels.items():
        if not stroke.startswith(stroke_prefix):
            continue  # Skip strokes that are not of the selected type

        values = stroke_data.get(stroke, {})
        total = sum(values.values())

        for category, color in zip(["winner_percentage", "induced_forced_percentage", "unforced_error_percentage"],
                                   ["green", "blue", "red"]):
            pct = values.get(category, 0)
            thickness = scaled_thickness(pct)

            fig.add_trace(go.Scatter(
                x=[player_position[0], x],
                y=[height - player_position[1], height - y],
                mode="lines",
                line=dict(color=color, width=thickness, shape="spline"),
                name=f"{stroke} - {category.replace('_', ' ').title()} ({pct:.1f}%)"
            ))

        # Central marker
        fig.add_trace(go.Scatter(
            x=[x], y=[height - y],
            mode="markers+text",
            marker=dict(size=40, color="rgba(0,0,0,0)", line=dict(width=2, color="white")),
            text=[stroke],
            textposition="middle center",
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[0, width]),
        yaxis=dict(visible=False, range=[0, height]),
        width=500,
        height=600,
        title=title,
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend = False
    )
    return fig

def plot_combined_stroke_bar_chart(stroke_data, stroke_prefix="F"):
    """Generate a combined bar chart for multiple strokes' performance breakdown."""
    # Mapping for readable category labels and their corresponding keys in the data
    category_mapping = {
        "Winner": "winner_percentage",
        "Induced Error": "induced_forced_percentage",
        "Unforced Error": "unforced_error_percentage"
    }

    # Mapping for descriptive stroke labels
    descriptive_labels = {
        "F-XC": "Cross Court",
        "F-DTM": "Down To the Middle",
        "F-DTL": "Down the Line",
        "B-XC": "Cross Court",
        "B-DTM": "Down To the Middle",
        "B-DTL": "Down the Line"
    }

    # Initialize lists to store values for each stroke
    stroke_values = {key: [] for key in descriptive_labels.keys()}

    # Define which strokes to plot based on the prefix
    if stroke_prefix == "F":
        strokes_to_plot = {key: value for key, value in stroke_data.items() if key in ["F-XC", "F-DTM", "F-DTL"]}
    elif stroke_prefix == "B":
        strokes_to_plot = {key: value for key, value in stroke_data.items() if key in ["B-XC", "B-DTM", "B-DTL"]}
    else:
        strokes_to_plot = {}

    # Prepare the data for each stroke
    for readable_label, data_key in category_mapping.items():
        for stroke, values in strokes_to_plot.items():
            stroke_values[stroke].append(values.get(data_key, 0))

    # Create the bar chart
    fig = go.Figure()

    # Add bars for each stroke
    for stroke, color in zip(strokes_to_plot.keys(), ["green", "blue", "red"]):
        fig.add_trace(go.Bar(
            x=stroke_values[stroke],
            y=list(category_mapping.keys()),
            name=descriptive_labels[stroke],  # Use descriptive labels
            orientation='h',
            marker_color=color,
            text=[f'{value:.1f}%' for value in stroke_values[stroke]],  # Add percentage text
            textposition='auto',  # Automatically position text
            textfont=dict(color='white')  # Set text color to black
        ))

    # Update the layout for the bar chart
    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            title=dict(text="Percentage", font=dict(size=14, color='white', family='Arial')),
            tickfont=dict(size=12, color='white', family='Arial Black, Arial, sans-serif'),
            showline=False
        ),
        yaxis=dict(
            tickfont=dict(size=12, color='white', family='Arial Black, Arial, sans-serif'),
            showline=False
        ),
        barmode="group",
        height=400,
        width=750,
        margin=dict(l=20, r=20, t=30, b=30)
    )

    return fig
