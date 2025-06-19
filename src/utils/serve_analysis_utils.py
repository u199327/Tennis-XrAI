from PIL import Image
import plotly.graph_objects as go
import streamlit as st
from src.sql_react_agent import ReActSQLAgent

def get_serve_placement(db, player_name):
    """Fetch serve placement data for the given player."""
    query = f"""
        SELECT 
            serve_number,
            (SUM(deuce_wide) * 100.0 / NULLIF(SUM(deuce_wide + deuce_middle + deuce_t + advantage_wide + advantage_middle + advantage_t), 0)) AS pct_deuce_wide,
            (SUM(deuce_middle) * 100.0 / NULLIF(SUM(deuce_wide + deuce_middle + deuce_t + advantage_wide + advantage_middle + advantage_t), 0)) AS pct_deuce_middle,
            (SUM(deuce_t) * 100.0 / NULLIF(SUM(deuce_wide + deuce_middle + deuce_t + advantage_wide + advantage_middle + advantage_t), 0)) AS pct_deuce_t,
            (SUM(advantage_wide) * 100.0 / NULLIF(SUM(deuce_wide + deuce_middle + deuce_t + advantage_wide + advantage_middle + advantage_t), 0)) AS pct_advantage_wide,
            (SUM(advantage_middle) * 100.0 / NULLIF(SUM(deuce_wide + deuce_middle + deuce_t + advantage_wide + advantage_middle + advantage_t), 0)) AS pct_advantage_middle,
            (SUM(advantage_t) * 100.0 / NULLIF(SUM(deuce_wide + deuce_middle + deuce_t + advantage_wide + advantage_middle + advantage_t), 0)) AS pct_advantage_t
        FROM charting_m_stats_ServeDirection_top25_2024
        WHERE player = '{player_name}'
        GROUP BY serve_number;
    """
    return db._execute_query(query)

def get_rally_win_percentage(db, player_name):
    """Fetch rally win percentage data for the given player."""
    query = f"""
        SELECT 
            AVG("%_won_1+_rally_length") AS avg_win_1_rally,
            AVG("%_won_3+_rally_length") AS avg_win_3_rally,
            AVG("%_won_5+_rally_length") AS avg_win_5_rally,
            AVG("%_won_7+_rally_length") AS avg_win_7_rally,
            AVG("%_won_9+_rally_length") AS avg_win_9_rally
        FROM charting_m_stats_ServeInfluence_top25_2024
        WHERE player = '{player_name}'
        GROUP BY serve_number;
    """
    return db._execute_query(query)

def generate_service_analysis_summary(serve_data, rally_data):
    agent = ReActSQLAgent(
        llm_model=None, 
        db_manager=st.session_state.db, 
        table_provider=st.session_state.table_provider,
        sql_dialect='SQLite'
    )
    
    # Construct the prompt with serve and rally data
    prompt = """
    You are an expert tennis analyst. Given the following serve placement and rally win percentage data, generate a concise textual summary highlighting key insights and provinding some advises.

    **Serve Placement Data (percentage of serves in each location):**
    - Deuce Wide: {deuce_wide}%
    - Deuce Middle: {deuce_middle}%
    - Deuce T: {deuce_t}%
    - Advantage Wide: {advantage_wide}%
    - Advantage Middle: {advantage_middle}%
    - Advantage T: {advantage_t}%

    **Rally Win Percentage Data:**
    - 1+ Shots: {rally_1}%
    - 3+ Shots: {rally_3}%
    - 5+ Shots: {rally_5}%
    - 7+ Shots: {rally_7}%
    - 9+ Shots: {rally_9}%

    Provide a short summary in a few bullet points, focusing on text insights rather than numbers. Identify:
    - The most common serve placements.
    - The player's strongest and weakest rally lengths based on win percentage.
    - Any interesting patterns or trends.
    - Advice

    Do not exceed 100 words
    """

    if serve_data and not isinstance(serve_data[0], str):
        serve_dict = {
            "deuce_wide": round(serve_data[0][1], 1),
            "deuce_middle": round(serve_data[0][2], 1),
            "deuce_t": round(serve_data[0][3], 1),
            "advantage_wide": round(serve_data[0][4], 1),
            "advantage_middle": round(serve_data[0][5], 1),
            "advantage_t": round(serve_data[0][6], 1),
        }
    else:
        serve_dict = {key: "N/A" for key in ["deuce_wide", "deuce_middle", "deuce_t", "advantage_wide", "advantage_middle", "advantage_t"]}

    if rally_data and not isinstance(rally_data[0], str):
        rally_dict = {
            "rally_1": round(rally_data[0][0], 1),
            "rally_3": round(rally_data[0][1], 1),
            "rally_5": round(rally_data[0][2], 1),
            "rally_7": round(rally_data[0][3], 1),
            "rally_9": round(rally_data[0][4], 1)
        }
    else:
        rally_dict = {f"rally_{i}": "N/A" for i in range(1, 6)}

    # Format the prompt with actual data
    prompt = prompt.format(**serve_dict, **rally_dict)

    # Invoke the agent with the prompt
    response_messages = agent.invoke(prompt)

    return response_messages["messages"][-1].content

def plot_serve_heatmap_on_image(serve_data):
    """Plot serve placement percentages over a tennis court image with enhanced visualization."""
    
    # Load the tennis court image
    court_image = Image.open("tennis_court.png")
    
    # Get the image dimensions
    width, height = court_image.size
    
    # Serve position data (using pixel coordinates)
    serve_positions_pixels = {
        "deuce_wide": (50, 100),
        "deuce_middle": (110, 100),
        "deuce_t": (165, 100),
        "advantage_t": (210, 100),
        "advantage_middle": (265, 100),
        "advantage_wide": (325, 100),
    }

    # Define points A and B for the curved rows
    point_A = (200, 560)  # Example: point A coordinates (adjust as needed)
    point_B = (165, 560)  # Example: point B coordinates (adjust as needed)

    # Calculate the total number of serves for each side
    deuce_total = sum([serve_data.get(key, 0) for key in serve_positions_pixels if "deuce" in key])
    advantage_total = sum([serve_data.get(key, 0) for key in serve_positions_pixels if "advantage" in key])

    # Function to calculate serve percentages
    def calculate_percentage(serve_count, total_count):
        return (serve_count / total_count) * 100 if total_count > 0 else 0

    # Create the figure with the image
    fig = go.Figure()

    # Add the image as the background
    fig.add_layout_image(
        dict(
            source=court_image,
            x=0,
            y=height,  # Align the top of the image with the y-axis maximum
            xref="x",
            yref="y",
            sizex=width,
            sizey=height,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Function to calculate line width based on percentage
    def calculate_line_width(value, max_value, max_width=5):
        return (value / max_value) * max_width

    # Plot curved lines from point A to the first 3 coordinates (deuce side)
    serve_styles = {
        "deuce_wide": {
            "color": "#E94B3C",  # Bright red
            "dash": "solid",
            "label": "Deuce Wide Serve"
        },
        "deuce_middle": {
            "color": "#B22222",  # Firebrick red (darker)
            "dash": "dash",
            "label": "Deuce Middle Serve"
        },
        "deuce_t": {
            "color": "#7C0A02",  # Deep red (almost maroon)
            "dash": "dot",
            "label": "Deuce T Serve"
        }
    }

    for key, serve_pos in {
        "deuce_wide": serve_positions_pixels["deuce_wide"],
        "deuce_middle": serve_positions_pixels["deuce_middle"],
        "deuce_t": serve_positions_pixels["deuce_t"],
    }.items():
        serve_count = serve_data.get(key, 0)
        serve_percentage = calculate_percentage(serve_count, deuce_total)
        line_width = calculate_line_width(serve_percentage, 100)

        label = serve_styles[key]["label"]

        fig.add_trace(go.Scatter(
            x=[point_A[0], serve_pos[0]],
            y=[height - point_A[1], height - serve_pos[1]],
            mode='lines',
            line=dict(
                color=serve_styles[key]["color"],
                width=line_width,
                shape='spline',
                dash=serve_styles[key]["dash"]
            ),
            name=label
        ))

    advantage_serve_styles = {
        "advantage_wide": {
            "color": "#5E239D",  # Dark purple
            "dash": "solid",
            "label": "Advantage Wide Serve"
        },
        "advantage_middle": {
            "color": "#4B1C87",  # Darker, rich purple
            "dash": "dash",
            "label": "Advantage Middle Serve"
        },
        "advantage_t": {
            "color": "#3A146C",  # Deep violet
            "dash": "dot",
            "label": "Advantage T Serve"
        }
    }

    # Plotting
    for key, serve_pos in {
        "advantage_wide": serve_positions_pixels["advantage_wide"],
        "advantage_middle": serve_positions_pixels["advantage_middle"],
        "advantage_t": serve_positions_pixels["advantage_t"],
    }.items():
        serve_count = serve_data.get(key, 0)
        serve_percentage = calculate_percentage(serve_count, advantage_total)
        line_width = calculate_line_width(serve_percentage, 100)

        label = advantage_serve_styles[key]["label"]

        fig.add_trace(go.Scatter(
            x=[point_B[0], serve_pos[0]],
            y=[height - point_B[1], height - serve_pos[1]],
            mode='lines',
            line=dict(
                color=advantage_serve_styles[key]["color"],
                width=line_width,
                shape='spline',
                dash=advantage_serve_styles[key]["dash"]
            ),
            name=label
        ))

    # Plot serve placement markers on top of the image using pixel coordinates
    for key, (x_pixel, y_pixel) in serve_positions_pixels.items():
        serve_count = serve_data.get(key, 0)
        
        # Determine the percentage for deuce and advantage sides
        if "deuce" in key:
            serve_percentage = calculate_percentage(serve_count, deuce_total)
        else:
            serve_percentage = calculate_percentage(serve_count, advantage_total)
        
        color_intensity = serve_percentage / 100  # Normalize percentage to a value between 0 and 1
        marker_size = serve_percentage * 1.5  # Adjust marker size based on percentage

        # Adjusting the y-axis to invert the coordinates
        fig.add_trace(go.Scatter(
            x=[x_pixel], 
            y=[height - y_pixel],  # Invert y-coordinate to align with image
            mode="markers+text",
            marker=dict(size=marker_size, color=f"rgba(255, 0, 0, {color_intensity})", opacity=0.8),
            text=[f"{serve_percentage:.1f}%"],
            textposition="middle center",
            textfont=dict(color="black", size=18, weight="bold"),  # Increased text size
            showlegend=False  # Hide legend for markers
        ))

    # Update layout to ensure everything fits
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, width]),  # Match the x-range with the image width
        yaxis=dict(visible=False, range=[0, height]),  # Match the y-range with the image height
        width=600,
        height=800,
        title="Serve Placement Heatmap",
        plot_bgcolor="rgba(0,0,0,0)"  # Transparent background to make image visible
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)
