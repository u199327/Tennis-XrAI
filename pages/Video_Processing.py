import streamlit as st
from src.utils.tendency_analysis_utils import generate_match_tendency_analysis

def video_upload_page():
    st.set_page_config(page_title="Tennis Video and Image Analysis", page_icon="🎥")
    st.title("🎥 Tennis Video and Image Analysis")
    st.markdown("Upload and analyze tennis match or training session videos to gain insights into player performance.")

    st.subheader("📤 Upload Video")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        try:
            st.video("data/output_data/output.mp4")           

            st.subheader("📊 Exploring Player Dynamics")
            col1, col2 = st.columns(2)

            with col1:
                st.image("data/output_data/bounce_depth_distribution.png", caption="Bounce Depth Distribution", use_container_width=True)

            with col2:
                st.image("data/output_data/combined_heatmap.png", caption="Combined Heatmap", use_container_width=True)

            st.subheader("📈 Tendency Analysis")
            
            tendency_analysis = generate_match_tendency_analysis()
            st.markdown(tendency_analysis)
        
        except Exception as e:
            st.error(f"An error occurred while displaying the video: {e}")
            st.warning("The video cannot be displayed properly. Please check the file format or try another video.")

    else:
        st.warning("Please upload a video file to begin analysis.")

# Run the video upload page
video_upload_page()
