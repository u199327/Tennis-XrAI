import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load Pretrained BLIP model and processor for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# Function to extract all frames from the video
def extract_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB and append to frames list
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames

# Function to generate a caption using BLIP model
def generate_video_caption(frames):
    # Ensure that the frames are not empty
    if len(frames) == 0:
        return "No frames available for caption generation."

    # Generate captions for each frame using BLIP
    captions = []
    for frame in frames:
        # Preprocess the frame and pass it to the BLIP model
        inputs = blip_processor(images=frame, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)

    # Combine all captions into one string for the final caption
    full_caption = " ".join(captions)
    return full_caption

# Process video
def process_video(video_path):
    frames = extract_all_frames(video_path)
    full_caption = generate_video_caption(frames)
    return full_caption

if __name__ == "__main__":
    video_path = "video_input3.mp4"  # Change to your video path
    caption = process_video(video_path)
    print(f"Video Caption: {caption}")