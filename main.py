# exercise_analyzer_gradio.py
import cv2
import gradio as gr
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
from utils import *
import numpy as np
import os
import tempfile
import logging
from datetime import datetime  # Added missing import

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("exercise_analyzer.log"), logging.StreamHandler()]
)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_video(video_path, exercise_type):
    """Process the uploaded video and generate output with rep counting"""
    logging.info(f"Processing video: {video_path} for {exercise_type}")

    # Open video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")
    except Exception as e:
        logging.error(f"Video open error: {e}")
        return f"Error opening video: {str(e)}", None

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 800  # Fixed width as in original code
    height = 480  # Fixed height as in original code

    # Temporary output video
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError("Video writer failed to initialize")
    except Exception as e:
        logging.error(f"Video writer setup error: {e}")
        cap.release()
        return f"Error setting up video output: {str(e)}", None

    # Setup MediaPipe pose detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        counter = 0  # Rep counter
        status = True  # Movement state

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            # Process frame with MediaPipe
            results = pose.process(frame_rgb)
            # Convert back to BGR for OpenCV
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Calculate reps
            try:
                landmarks = results.pose_landmarks.landmark
                counter, status = TypeOfExercise(landmarks).calculate_exercise(exercise_type, counter, status)
            except Exception as e:
                logging.warning(f"Pose detection error: {e}")

            # Overlay rep count and status
            frame = score_table(exercise_type, frame, counter, status)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2),
            )

            # Write frame to output video
            out.write(frame)

        # Cleanup
        cap.release()
        out.release()

    # Final output path
    final_video_path = os.path.join("Exercise_analysis", f"{exercise_type}_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    os.makedirs("Exercise_analysis", exist_ok=True)
    os.rename(temp_video_path, final_video_path)

    if not os.path.exists(final_video_path):
        return f"Error: Final video not generated", None

    return f"Processed {exercise_type} with {counter} reps", final_video_path

def gradio_interface(video_file, exercise_type):
    """Gradio interface function"""
    if not video_file or not exercise_type:
        return "Error: Please upload a video and specify an exercise type", None
    
    try:
        text_output, video_output = process_video(video_file, exercise_type.lower())
        if video_output and os.path.exists(video_output):
            return text_output, gr.File(value=video_output, label="Download Processed Video")
        else:
            return f"{text_output}\nError: Video processing failed", None
    except Exception as e:
        logging.error(f"Gradio processing error: {e}")
        return f"Error during processing: {str(e)}", None

# Define Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Video(label="Upload Exercise Video"),
        gr.Textbox(label="Exercise Type (e.g., push-up, squat, pull-up)", placeholder="Enter exercise name")
    ],
    outputs=[
        gr.Textbox(label="Analysis Results"),
        gr.File(label="Processed Video with Reps")
    ],
    title="Exercise Analyzer",
    description="Upload a video and specify the exercise type to get rep counts and pose landmarks in a processed video!"
)

if __name__ == "__main__":
    interface.launch(share=True)  # Enable public link