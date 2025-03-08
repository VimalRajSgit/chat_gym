# exercise_analyzer_gradio.py
import cv2
import gradio as gr
from groq import Groq
import pyttsx3
import numpy as np
import os
import tempfile
import base64
from datetime import datetime
import av
from fractions import Fraction
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("exercise_analyzer.log"), logging.StreamHandler()]
)

# Initialize APIs
groq_client = Groq(api_key="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")

# Initialize local TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)
tts_engine.setProperty('volume', 1.0)

# Directory setup
BASE_DIR = "Exercise_analysis"
VIDEO_DIR = "Exercise_videos"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

def analyze_frame_with_grok(frame, exercise_type, previous_counter):
    """Analyze frame with LLaMA Vision for rep counting and motivation"""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        prompt = f"This is a {exercise_type} video. Based on body position, estimate the current rep count (previous count was {previous_counter}) and give a short, energetic motivational phrase like a gym trainer!"
        
        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            temperature=0.8,
            max_tokens=100,
            timeout=3
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Grok analysis error: {e}")
        return f"Reps: {previous_counter} - Keep pushing, you got this!"

def extract_rep_count(analysis_text):
    """Extract rep count from Grok's response"""
    try:
        import re
        match = re.search(r'Reps: (\d+)|(\d+) reps', analysis_text, re.IGNORECASE)
        if match:
            return int(match.group(1) or match.group(2))
        return None
    except Exception as e:
        logging.error(f"Rep extraction error: {e}")
        return None

def overlay_text(frame, text, position=(50, 50), font_scale=1, thickness=2):
    """Overlay text on frame"""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def generate_audio(feedback_text, audio_path):
    """Generate audio with pyttsx3"""
    try:
        logging.info(f"Generating audio for: {feedback_text[:50]}...")
        tts_engine.save_to_file(feedback_text, audio_path)
        tts_engine.runAndWait()
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            logging.info(f"Audio generated successfully: {audio_path}")
            return audio_path
        else:
            logging.error("Audio file not generated or empty")
            return None
    except Exception as e:
        logging.error(f"Audio generation error: {e}")
        return None

def process_video(video_path, exercise_type):
    """Process video with rep counting and gym trainer motivation"""
    start_time = time.time()
    logging.info(f"Starting processing: {video_path} for {exercise_type}")

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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analysis_interval = int(fps * 5)  # Analyze every 5 seconds

    # Temporary video
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

    counter = 0
    frame_count = 0
    all_feedback = ""
    last_motivation = "Let’s go, champ!"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % analysis_interval == 0:
            analysis_text = analyze_frame_with_grok(frame, exercise_type, counter)
            all_feedback += analysis_text + " "
            logging.info(f"Analysis at {frame_count}s: {analysis_text}")
            
            new_counter = extract_rep_count(analysis_text)
            if new_counter is not None and new_counter > counter:
                counter = new_counter
            last_motivation = analysis_text.split(" - ")[-1] if " - " in analysis_text else "Keep it up!"

        # Overlay rep count and motivation
        overlay_text(frame, f"{exercise_type}: {counter} reps", (50, 50))
        overlay_text(frame, last_motivation, (50, 100), font_scale=0.7)

        out.write(frame)
        frame_count += 1

    # Cleanup
    cap.release()
    out.release()

    # Generate audio
    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    audio_path = generate_audio(all_feedback, temp_audio_path)

    # Final video with audio using PyAV
    final_video_path = os.path.join(BASE_DIR, f"{exercise_type}_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    try:
        if audio_path and os.path.exists(audio_path):
            # Open output container
            output = av.open(final_video_path, mode='w')
            video_stream = output.add_stream('h264', rate=Fraction(int(fps), 1))
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = 'yuv420p'  # Ensure compatibility
            audio_stream = output.add_stream('aac')

            # Re-encode video from temp file
            input_video = cv2.VideoCapture(temp_video_path)
            while input_video.isOpened():
                ret, frame = input_video.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
                for packet in video_stream.encode(av_frame):
                    output.mux(packet)
            input_video.release()

            # Encode audio
            input_audio = av.open(audio_path)
            for frame in input_audio.decode(audio=0):
                for packet in audio_stream.encode(frame):
                    output.mux(packet)

            # Flush streams
            for packet in video_stream.encode():
                output.mux(packet)
            for packet in audio_stream.encode():
                output.mux(packet)

            output.close()
            input_audio.close()
            logging.info(f"Video and audio muxed successfully: {final_video_path}")
            os.remove(temp_audio_path)
        else:
            logging.warning("Audio not generated. Using video without audio.")
            os.rename(temp_video_path, final_video_path)
    except Exception as e:
        logging.error(f"PyAV muxing error: {e}")
        final_video_path = temp_video_path  # Fallback

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    if not os.path.exists(final_video_path):
        text_output = f"Total {exercise_type} reps: {counter}\nProcessing time: {(time.time() - start_time):.2f}s\nError: Final video not generated"
        return text_output, None

    text_output = f"Total {exercise_type} reps: {counter}\nProcessing time: {(time.time() - start_time):.2f}s"
    return text_output, final_video_path

def gradio_interface(video_file, exercise_type):
    """Gradio interface"""
    if not video_file or not exercise_type:
        return "Error: Please upload a video and specify an exercise type", None
    
    try:
        text_output, video_output = process_video(video_file, exercise_type.lower())
        if video_output and os.path.exists(video_output):
            return text_output, gr.File(value=video_output, label="Download Processed Video")
        else:
            return f"{text_output}\nError: Video processing failed", None
    except Exception as e:
        logging.error(f"Gradio interface error: {e}")
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
        gr.File(label="Processed Video with Reps and Audio")
    ],
    title="AI Gym Trainer",
    description="Upload an exercise video and get rep counts with real gym trainer motivation in one MP4!"
)

if __name__ == "__main__":
    interface.launch()