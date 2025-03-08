# main_with_groq.py
import cv2
import argparse
import os
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
from groq import Groq
import base64
from datetime import datetime

# Initialize Groq client (you'll need to set your API key)
groq_client = Groq(api_key="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")

# Directory setup
BASE_DIR = "Exercise_analysis"
VIDEO_DIR = "Exercise_videos"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_results")
FRAME_DIR = os.path.join(BASE_DIR, "extracted_frames")

# Create directories if they don't exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

def analyze_frame_with_groq(frame, exercise_type):
    """Analyze frame using Groq LLaMA Vision model"""
    # Convert frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    prompt = f"Analyze this frame from an exercise video showing {exercise_type}. Describe the body position and suggest if the form is correct.and also motivate him"
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",  # Using LLaMA 3.2 Vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing frame: {str(e)}"

def save_analysis(exercise_type, counter, analysis_text):
    """Save analysis results to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{exercise_type}_analysis_{timestamp}.txt"
    filepath = os.path.join(ANALYSIS_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"Exercise: {exercise_type}\n")
        f.write(f"Reps counted: {counter}\n")
        f.write(f"Analysis: {analysis_text}\n")
    
    return filepath

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--exercise_type", type=str, help='Type of activity to do', required=True)
    ap.add_argument("-v", "--video", type=str, help='Specific video file name', required=False)
    args = vars(ap.parse_args())

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Get video source
    if args["video"]:
        video_path = os.path.join(VIDEO_DIR, args["video"])
    else:
        # Use first video in directory if no specific video specified
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            print(f"No video files found in {VIDEO_DIR}")
            return
        video_path = os.path.join(VIDEO_DIR, video_files[0])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    cap.set(3, 800)  # width
    cap.set(4, 480)  # height

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        counter = 0
        status = True
        frame_count = 0
        analysis_interval = 30  # Analyze every 30 frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                exercise_obj = TypeOfExercise(landmarks)
                counter, status = exercise_obj.calculate_exercise(args["exercise_type"], counter, status)

                # Analyze with Groq every few frames
                if frame_count % analysis_interval == 0:
                    analysis_text = analyze_frame_with_groq(frame, args["exercise_type"])
                    print(f"Frame {frame_count} analysis: {analysis_text}")
                    
                    # Save frame for reference
                    frame_path = os.path.join(FRAME_DIR, f"frame_{frame_count}_{args['exercise_type']}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Save analysis
                    analysis_file = save_analysis(args["exercise_type"], counter, analysis_text)

            except Exception as e:
                print(f"Error processing frame: {e}")

            frame = score_table(args["exercise_type"], frame, counter, status)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2),
            )

            cv2.imshow('Video', frame)
            frame_count += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Exercise completed. Total reps: {counter}")
        print(f"Analysis files saved in: {ANALYSIS_DIR}")
        print(f"Extracted frames saved in: {FRAME_DIR}")

if __name__ == "__main__":
    main()