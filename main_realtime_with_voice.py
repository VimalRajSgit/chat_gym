# main_realtime_with_voice.py
import cv2
import argparse
import mediapipe as mp
from groq import Groq
import pyttsx3  # Local TTS fallback
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
from utils import *
import threading
import queue
import numpy as np
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize APIs
groq_client = Groq(api_key="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")

# Initialize local TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed up speech

# Queue for feedback
feedback_queue = queue.Queue()
MAX_QUEUE_SIZE = 2

def speak_text(text):
    """Use local TTS as primary audio output"""
    if feedback_queue.qsize() >= MAX_QUEUE_SIZE:
        logging.info("Feedback queue full, skipping")
        return
    feedback_queue.put(text)
    logging.info(f"Queued feedback: {text}")

def play_feedback_queue():
    """Thread to play feedback using local TTS"""
    while True:
        text = feedback_queue.get()
        if text is None:
            break
        try:
            logging.info(f"Speaking: {text}")
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"TTS error: {e}")
            print(text)  # Fallback to console
        feedback_queue.task_done()

def get_motivational_message(counter, exercise_type):
    """Simplified motivational messages"""
    if counter == 0:
        return f"Start {exercise_type}s!"
    elif counter % 5 == 0:
        return f"{counter} reps!"
    return ""

def analyze_exercise(landmarks, exercise_type, counter):
    """Simplified analysis without Groq for speed"""
    try:
        exercise_obj = BodyPartAngle(landmarks)
        form_tip = ""
        
        if exercise_type == "push-up":
            angle = (exercise_obj.angle_of_the_left_arm() + exercise_obj.angle_of_the_right_arm()) / 2
            form_tip = "Straighten!" if angle < 70 else ""
        elif exercise_type == "squat":
            angle = (exercise_obj.angle_of_the_left_leg() + exercise_obj.angle_of_the_right_leg()) / 2
            form_tip = "Lower!" if angle > 90 else ""

        motivational_msg = get_motivational_message(counter, exercise_type)
        return f"{motivational_msg} {form_tip}".strip()
    except Exception as e:
        return f"{counter} reps!"

def process_video(video_path, exercise_type, pose):
    """Ultra-optimized video processing"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps // 5))  # Target 5 FPS processing
    speak_interval = int(fps * 3)  # Speak every 3 seconds

    counter = 0
    status = True
    last_spoken = None
    frame_count = 0
    total_time = 0

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Minimal processing
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)  # Fast resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            exercise_obj = TypeOfExercise(landmarks)
            counter, status = exercise_obj.calculate_exercise(exercise_type, counter, status)

            if frame_count % speak_interval == 0 or counter != last_spoken:
                analysis = analyze_exercise(landmarks, exercise_type, counter)
                if analysis:  # Only speak if there's something to say
                    threading.Thread(target=speak_text, args=(analysis,), daemon=True).start()
                    last_spoken = counter

        # Minimal visualization
        frame = score_table(exercise_type, frame, counter, status)
        cv2.imshow('Exercise Trainer', frame)
        frame_count += 1

        frame_time = time.time() - start_time
        total_time += frame_time
        if frame_count % 50 == 0:
            logging.info(f"Frame {frame_count}: {frame_time:.3f}s")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    logging.info(f"Average frame time: {total_time / frame_count:.3f}s")
    return counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--exercise_type", type=str, help='Type of activity to do', required=True)
    args = vars(ap.parse_args())

    # Start feedback thread
    feedback_thread = threading.Thread(target=play_feedback_queue, daemon=True)
    feedback_thread.start()

    # Setup MediaPipe
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    ) as pose:
        video_folder = "Exercise_videos"
        if not os.path.exists(video_folder):
            logging.error(f"{video_folder} folder not found")
            return

        total_reps = 0
        for video_file in os.listdir(video_folder):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(video_folder, video_file)
                logging.info(f"Processing {video_file}")
                start_time = time.time()
                reps = process_video(video_path, args["exercise_type"], pose)
                total_reps += reps
                logging.info(f"Completed {video_file}. Reps: {reps}. Time: {time.time() - start_time:.2f}s")

    cv2.destroyAllWindows()
    feedback_queue.put(None)
    feedback_thread.join()
    logging.info(f"Total {args['exercise_type']} reps: {total_reps}")

if __name__ == "__main__":
    main()