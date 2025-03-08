import gradio as gr
import tempfile
import os
import requests
from optimized_exercise_counter import count_reps
from dotenv import load_dotenv

load_dotenv()

# Groq API configuration
def get_llama_response(prompt, system_prompt="You are a motivational fitness coach. Keep responses short, encouraging, and direct."):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: API key not configured"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize state
workout_plan = {}
current_exercise = None
target_reps = 0
completed_reps = {}
chat_history = []

def add_to_plan(exercise, reps):
    global workout_plan, completed_reps, current_exercise, target_reps, chat_history
    
    workout_plan[exercise] = reps
    completed_reps[exercise] = 0
    
    # Set as current exercise
    current_exercise = exercise
    target_reps = reps
    
    # Start chat
    prompt = f"Alright, {reps} {exercise}s are in your plan! Ready to crush them? Upload your video anytime or let me know what you want to do!"
    response = get_llama_response(prompt)
    chat_history.append(("Coach", response))
    
    # Update plan display
    plan_text = get_plan_text()
    
    return chat_history, plan_text

def get_plan_text():
    global workout_plan, completed_reps
    
    if not workout_plan:
        return "No exercises added yet. Add some to get started!"
    
    plan_text = ""
    for exercise, reps in workout_plan.items():
        completed = completed_reps.get(exercise, 0)
        remaining = max(0, reps - completed)
        plan_text += f"• {exercise}s: {reps} total, {remaining} remaining\n"
    
    return plan_text

def chat_input(message):
    global workout_plan, current_exercise, target_reps, completed_reps, chat_history
    
    if not message:
        return chat_history, ""
    
    exercise_options = ["push-up", "pull-up", "squat", "sit-up", "walk"]
    chat_history.append(("You", message))
    message = message.lower()
    
    if "remaining" in message:
        if current_exercise:
            remaining = max(0, target_reps - completed_reps.get(current_exercise, 0))
            prompt = f"You've got {remaining} {current_exercise}s left to hit your target of {target_reps}! Ready to crush them?"
            response = get_llama_response(prompt)
            chat_history.append(("Coach", response))
    
    elif "upload" in message or "ready" in message:
        if current_exercise:
            response = f"Great! Upload your {current_exercise} video whenever you're ready!"
            chat_history.append(("Coach", response))
        else:
            response = "Which exercise are you working on? Tell me or pick one from your plan!"
            chat_history.append(("Coach", response))
    
    elif any(exercise in message for exercise in exercise_options):
        for exercise in exercise_options:
            if exercise in message:
                if exercise in workout_plan:
                    current_exercise = exercise
                    target_reps = workout_plan[exercise]
                    remaining = max(0, target_reps - completed_reps.get(exercise, 0))
                    prompt = f"Switching to {exercise}s! You've got {remaining} left to hit {target_reps}. Upload your video anytime!"
                    response = get_llama_response(prompt)
                    chat_history.append(("Coach", response))
                else:
                    response = f"{exercise} isn't in your plan yet. Want to add it? Tell me how many reps!"
                    chat_history.append(("Coach", response))
                break
    else:
        # Generic response
        prompt = f"I'm your workout buddy! {message}"
        response = get_llama_response(prompt)
        chat_history.append(("Coach", response))
    
    return chat_history, ""

def process_video(video_path):
    global workout_plan, current_exercise, target_reps, completed_reps, chat_history
    
    if not current_exercise:
        chat_history.append(("Coach", "Which exercise is this video for? Tell me so I can count your reps!"))
        return chat_history, get_plan_text(), None
    
    try:
        # Analyze video
        result = count_reps(video_path, current_exercise)
        actual_reps = result["total_reps"]
        
        # Update completed reps
        current_completed = completed_reps.get(current_exercise, 0)
        completed_reps[current_exercise] = current_completed + actual_reps
        
        remaining = max(0, target_reps - completed_reps[current_exercise])
        success_message = f"You completed {actual_reps} {current_exercise}s! {remaining} left to go!"
        
        # Motivation based on performance
        if remaining == 0:
            prompt = f"You just completed {actual_reps} {current_exercise}s, hitting your target of {target_reps}! Give me a short, energetic congratulations!"
            workout_plan.pop(current_exercise, None)
            completed_reps.pop(current_exercise, None)
            
            # Check if all exercises are complete
            if not workout_plan:
                final_prompt = "I just completed my entire workout! Give a short, energetic congratulations!"
                final_motivation = get_llama_response(final_prompt)
                chat_history.append(("Coach", final_motivation))
        else:
            prompt = f"You completed {actual_reps} {current_exercise}s, with {remaining} left to reach {target_reps}. Give me a short, encouraging message!"
        
        motivation = get_llama_response(prompt)
        chat_history.append(("Coach", motivation))
        
        return chat_history, get_plan_text(), success_message
    
    except Exception as e:
        chat_history.append(("Coach", f"Error analyzing video: {str(e)}"))
        return chat_history, get_plan_text(), None

# Create Gradio interface
with gr.Blocks(title="Simple Workout Tracker") as app:
    gr.Markdown("# Simple Workout Tracker")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## What's your workout plan today?")
            
            exercise_dropdown = gr.Dropdown(
                choices=["push-up", "pull-up", "squat", "sit-up", "walk"],
                label="Choose an exercise:"
            )
            rep_counter = gr.Number(label="How many reps do you want to do?", value=10, minimum=1, maximum=100)
            add_button = gr.Button("Add to Plan")
            
            plan_display = gr.Textbox(
                label="Your Plan",
                value="No exercises added yet. Add some to get started!",
                interactive=False,
                lines=6
            )
        
        with gr.Column(scale=2):
            gr.Markdown("## Let's work out!")
            
            chatbot = gr.Chatbot(label="Chat with Your Coach", height=300)
            msg_input = gr.Textbox(label="Type your response here...", placeholder="Type here and press Enter")
            video_input = gr.Video(label="Upload your exercise video")
            
            success_message = gr.Textbox(label="Status", interactive=False, visible=False)
    
    # Set up event handlers
    add_button.click(
        add_to_plan,
        inputs=[exercise_dropdown, rep_counter],
        outputs=[chatbot, plan_display]
    )
    
    msg_input.submit(
        chat_input,
        inputs=[msg_input],
        outputs=[chatbot, msg_input]
    )
    
    video_input.change(
        process_video,
        inputs=[video_input],
        outputs=[chatbot, plan_display, success_message]
    )

if __name__ == "__main__":
    app.launch()