#!/home/ver/env2/bin/python

import asyncio
import gpiod
from gpiod.line import Edge
from google import genai
from groq import Groq
from kokoro_onnx import Kokoro
import librosa

from google.genai import types
# Remove: from picamera2 import Picamera2
import time
import cv2
import threading
import sounddevice as sd
import queue
import os
import numpy as np
import wave
import whisper
from flask import Flask, request, render_template, jsonify
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from piper.voice import PiperVoice
import subprocess
from Bluetooth_Connection import *
from multiprocessing import Process
import bluetooth
import base64
import re
import io

# ----------- Google / Groq API client -------------
client = genai.Client(api_key="")
client_groq = Groq(api_key="")

vision_model = 0

# ----------- Kokoro TTS ------------------
kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")

# Remove: from picamera2 import Picamera2
# Add webcam initialization
def initialize_webcam():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Webcam could not be opened.")
    return cam

webcam = initialize_webcam()

# ----------- Globals -----------------------
state = 0
loop = None
image_bytes = None
recognized_text = None
recording_task = None
speech_task = None
q = queue.Queue()
searching_thread = None

# Flask
app_text = None
received_text_from_app = asyncio.Event()

# Event trackers
cancel_event = asyncio.Event()
record_stop_event = asyncio.Event()
speech_task_event = asyncio.Event()

# Audio
channels = 1
dtype = 'int16'
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav"


# ---------- Flask App ---------------------
app = Flask(__name__)

# ---------- Website ---------------------
@app.route("/")
def index():
    return render_template('index.html')
   
@app.route('/scan')
def scan_devices():
    print("Scanning for Bluetooth devices...")
    nearby_devices = bluetooth.discover_devices(duration=5, lookup_names=True)
    devices = [{'address': addr, 'name': name} for addr, name in nearby_devices]
    print(f"Found {len(devices)} device(s)")
    return jsonify(devices)

@app.route('/select_device', methods=['POST'])
def select_device():
    data = request.get_json()
    address = data.get('address')
    try:
        with open("saved_mac.txt", 'w') as file:
            file.write(address)
        pair_trust_connect(address)
        return "OK", 200
    except subprocess.CalledProcessError:
        return "Failed", 500

# ---------- Phone ---------------------
@app.route('/receive_mac', methods=['POST'])
def receive_mac():
    mac = request.data.decode().strip()
    print(f"Received MAC: {mac}")
    try:
        with open("saved_mac.txt", 'w') as file:
            file.write(mac)
        pair_trust_connect(mac)
        return "OK", 200
    except subprocess.CalledProcessError:
        return "Failed", 500

@app.route('/receive_text', methods=['POST'])
def receive_text():
    global app_text, received_text_from_app, loop
    text = request.data.decode()
    app_text = text
    if loop:
        # a callback from another thread to run in the main event thread
        loop.call_soon_threadsafe(received_text_from_app.set)
        print(f"Received text: {text}")
    return "OK", 200

def run_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)


# ---------- Audio Handling ----------------
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    q.put(indata.copy())

def record_audio_sync():
    device_info = sd.query_devices(None, "input")
    samplerate = int(device_info["default_samplerate"])
    frames = []

    with sd.InputStream(samplerate=samplerate, channels=channels,
                        dtype=dtype, callback=audio_callback,
                        blocksize=CHUNK):
        print("Recording... Press stop or cancel to end.")
        while not record_stop_event.is_set() and not cancel_event.is_set():
            try:
                data = q.get(timeout=0.5)
                frames.append(data)
            except queue.Empty:
                continue

    print("Recording stopped. Saving...")
    audio = np.concatenate(frames, axis=0)
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

    return frames

async def record_audio():
    # worker threads that can run a blocking code in parallel
    # spin up a temporary thread to run blocking code 
    with ThreadPoolExecutor() as executor:
        # gets the current event loop
        loop = asyncio.get_event_loop()
        # run the record_audio_sync in a background thread without blocking the event loop
        # and await the result (wait until the completion)
        return await loop.run_in_executor(executor, record_audio_sync)

# ---------- Camera Capture ----------------
def capture_image():
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture image from webcam")
        return None
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

# ---------- Text to Speech ----------------


       
       
       
async def text_to_speech(text):
    global speech_task_event
    speech_task_event.set()

    speech_file_path = "speech.wav"
    model = "playai-tts"
    voice = "Cheyenne-PlayAI"
    response_format = "wav"

    try:
        # Generate speech audio via remote model and save to file
        response = client_groq.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format
        )
       
        audio_bytes = io.BytesIO(response.content)

        # Load the WAV file
        data, samplerate = sf.read(audio_bytes, dtype='int16')

        # Play audio asynchronously
        sd.play(data, samplerate=samplerate, blocking=False)

        # Wait for playback to finish or be cancelled
        while sd.get_stream().active:
            await asyncio.sleep(0.1)
            if cancel_event.is_set():
                sd.stop()
                raise asyncio.CancelledError

    except asyncio.CancelledError:
        sd.stop()
        raise

    finally:
        speech_task_event.clear()
        if os.path.exists(speech_file_path):
            os.remove(speech_file_path)  # Clean up the temporary file
       
       
       
       
       
# ---------- Task Determination -----------------
def task_selection(text):
   
    input_message = f"Classify the following user input into one of these categories by providing only the corresponding number. \
                    Do not include any additional text or explanation. User Input: {recognized_text}. Categories: \
                    1. **Live Task Execution:** The user wants an action performed immediately based on a detected event or condition. (e.g., Tell me when you see a cat, Notify me if the light turns red.) \
                    2. **Image Analysis/Question:** The user is requesting a description of an image, or asking a question about its content. (e.g., Describe what's in front of me, Is there a laptop in this picture?, Whats in front of me?, Can you see a tree?) \
                    3. **General Conversation/Information Retrieval:** The user is engaging in a general conversation or asking a factual question. (e.g., What's the weather like?,Tell me a joke,Who is the prime minister?) \
                    4. **Image Reading:** The user wants something read or some information from text in the image. (e.g., Can you read this sign?, Does this have gluten?, What are the ingredients in this?, How much salt is in this?)  \
                    5. **Help:** The user wants an explanation of the functions available (e.g., Help)"

   
    chat_completion = client_groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="llama-3.3-70b-versatile",)
       
    return chat_completion.choices[0].message.content
   


# ---------- Tasks -----------------
## Task 1 -> threads
def object_searching(recognized_text, loop):
    global vision_model, cancel_event, speech_task, speech_task_event
   
    object_found = False
   
    chat_completion = client_groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Tell me, if there is one, what object is being searched for in this input: {recognized_text}, only tell me the object no additional information or tell me 'no object' if there was no object given",
            }
        ],
        model="llama-3.3-70b-versatile",)
       
    searching = chat_completion.choices[0].message.content
    print("User searching for ", searching)
   
    match = re.search(r"no object", searching, re.IGNORECASE)
    if match:
        print("No object in text")
        return
   
   
    while not object_found:
        if vision_model:
            model_running = "meta-llama/llama-4-maverick-17b-128e-instruct"
            vision_model = 0
        else:
            model_running = "meta-llama/llama-4-scout-17b-16e-instruct"
            vision_model = 1
       
       
        if cancel_event.is_set():
            return
       
        image_bytes = capture_image()
        if image_bytes and searching.strip():
            base_64_image = base64.b64encode(image_bytes).decode('utf-8')
           
            try:
                print("Sending to Model")
                completion = client_groq.chat.completions.create(
                    model=model_running,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Does this image contain {searching}, answer only with yes or no and if the answer is yes tell me where it is using clockface coordinates using only between 10 and 2 o'clock with 10 o'clock being leftmost and 2 o'clock being rightmost and its relative position with respect to other objects, give distance estimate in metres"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base_64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
               
                response = completion.choices[0].message.content
                print(f"Groq Response: {response}")
               
                match = re.search(r"yes", response, re.IGNORECASE)
               
                if match:
                    print(f"Found {searching}")
                    object_found = True
                    while speech_task_event.is_set():
                        print("Waiting")
                        time.sleep(1)
                    speech_task = asyncio.run_coroutine_threadsafe(text_to_speech(f"Found {searching}. {response}"), loop)
                    return
                else:
                    time.sleep(10)
            except Exception as e:
                print(f"Error from Groq API: {e}")

## Task 2
async def image_description(text):
    """
    This task captures an image, sends the image to Groq with user-provided instructions and uses the 
    response to initiate a text to speech task.

    Args:
        text (str): prompt to be included in the query to Groq. 
    """
    global speech_task, vision_model, speech_task_event
   
    # Captures an image from the camera
    image_bytes = capture_image()
   
    # Alternates between two model variants
    if vision_model:
        model_running = "meta-llama/llama-4-maverick-17b-128e-instruct"
        vision_model = 0
    else:
        model_running = "meta-llama/llama-4-scout-17b-16e-instruct"
        vision_model = 1
   
   
    if image_bytes:
            # Encodes image in base 64
            base_64_image = base64.b64encode(image_bytes).decode('utf-8')
           
            try:
                print("Sending to Model")
                # Sends user instructions with image to model
                completion = client_groq.chat.completions.create(
                    model=model_running,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Answer this about the given image in less than 50 words, answer only the following: {recognized_text}"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base_64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                )

                # Extracts response text from model output
                response = completion.choices[0].message.content
                print(f"Groq Response: {response}")

                # Wait for any currently running speech tasks to finish
                while speech_task_event.is_set():
                    await asyncio.sleep(0.1)

                # Start new text to speech task with model response
                speech_task = asyncio.create_task(text_to_speech(response))

            # Handle and report any API related errors
            except Exception as e:
                print(f"Error from Groq API: {e}")

## Task 3
async def google_searching(recognized_text):
    """
    Sends an image and user prompt to Gemini with google search grounding. Uses the 
    response to initiate a text to speech task.
    
    Args:
        recognized_text (str): prompt to be included in the query to Gemini.
    """
    global speech_task, speech_task_event

    # Captures an image from the camera
    image_bytes = capture_image()

    # Creates a grounding tool with google search capabilities
    grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
    )
   
    # Configure generation settings
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )
   
    try:
        # Send the image and user query to the Gemini model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    f'Answer this question directly in less than 50 words, use the image if its required: {recognized_text}'
                    ],
            config=config,
        )

        # Print the grounded response
        print(response.text)

        # Wait for any currently running speech tasks to finish
        while speech_task_event.is_set():
            await asyncio.sleep(0.1)
        
        # Start new text to speech task with model response
        speech_task = asyncio.create_task(text_to_speech(response.text))

    # Handle and report any API related errors
    except Exception as e:
        print(f"Error from Gemini API: {e}")

## Task 4
async def text_description(recognized_text):
    """
    Sends an image of text and user prompt to Groq VLM. Uses the response to initiate a text to speech task.
    
    Args:
        recognized_text (str): prompt to be included in the query to Gemini.
    """
    global vision_model, speech_task_event

    # Captures an image from the camera
    cropped_image = capture_image()
   
    # Alternate between two model variants
    if vision_model:
        model_running = "meta-llama/llama-4-maverick-17b-128e-instruct"
        vision_model = 0
    else:
        model_running = "meta-llama/llama-4-scout-17b-16e-instruct"
        vision_model = 1
   
    if cropped_image:
        # Encodes image in base 64
        base_64_image = base64.b64encode(cropped_image).decode('utf-8')
       
        try:
            print("Sending to Model")
            # Send the image and prompt to the selected model
            completion = client_groq.chat.completions.create(
                model=model_running,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Answer this about the given image in less than 50 words, answer only the following: {recognized_text}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base_64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            # Extracts response text from model output
            response = completion.choices[0].message.content
            print(f"Groq Response: {response}")

            # Wait for any currently running speech tasks to finish
            while speech_task_event.is_set():
                await asyncio.sleep(0.1)

            # Start new text to speech task with model response
            speech_task = asyncio.create_task(text_to_speech(response))
        
        # Handle and report any API related errors
        except Exception as e:
            print(f"Error from Groq API: {e}")
               
               
               
## Task 5
async def help_function():
    """
    Triggers a text to speach task with predetermined 'help' message. 
    """
    global speech_task_event
    
    # Predefined help message
    text = "Help task reached, explain different functions / kind of inputs expected"

    # Wait for any currently running speech tasks to finish
    while speech_task_event.is_set():
            await asyncio.sleep(0.1)
    
    # Start new text to speech task with model response
    speech_task = asyncio.create_task(text_to_speech(text))
   

# ---------- Flask Trigger -----------------
async def watch_flask_trigger():
    global state, image_bytes, recognized_text, app_text, speech_task, speech_task_event
   
    while True:
        # flag that is set when text comes in
        await received_text_from_app.wait()
       
        received_text_from_app.clear()
        cancel_event.clear()
        record_stop_event.clear()
        speech_task_event.clear()
       
        print("Flask Trigger")
        image_bytes = capture_image()
        recognized_text = app_text.strip()
       
        task = task_selection(recognized_text)
        print("Task: ", task)
       
                   
        if task == "1":
            searching_thread = threading.Thread(target=object_searching, args=(recognized_text, loop), daemon=True)
            searching_thread.start()
        elif task == "2":
            await image_description(recognized_text.strip())
        elif task == "3":
            await google_searching(recognized_text.strip())
        elif task == "4":
            await text_description(recognized_text.strip())
        elif task == "5":
            await help_function()
      
       
        state = 0
        app_text = None
               

# ---------- Button Handlers ---------------
async def handle_main_button(loop):
    global state, image_bytes, recognized_text, recording_task, speech_task, searching_thread, speech_task_event

    if state == 0:
        print("Starting voice input...")
        # ~ image_bytes = capture_image()
        state = 1
        cancel_event.clear()
        record_stop_event.clear()
        # schedule a task on the event loop as soon as possible
        recording_task = asyncio.create_task(record_audio())

    elif state == 1:
        record_stop_event.set()
        await recording_task

        if cancel_event.is_set():
            speech_task_event.clear()
            print("Recording cancelled")
            state = 0
            return

        ### --- Local  Whisper TTS
        # ~ model = whisper.load_model("tiny.en")
        # ~ result = model.transcribe("output.wav")
        # ~ recognized_text = result["text"]
       
       
        #### Cloud Groq Whisper tts
        filename = os.path.dirname(__file__) + "/output.wav"
       
        with open(filename, "rb") as file:
            transcription = client_groq.audio.transcriptions.create(
                  file=file, # Required audio file
                  model="whisper-large-v3-turbo", # Required model to use for transcription
                  prompt="Specify context or spelling",  # Optional
                )
            recognized_text = transcription.text
       
        print(f"Recognised text: {recognized_text}")
       
        task = task_selection(recognized_text).strip()
        print("Task: ", task)
       
        if not recognized_text.strip():
            print("No text found")
            state = 0
            return
           
        if task == "1":
            searching_thread = threading.Thread(target=object_searching, args=(recognized_text, loop), daemon=True)
            searching_thread.start()
        elif task == "2":
            await image_description(recognized_text.strip())
        elif task == "3":
            await google_searching(recognized_text.strip())
        elif task == "4":
            await text_description(recognized_text.strip())
        elif task == "5":
            await help_function()

        state = 0

async def handle_cancel_button():
    global state, recording_task, speech_task
    print("Cancel pressed. Aborting operation.")
    cancel_event.set()
    record_stop_event.set()

    if recording_task and not recording_task.done():
        recording_task.cancel()

    if speech_task and not speech_task.done():
        speech_task.cancel()
        try:
            await speech_task
        except asyncio.CancelledError:
            print("Speech cancelled")

    state = 0

# ---------- GPIO Button Watching ----------
def watch_gpio_button(loop, chip_path, line_offsets):
    config = {offset: gpiod.LineSettings(edge_detection=Edge.RISING, debounce_period=timedelta(milliseconds=20)) for offset in line_offsets}
    with gpiod.request_lines(chip_path, consumer="voice-image-button", config=config) as request:
        while True:
            for event in request.read_edge_events():
                if event.line_offset == 2:
                    # This method allows us to schedule the given task which may be running on a different thread
                    # back on the original thread
                    asyncio.run_coroutine_threadsafe(handle_main_button(loop), loop)
                elif event.line_offset == 3:
                    asyncio.run_coroutine_threadsafe(handle_cancel_button(), loop)



# ---------- Main Entry --------------------
async def main():
    global loop, state, recording_task, cancel_event, record_stop_event
   
    loop = asyncio.get_running_loop()
   
    gpio_thread = threading.Thread(target=watch_gpio_button, args=(loop, "/dev/gpiochip0", [2, 3]), daemon=True)
    gpio_thread.start()
   
    if os.path.getsize("saved_mac.txt") == 0:
        print("No saved device")
    else:
        with open("saved_mac.txt", 'r') as file:
            content = file.read()
            pair_trust_connect(content)
   
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
 
    state = 0
    recording_task = None
    cancel_event.clear()
    record_stop_event.clear()

    try:
        await watch_flask_trigger()
    except Exception as e:
        print(f"GPIO error: {e}")

if __name__ == "__main__":
    with open("/home/ver/log_out.txt", "a") as f:
        f.write("Started")
    try:
        asyncio.run(main())
    finally:
        webcam.release()
