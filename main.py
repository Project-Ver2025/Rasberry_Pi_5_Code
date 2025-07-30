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
import soundfile as sf
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
    """
    Route for the homepage. 
    Renders the 'index.html' template to display the main interface of the web application.
    """
    
    return render_template('index.html') # imported render_template function from flask library
   
@app.route('/scan')
def scan_devices():
    """
    Scans for nearby Bluetooth devices. 
    
    Returns:
        JSON list of discovered Bluetooth devices with their names and addresses.
    """
    
    print("Scanning for Bluetooth devices...")
    nearby_devices = bluetooth.discover_devices(duration=5, lookup_names=True)     # imported blutooth library function
    devices = [{'address': addr, 'name': name} for addr, name in nearby_devices]   # list of avalible device names and address   
    print(f"Found {len(devices)} device(s)")
    return jsonify(devices)



@app.route('/select_device', methods=['POST'])
def select_device():
    """
    Receives the selected Bluetooth device mac address from the frontend app or website.
    Saves it to a file and attempts to pair, trust, and connect to it.

    Returns:
        HTTP 200 if successful, 500 if an error occurs during connection.
    """
    
    data = request.get_json()
    address = data.get('address')
    try:
        with open("saved_mac.txt", 'w') as file:
            file.write(address)
        pair_trust_connect(address)
        return "OK", 200
    except subprocess.CalledProcessError:
        return "Failed", 500



# ---------- Phone ---------------------??? app??? - what is the difference to the function above?
@app.route('/receive_mac', methods=['POST'])
def receive_mac():
    """
    Receives a MAC address from phone client.
    Saves the MAC address and attempts to pair, trust, and connect.

    Returns:
        HTTP 200 if successful, 500 if pairing fails.
    """
    
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
    """
    Receives text data from the app (e.g., command or instruction).
    If event loop is active, signal event for main thread to process the text.

    Returns:
        HTTP 200 acknowledgment.
    """
    
    global app_text, received_text_from_app, loop
    text = request.data.decode()
    app_text = text
    if loop:
        # Signal the main thread to process the received text
        loop.call_soon_threadsafe(received_text_from_app.set)
        print(f"Received text: {text}")
    return "OK", 200



def run_flask():
    """
    Starts the Flask web server on all available IP addresses (0.0.0.0) using port 5000.
    Disables auto-reloading for better thread safety.
    """
    
    app.run(host='0.0.0.0', port=5000, use_reloader=False)


# ---------- Audio Handling ----------------
def audio_callback(indata, frames, time, status):
    """
    Callback function for the audio stream.
    
    Args:
        indata (numpy.ndarray): Audio input buffer.
        frames (int): Number of frames.
        time: Time information.
        status: Status of the audio input.

    Adds the incoming audio buffer to the shared queue if status is active.
    """
    
    if status:
        print(f"Audio status: {status}")
    q.put(indata.copy())



def record_audio_sync():
     """
    Synchronously records audio until either the stop or cancel event is set.
    Utilises the audio_callback function within the Input Stream.
    
    Returns:
        List of numpy arrays containing audio data chunks.
    """
    
    device_info = sd.query_devices(None, "input")
    samplerate = int(device_info["default_samplerate"])
    frames = []

    with sd.InputStream(samplerate=samplerate, channels=channels,
                        dtype=dtype, callback=audio_callback,
                        blocksize=CHUNK):
        print("Recording... Press stop or cancel to end.")
        while not record_stop_event.is_set() and not cancel_event.is_set():
            try:
                # get data from the Input Stream that utilises audio_callback
                data = q.get(timeout=0.5)
                # append this data to the array of audio frames to be combined later
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
    """
    Asynchronously records audio using a thread pool executor to avoid blocking the event loop for the rest of the devices execution
    
    Utilises the record_audio_sync function.
    
    Returns:
        Audio frames recorded by `record_audio_sync()`.
    """

    with ThreadPoolExecutor() as executor:
        # gets the current event loop
        loop = asyncio.get_event_loop()
        # run the record_audio_sync in a background thread without blocking the event loop and await the result (wait until the completion)
        return await loop.run_in_executor(executor, record_audio_sync)



# ---------- Camera Capture ----------------
def capture_image():
    """
    Captures a single frame from the webcam and encodes it as JPEG bytes.

    Returns:
        Bytes of the JPEG-encoded image if successful, None otherwise.
    """
    
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture image from webcam")
        return None
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()




# ---------- Text to Speech ----------------
       
async def text_to_speech(text):
    """
    This function takes a response string that is to be read to the user and converts it to audio bytes to be streamed over
    Bluetooth to the connected audio device.
    
    Args:
        text (str): The response string text to be converted to audio by the model. 

    """
    global speech_task_event
    speech_task_event.set()

    # Set up the file to save audio to and the speech model parameters
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
       
       # Save the model response as audio bytes so that it can be read
        audio_bytes = io.BytesIO(response.content)

        # Load the WAV file
        data, samplerate = sf.read(audio_bytes, dtype='int16')

        # Play audio asynchronously over bluetooth
        sd.play(data, samplerate=samplerate, blocking=False)

        # Wait for playback to finish or be cancelled
        while sd.get_stream().active:
            await asyncio.sleep(0.1)
            if cancel_event.is_set(): # Cancel the audio when the cancel button is pressed
                sd.stop()
                raise asyncio.CancelledError

    except asyncio.CancelledError:
        sd.stop() # Stop the bluetooth stream if audio reading fails
        raise

    finally:
        # Reading text task finished so clear the event
        speech_task_event.clear()
        if os.path.exists(speech_file_path):
            os.remove(speech_file_path)  # Clean up the temporary file
       
       
       
       
       
# ---------- Task Determination -----------------
def task_selection(text):
    """
    Converts the text converted user query to a use case classification using groq.
    
    Args:
        text (str): Unused. 

    Returns:
        A string of a number from 1-5 representing the task classification of the user query.
    """
    # The question that is asked to the model to classify the user's text string into a use case.
    # recognized_text is a global containig the query text string converted from audio recorded from the user.
    input_message = f"Classify the following user input into one of these categories by providing only the corresponding number. \
                    Do not include any additional text or explanation. User Input: {recognized_text}. Categories: \
                    1. **Live Task Execution:** The user wants an action performed immediately based on a detected event or condition. (e.g., Tell me when you see a cat, Notify me if the light turns red.) \
                    2. **Image Analysis/Question:** The user is requesting a description of an image, or asking a question about its content. (e.g., Describe what's in front of me, Is there a laptop in this picture?, Whats in front of me?, Can you see a tree?) \
                    3. **General Conversation/Information Retrieval:** The user is engaging in a general conversation or asking a factual question. (e.g., What's the weather like?,Tell me a joke,Who is the prime minister?) \
                    4. **Image Reading:** The user wants something read or some information from text in the image. (e.g., Can you read this sign?, Does this have gluten?, What are the ingredients in this?, How much salt is in this?)  \
                    5. **Help:** The user wants an explanation of the functions available (e.g., Help)"

    # Send the user query to qroq to generate a classification
    chat_completion = client_groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="llama-3.3-70b-versatile",)
    
    # Return the string of the task classification number.
    return chat_completion.choices[0].message.content
   


# ---------- Tasks -----------------
## Task 1 -> threads
def object_searching(recognized_text, loop):
    """
    Determines what object the user is looking for based on their text query and takes an image of their environment to search for it.
    Alerts the user when the object is found.
    
    Args:
        recognized_text (str): The converted text of the user query to search for an object.
        loop (asyncio): The event loop that runs on a separate thread to read the model response to the user when the object is found.

    """
    global vision_model, cancel_event, speech_task, speech_task_event
   
    object_found = False
   
    # Send the user query to find an object and to qroq and generate a response of what is being searched for
    chat_completion = client_groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Tell me, if there is one, what object is being searched for in this input: {recognized_text}, only tell me the object no additional information or tell me 'no object' if there was no object given",
            }
        ],
        model="llama-3.3-70b-versatile",)
    
    # Extract the response from the model and print it to consol
    searching = chat_completion.choices[0].message.content
    print("User searching for ", searching)
    
    # Check if no object to search for is identified in the user query and return
    match = re.search(r"no object", searching, re.IGNORECASE)
    if match:
        print("No object in text")
        return
   
    # Configure the vision model to use to search for the object (switch between them to avoid runnion out of query allowance).
    while not object_found:
        if vision_model:
            model_running = "meta-llama/llama-4-maverick-17b-128e-instruct"
            vision_model = 0
        else:
            model_running = "meta-llama/llama-4-scout-17b-16e-instruct"
            vision_model = 1
       
        # Stop searching if the cancel button is pressed
        if cancel_event.is_set():
            return
        
        # Take an image from the camera and convert it to base 64 encoding
        image_bytes = capture_image()
        if image_bytes and searching.strip():
            base_64_image = base64.b64encode(image_bytes).decode('utf-8')
           
            try:
                print("Sending to Model")
                # Boiler plate to setup groq for searching
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
               
                # Extract the model response
                response = completion.choices[0].message.content
                print(f"Groq Response: {response}")
               
                # Check if the desired object has been detected in the image
                match = re.search(r"yes", response, re.IGNORECASE)
               
                # Alert the user the object has been found by creating an text response that is converted to audio using text_to_speech()
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
