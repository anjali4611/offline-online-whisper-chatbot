import tkinter as tk
from tkinter import scrolledtext
import threading
import speech_recognition as sr
import whisper
import pyttsx3
import soundfile as sf
import numpy as np
import librosa
import torch
from datetime import datetime
import sqlite3
import openai
import requests
import os

# -------------------- CONFIG --------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your key in environment variable
DB_FILE = "chat_memory.db"

# -------------------- DATABASE SETUP --------------------
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    bot_response TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# -------------------- Load Whisper Model --------------------
print("🔹 Loading Whisper model (base)...")
model = whisper.load_model("base")
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# -------------------- Internet Check --------------------
def internet_connected():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

# -------------------- Speech to Text --------------------
def recognize_speech():
    print("🎙️ Listening...")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(audio.get_wav_data())

    try:
        # Read audio without ffmpeg
        audio_data, samplerate = sf.read(temp_file, dtype='float32')
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if samplerate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
        audio_tensor = torch.from_numpy(audio_data)

        # Transcribe locally
        result = model.transcribe(audio_tensor, fp16=False)
        text_local = result["text"].strip()
        lang = result.get("language", "en")

        # If online, validate with OpenAI API
        if internet_connected() and openai.api_key:
            print("🌐 Validating with online Whisper...")
            with open(temp_file, "rb") as f:
                validated = openai.Audio.transcriptions.create(model="whisper-1", file=f)
                text_online = validated.text.strip()
            if text_online and text_online.lower() != text_local.lower():
                text = text_online
                print(f"✅ Validated text: {text}")
            else:
                text = text_local
        else:
            text = text_local

        print(f"🧑 You ({lang}): {text}")
        return text, lang

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return "", "en"

# -------------------- Text to Speech --------------------
def speak(text, lang="en"):
    voices = engine.getProperty("voices")
    if "hi" in lang:
        engine.setProperty("voice", voices[1].id)
    else:
        engine.setProperty("voice", voices[0].id)
    engine.say(text)
    engine.runAndWait()

# -------------------- Bot Logic --------------------
def get_bot_response(user_input):
    user_input = user_input.lower()
    response = None

    # Check if similar input exists in memory
    cursor.execute("SELECT bot_response FROM chat_memory WHERE user_input LIKE ?", (f"%{user_input}%",))
    saved = cursor.fetchone()
    if saved:
        return saved[0]

    # Basic responses
    if "hello" in user_input or "hi" in user_input:
        response = "Hi there! How can I help you?"
    elif "your name" in user_input:
        response = "I’m your hybrid voice assistant."
    elif "time" in user_input:
        response = f"The current time is {datetime.now().strftime('%I:%M %p')}."
    elif "bye" in user_input:
        response = "Goodbye! Have a great day!"
    elif "thank" in user_input:
        response = "You’re very welcome!"
    else:
        response = "I'm still learning, but I can understand many languages!"

    # Store to memory
    cursor.execute("INSERT INTO chat_memory (user_input, bot_response) VALUES (?, ?)", (user_input, response))
    conn.commit()
    return response

# -------------------- Voice Input --------------------
def handle_voice_input():
    user_input, lang = recognize_speech()
    if not user_input:
        return
    chat_window.insert(tk.END, f"\n🧑 You ({lang}): {user_input}")
    bot_response = get_bot_response(user_input)
    chat_window.insert(tk.END, f"\n🤖 Bot: {bot_response}")
    speak(bot_response, lang)
    chat_window.yview(tk.END)

# -------------------- Text Input --------------------
def handle_text_input():
    user_input = text_entry.get().strip()
    if not user_input:
        return
    chat_window.insert(tk.END, f"\n🧑 You (text): {user_input}")
    text_entry.delete(0, tk.END)
    bot_response = get_bot_response(user_input)
    chat_window.insert(tk.END, f"\n🤖 Bot: {bot_response}")
    speak(bot_response)
    chat_window.yview(tk.END)

# -------------------- GUI --------------------
root = tk.Tk()
root.title("Hybrid Offline-Online Voice Chatbot")
root.geometry("500x430")
root.config(bg="#202020")

chat_window = scrolledtext.ScrolledText(
    root, wrap=tk.WORD, bg="#101010", fg="#f5f5f5",
    font=("Segoe UI", 12), height=15, width=55
)
chat_window.pack(padx=10, pady=10)
chat_window.insert(tk.END, "🤖 Bot: Hello! Speak or type below (offline + online supported).\n")

frame = tk.Frame(root, bg="#202020")
frame.pack(fill=tk.X, padx=10, pady=5)

text_entry = tk.Entry(frame, font=("Segoe UI", 12), bg="#2b2b2b", fg="white", width=35)
text_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

send_button = tk.Button(frame, text="Send", command=handle_text_input,
                        bg="#28a745", fg="white", font=("Segoe UI", 11, "bold"))
send_button.pack(side=tk.LEFT, padx=5)

voice_button = tk.Button(frame, text="🎤 Speak",
                         command=lambda: threading.Thread(target=handle_voice_input).start(),
                         bg="#007bff", fg="white", font=("Segoe UI", 11, "bold"))
voice_button.pack(side=tk.LEFT, padx=5)

root.mainloop()
