#!/usr/bin/env python3
"""
Jarvis AI Agent – Fully local, voice‑controlled assistant.
Runs on Windows / Linux / macOS.
"""

import os
import sys
import json
import time
import random
import subprocess
import tempfile
from datetime import datetime

# Voice & audio
import speech_recognition as sr
import pyttsx3
import pyaudio
import numpy as np

# Wake word
from openwakeword.model import Model

# Optional: custom GPT model (if you have a trained .pth file)
try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not installed. Custom GPT model disabled. Falling back to rule‑based mode.")

# ==============================================
# 1. CONFIGURATION
# ==============================================

WAKE_WORD_MODEL_URL = "https://github.com/dscripka/openWakeWord/raw/main/models/hey_jarvis_v0.1.onnx"
WAKE_WORD_MODEL_FILE = "hey_jarvis_v0.1.onnx"
CUSTOM_MODEL_FILE = "jarvis_model.pth"      # trained from Colab
TOKENIZER_FILE = "tokenizer.pkl"            # saved from Colab
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 100ms for wake word

# ==============================================
# 2. TEXT‑TO‑SPEECH (offline)
# ==============================================

engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('volume', 0.9)

def speak(text):
    """Convert text to speech (offline)."""
    print(f"Jarvis: {text}")
    engine.say(text)
    engine.runAndWait()

# ==============================================
# 3. SPEECH‑TO‑TEXT (free Google API)
# ==============================================

recognizer = sr.Recognizer()
microphone = sr.Microphone()

def listen_for_command(timeout=5):
    """Listen for a single command (after wake word). Returns text or empty string."""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            print("[INFO] Listening for command...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            return ""
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"[User] {text}")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        speak("I'm having trouble reaching the speech service. Check your internet.")
        return ""

# ==============================================
# 4. WAKE WORD DETECTION (openWakeWord)
# ==============================================

def download_wakeword_model():
    """Download the 'Hey Jarvis' model if not present."""
    if os.path.exists(WAKE_WORD_MODEL_FILE):
        return
    print("[INFO] Downloading wake word model...")
    import urllib.request
    urllib.request.urlretrieve(WAKE_WORD_MODEL_URL, WAKE_WORD_MODEL_FILE)
    print("[INFO] Download complete.")

def init_wake_word():
    """Initialize openWakeWord model and audio stream."""
    download_wakeword_model()
    model = Model(wakeword_models=[WAKE_WORD_MODEL_FILE])
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    return model, stream, pa

def listen_for_wake_word(model, stream):
    """Continuously listen for wake word. Returns True when detected."""
    while True:
        audio_chunk = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)
        prediction = model.predict(audio_chunk)
        # The key is the model name without .onnx
        if prediction[WAKE_WORD_MODEL_FILE.replace('.onnx', '')] > 0.5:
            return True

# ==============================================
# 5. CUSTOM GPT MODEL (if available)
# ==============================================

class SimpleJarvisGPT(nn.Module):
    """Minimal GPT‑like model (same architecture as Colab notebook)."""
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[self._make_block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _make_block(self, n_embd, n_head):
        return nn.TransformerDecoderLayer(n_embd, n_head, batch_first=True, dropout=0.1)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        x = self.blocks(x, tgt_mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=0.7):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def load_custom_model():
    """Load the model and tokenizer if they exist."""
    if not (os.path.exists(CUSTOM_MODEL_FILE) and os.path.exists(TOKENIZER_FILE) and TORCH_AVAILABLE):
        return None, None, None
    import pickle
    with open(TOKENIZER_FILE, 'rb') as f:
        stoi, itos = pickle.load(f)
    vocab_size = len(stoi)
    model = SimpleJarvisGPT(vocab_size)
    model.load_state_dict(torch.load(CUSTOM_MODEL_FILE, map_location='cpu'))
    model.eval()
    return model, stoi, itos

def generate_response_custom(prompt, model, stoi, itos, max_tokens=80):
    """Generate response using custom GPT."""
    if model is None:
        return None
    # encode prompt
    prompt = prompt.lower()
    # simple character‑level encode
    context = torch.tensor([stoi.get(ch, 0) for ch in prompt], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=max_tokens)
    response = ''.join([itos[int(i)] for i in out[0].tolist()])
    # extract after "jarvis:"
    if "jarvis:" in response:
        response = response.split("jarvis:")[-1].strip()
    # remove trailing "user:" part
    if "user:" in response:
        response = response.split("user:")[0].strip()
    return response

# ==============================================
# 6. FALLBACK RESPONDER (rule‑based)
# ==============================================

def fallback_response(command):
    """Simple rule‑based responses when custom model is not available."""
    cmd = command.lower()
    if "weather" in cmd:
        return "I cannot fetch live weather without an API key, but it looks pleasant outside."
    elif "time" in cmd:
        now = datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}."
    elif "light" in cmd or "lights" in cmd:
        if "on" in cmd:
            return "[LIGHT_ON] I have turned on the lights."
        elif "off" in cmd:
            return "[LIGHT_OFF] I have turned off the lights."
        else:
            return "Please specify on or off."
    elif "joke" in cmd:
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything.",
            "What do you call a fake noodle? An impasta.",
            "Why did the scarecrow win an award? He was outstanding in his field."
        ]
        return random.choice(jokes)
    elif "who are you" in cmd or "your name" in cmd:
        return "I am Jarvis, your personal AI assistant, running entirely on your local machine."
    elif "exit" in cmd or "quit" in cmd or "goodbye" in cmd:
        return "[EXIT]"
    else:
        return "I'm sorry, I didn't understand that. You can train a custom model to improve me."

# ==============================================
# 7. ACTION EXECUTOR (home automation)
# ==============================================

def execute_action(response_text):
    """
    Parse actions from LLM response (e.g., [LIGHT_ON living_room]).
    Replace this with real API calls to your smart home devices.
    """
    if "[LIGHT_ON" in response_text:
        # Example: send HTTP request to a smart bulb
        print("[ACTION] Turning lights ON")
        # requests.get("http://192.168.1.100/on")
        return True
    elif "[LIGHT_OFF" in response_text:
        print("[ACTION] Turning lights OFF")
        return True
    return False

# ==============================================
# 8. MAIN LOOP
# ==============================================

def main():
    print("=" * 50)
    print("Jarvis Agent starting...")
    print("=" * 50)

    # Load custom GPT if available
    model, stoi, itos = load_custom_model()
    if model:
        speak("Custom Jarvis brain loaded. I am ready, sir.")
    else:
        speak("Jarvis is online in basic mode. Train a custom model for better conversations.")

    # Initialize wake word
    try:
        wake_model, audio_stream, pyaudio_handle = init_wake_word()
        print("[INFO] Wake word 'Hey Jarvis' active.")
    except Exception as e:
        print(f"[ERROR] Wake word init failed: {e}")
        speak("Wake word engine failed. Falling back to button trigger.")
        wake_model = None

    # Optional: one‑time ambient noise adjustment
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    # Main loop
    try:
        while True:
            # Wait for wake word (if available)
            if wake_model:
                print("[INFO] Say 'Hey Jarvis'...")
                listen_for_wake_word(wake_model, audio_stream)
                speak("Yes?")
            else:
                # fallback: manual trigger via keyboard
                input("Press Enter to start listening...")

            # Get command
            command = listen_for_command()
            if not command:
                speak("I didn't hear anything.")
                continue

            # Generate response
            if model and stoi and itos:
                prompt = f"user: {command}\njarvis:"
                response = generate_response_custom(prompt, model, stoi, itos)
                if response is None:
                    response = fallback_response(command)
            else:
                response = fallback_response(command)

            # Check for exit command
            if response == "[EXIT]":
                speak("Goodbye, sir.")
                break

            # Execute any actions embedded in the response
            execute_action(response)

            # Remove action markers from spoken response
            clean_response = response
            for marker in ["[LIGHT_ON", "[LIGHT_OFF"]:
                if marker in clean_response:
                    clean_response = clean_response.split(marker)[0].strip()
            if clean_response:
                speak(clean_response)

    except KeyboardInterrupt:
        speak("Shutting down.")
    finally:
        if wake_model:
            audio_stream.stop_stream()
            audio_stream.close()
            pyaudio_handle.terminate()
        print("[INFO] Jarvis terminated.")

if __name__ == "__main__":
    main()