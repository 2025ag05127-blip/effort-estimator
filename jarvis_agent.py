#!/usr/bin/env python3
"""
Jarvis AI Agent – Learning mode uses DeepSeek API as teacher.
Set LEARNING_MODE = True and add your DeepSeek API key.
"""

import os
import time
import random
from datetime import datetime

# Voice & audio
import speech_recognition as sr
import pyttsx3
import pyaudio
import numpy as np

# Wake word
from openwakeword.model import Model

# DeepSeek API (OpenAI-compatible)
import openai

# For custom model after fine-tuning
try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ==============================================
# 0. LEARNING MODE CONFIGURATION
# ==============================================
LEARNING_MODE = True                # True = learn from DeepSeek, False = use fine-tuned model
DEEPSEEK_API_KEY = "sk-faf9d8d30a234128bbc64fb14562a9d1"   # <-- PASTE YOUR DEEPSEEK API KEY
TRAINING_LOG = "training_data.txt"

# ==============================================
# 1. OTHER CONFIGURATION
# ==============================================
WAKE_WORD_MODEL_URL = "https://github.com/dscripka/openWakeWord/raw/main/models/hey_jarvis_v0.1.onnx"
WAKE_WORD_MODEL_FILE = "hey_jarvis_v0.1.onnx"
CUSTOM_MODEL_FILE = "jarvis_model_finetuned.pth"
TOKENIZER_FILE = "tokenizer.pkl"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600

# ==============================================
# 2. TEXT-TO-SPEECH
# ==============================================
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('volume', 0.9)

def speak(text):
    print(f"Jarvis: {text}")
    engine.say(text)
    engine.runAndWait()

# ==============================================
# 3. SPEECH-TO-TEXT
# ==============================================
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def listen_for_command(timeout=5):
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
    except:
        return ""

# ==============================================
# 4. WAKE WORD
# ==============================================
def download_wakeword_model():
    if os.path.exists(WAKE_WORD_MODEL_FILE):
        return
    import urllib.request
    urllib.request.urlretrieve(WAKE_WORD_MODEL_URL, WAKE_WORD_MODEL_FILE)

def init_wake_word():
    download_wakeword_model()
    model = Model(wakeword_models=[WAKE_WORD_MODEL_FILE])
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                     input=True, frames_per_buffer=CHUNK_SIZE)
    return model, stream, pa

def listen_for_wake_word(model, stream):
    while True:
        audio_chunk = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)
        pred = model.predict(audio_chunk)
        if pred[WAKE_WORD_MODEL_FILE.replace('.onnx', '')] > 0.5:
            return True

# ==============================================
# 5. DEEPSEEK TEACHER (temporary)
# ==============================================
openai.api_key = DEEPSEEK_API_KEY
openai.base_url = "https://api.deepseek.com/v1"   # DeepSeek endpoint

def teacher_response_deepseek(command):
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are Jarvis, a helpful AI assistant. Keep responses concise (1-2 sentences)."},
                {"role": "user", "content": command}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DeepSeek error] {e}")
        return "I'm having trouble reaching DeepSeek. Please check your API key and internet."

def get_teacher_response(command):
    return teacher_response_deepseek(command)

# ==============================================
# 6. CUSTOM MODEL (used after fine-tuning)
# ==============================================
class SimpleJarvisGPT(nn.Module):
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
        x = self.token_embedding(idx) + self.position_embedding(pos[:, :T])
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

def generate_response_custom(command, model, stoi, itos, max_tokens=80):
    if model is None:
        return None
    prompt = f"user: {command}\njarvis:"
    context = torch.tensor([stoi.get(ch, 0) for ch in prompt], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=max_tokens)
    response = ''.join([itos[int(i)] for i in out[0].tolist()])
    if "jarvis:" in response:
        response = response.split("jarvis:")[-1].strip()
    if "user:" in response:
        response = response.split("user:")[0].strip()
    return response

# ==============================================
# 7. FALLBACK (very basic)
# ==============================================
def fallback_response(command):
    if "time" in command:
        return datetime.now().strftime("The time is %I:%M %p.")
    elif "exit" in command:
        return "[EXIT]"
    else:
        return "I'm in basic mode. Please train me with DeepSeek first."

# ==============================================
# 8. ACTION EXECUTOR
# ==============================================
def execute_action(response):
    if "[LIGHT_ON" in response:
        print("[ACTION] Lights ON")
    elif "[LIGHT_OFF" in response:
        print("[ACTION] Lights OFF")

# ==============================================
# 9. MAIN LOOP
# ==============================================
def main():
    print("=" * 50)
    print("Jarvis Agent with DeepSeek Learning Mode")
    print("=" * 50)

    if LEARNING_MODE:
        speak("Learning mode active. Using DeepSeek as teacher. All conversations will be logged.")
    else:
        speak("Standalone mode.")
        model, stoi, itos = load_custom_model()
        if model:
            speak("Fine-tuned model loaded.")
        else:
            speak("No custom model found. Run learning mode first.")

    # Wake word
    try:
        wake_model, audio_stream, pyaudio_handle = init_wake_word()
        print("[INFO] Wake word 'Hey Jarvis' active.")
    except:
        speak("Wake word failed. Using manual trigger.")
        wake_model = None

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    try:
        while True:
            if wake_model:
                listen_for_wake_word(wake_model, audio_stream)
                speak("Yes?")
            else:
                input("Press Enter to listen...")

            command = listen_for_command()
            if not command:
                speak("I didn't catch that.")
                continue

            if LEARNING_MODE:
                # Use DeepSeek teacher
                response = get_teacher_response(command)
                # Log to file
                with open(TRAINING_LOG, "a") as f:
                    f.write(f"User: {command}\nJarvis: {response}\n\n")
                print(f"[DeepSeek] {response}")
            else:
                # Use fine-tuned model
                if 'model' in locals() and model:
                    response = generate_response_custom(command, model, stoi, itos)
                    if response is None:
                        response = fallback_response(command)
                else:
                    response = fallback_response(command)

            if response == "[EXIT]":
                speak("Goodbye.")
                break

            execute_action(response)
            clean = response.split("[LIGHT")[0].strip()
            if clean:
                speak(clean)

    except KeyboardInterrupt:
        speak("Shutting down.")
    finally:
        if wake_model:
            audio_stream.stop_stream()
            audio_stream.close()
            pyaudio_handle.terminate()

if __name__ == "__main__":
    main()