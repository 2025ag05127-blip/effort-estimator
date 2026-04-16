#!/usr/bin/env python3
"""
Jarvis Smart Agent – with internet tools (search, stocks, weather, news).
Supports learning mode (DeepSeek teacher) and standalone fine‑tuned mode.
"""

import os
import time
import random
import json
import requests
import yfinance as yf
from datetime import datetime

# Voice & audio
import speech_recognition as sr
import pyttsx3
import pyaudio
import numpy as np

# Wake word
from openwakeword.model import Model

# DeepSeek / OpenAI for learning mode
import openai

# Optional: custom GPT model after fine-tuning
try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ==============================================
# 0. CONFIGURATION
# ==============================================
LEARNING_MODE = True                # True = learn from DeepSeek, False = use fine-tuned model
DEEPSEEK_API_KEY = "sk-faf9d8d30a234128bbc64fb14562a9d1"   # <-- YOUR DEEPSEEK API KEY (if LEARNING_MODE)
TRAINING_LOG = "training_data.txt"

# Tool API keys (free tiers)
BRAVE_API_KEY = ""                  # Get from https://brave.com/search/api/ (optional)
GNEWS_API_KEY = ""                  # Get from https://gnews.io/ (optional)

# Model files
WAKE_WORD_MODEL_URL = "https://github.com/dscripka/openWakeWord/raw/main/models/hey_jarvis_v0.1.onnx"
WAKE_WORD_MODEL_FILE = "hey_jarvis_v0.1.onnx"
CUSTOM_MODEL_FILE = "jarvis_model_finetuned.pth"
TOKENIZER_FILE = "tokenizer.pkl"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600

# ==============================================
# 1. TEXT-TO-SPEECH
# ==============================================
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('volume', 0.9)

def speak(text):
    print(f"Jarvis: {text}")
    engine.say(text)
    engine.runAndWait()

# ==============================================
# 2. SPEECH-TO-TEXT
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
# 3. WAKE WORD
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
# 4. INTERNET TOOLS (smart mode)
# ==============================================
def search_web(query):
    """Brave Search API (free tier)"""
    if not BRAVE_API_KEY:
        return "Web search not configured. Please add your Brave API key."
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "count": 3}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get('web', {}).get('results', [])
        if not results:
            return "No results found."
        return "\n".join([f"• {r['title']}: {r['url']}" for r in results[:3]])
    except Exception as e:
        return f"Search error: {e}"

def get_stock_price(symbol):
    try:
        ticker = yf.Ticker(symbol.upper())
        price = ticker.info.get('regularMarketPrice')
        if not price:
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
        return f"{symbol.upper()} is currently ${price:.2f}"
    except:
        return "Could not fetch stock price."

def get_weather(city="London"):
    """Open-Meteo geocoding + weather (no API key)"""
    # Geocode city to lat/lon
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    try:
        geo_resp = requests.get(geo_url, timeout=10).json()
        if not geo_resp.get('results'):
            return f"City '{city}' not found."
        lat = geo_resp['results'][0]['latitude']
        lon = geo_resp['results'][0]['longitude']
        # Weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w = requests.get(weather_url, timeout=10).json()['current_weather']
        temp = w['temperature']
        wind = w['windspeed']
        return f"In {city}, it's {temp}°C, wind speed {wind} km/h."
    except Exception as e:
        return f"Weather error: {e}"

def get_news(topic="technology"):
    """GNews API – free tier"""
    if not GNEWS_API_KEY:
        return "News API key missing."
    url = f"https://gnews.io/api/v4/search?q={topic}&lang=en&max=3&apikey={GNEWS_API_KEY}"
    try:
        resp = requests.get(url, timeout=10).json()
        articles = resp.get('articles', [])
        if not articles:
            return "No news found."
        headlines = [f"• {a['title']}" for a in articles]
        return "\n".join(headlines)
    except:
        return "News unavailable."

def dispatch_tool(command):
    """Route user command to appropriate tool"""
    cmd = command.lower()
    if "search for" in cmd:
        query = command.split("search for")[-1].strip()
        return search_web(query)
    elif "stock price of" in cmd or "stock" in cmd and "price" in cmd:
        # extract symbol
        parts = command.split()
        for i, p in enumerate(parts):
            if p.lower() in ("of", "price"):
                if i+1 < len(parts):
                    symbol = parts[i+1].strip().upper()
                    return get_stock_price(symbol)
        return "Please specify a stock symbol, e.g., 'stock price of AAPL'."
    elif "weather" in cmd:
        # extract city after "in" or "weather"
        if "in" in cmd:
            city = cmd.split("in")[-1].strip()
        else:
            city = "London"
        return get_weather(city)
    elif "news" in cmd:
        topic = cmd.replace("news", "").strip()
        if not topic:
            topic = "technology"
        return get_news(topic)
    return None   # no tool matched

# ==============================================
# 5. DEEPSEEK TEACHER (for learning mode)
# ==============================================
openai.api_key = DEEPSEEK_API_KEY
openai.base_url = "https://api.deepseek.com/v1"

def teacher_response_deepseek(command):
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are Jarvis, a helpful assistant. Keep responses concise (1-2 sentences)."},
                {"role": "user", "content": command}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DeepSeek error] {e}")
        return "DeepSeek unavailable."

# ==============================================
# 6. CUSTOM MODEL (after fine-tuning)
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
# 7. FALLBACK (when no model and no tool)
# ==============================================
def fallback_response(command):
    cmd = command.lower()
    if "time" in cmd:
        return datetime.now().strftime("The time is %I:%M %p.")
    elif "exit" in cmd or "quit" in cmd:
        return "[EXIT]"
    else:
        return "I'm in basic mode. Train me with DeepSeek or add API keys for tools."

# ==============================================
# 8. MAIN LOOP
# ==============================================
def main():
    print("=" * 50)
    print("Jarvis Smart Agent with Internet Tools")
    print("=" * 50)

    if LEARNING_MODE:
        speak("Learning mode active. Using DeepSeek teacher. Logging to file.")
    else:
        speak("Standalone mode. Loading fine-tuned model...")
        model, stoi, itos = load_custom_model()
        if model:
            speak("Fine-tuned model loaded.")
        else:
            speak("No custom model. Running in basic mode.")

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
                speak("Didn't catch that.")
                continue

            # ---- STEP 1: Try internet tool ----
            tool_result = dispatch_tool(command)
            if tool_result:
                response = tool_result
                print(f"[Tool] {response}")
            else:
                # ---- STEP 2: No tool matched, use teacher or custom model ----
                if LEARNING_MODE:
                    response = teacher_response_deepseek(command)
                    # Log to training file
                    with open(TRAINING_LOG, "a") as f:
                        f.write(f"User: {command}\nJarvis: {response}\n\n")
                else:
                    # Use custom model if available
                    if 'model' in locals() and model:
                        response = generate_response_custom(command, model, stoi, itos)
                        if response is None:
                            response = fallback_response(command)
                    else:
                        response = fallback_response(command)

            if response == "[EXIT]":
                speak("Goodbye.")
                break

            # Remove action markers (if any)
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