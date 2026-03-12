# 🤖 Emotion-Aware AI Companion

A Python application that acts as an empathetic AI companion — it **sees your emotions** through the webcam, **speaks** to you, **listens** to your voice, **sees what you show it** (like Gemini Live!), and **responds conversationally** using free LLMs, adapting its tone based on how you're feeling.

Built entirely with **free and open-source tools**.

---

## ✨ Features

| Feature | Technology |
|---|---|
| Real-time facial emotion detection | OpenCV + FER |
| Speech-to-text (voice input) | SpeechRecognition (Google free API) |
| Text-to-speech (voice output) | pyttsx3 (offline) + gTTS (web) |
| Empathetic conversation AI | Groq API (LLaMA 3.3 70B) |
| 👁️ **Vision analysis** (NEW) | Groq API (LLaMA 4 Scout 17B) |
| 🤖 **Proactive AI observation** (NEW) | AI watches camera & comments |
| 🌍 **20+ languages** | Script detection + langdetect |
| Emotion-aware responses | Facial emotion + text sentiment |
| 3D animated web interface | Three.js + Socket.IO |

---

## 🎥 Demo Features

### Vision Capabilities (Gemini-like)
Show any object, document, or product to the camera and ask:
- "What is this?"
- "Read this"
- "What am I holding?"

### Proactive AI
When idle, the AI observes through your camera and may:
- Comment on interesting objects it sees
- Ask caring questions if you look sad
- Make conversation starters based on what it observes

### Multilingual Support
Speak or type in Telugu (తెలుగు), Hindi (हिंदी), Tamil, Spanish, French, and 15+ more languages. The AI detects the script and responds in the same language!

---

## 🏗️ Architecture

```
User Camera  →  Face Detection  →  Emotion Detection
                                        ↓
                              AI Generates Empathetic Message
                                        ↓
                                  Text-to-Speech
                                        ↓
                              User Responds via Microphone
                                        ↓
                                  Speech-to-Text
                                        ↓
                          Send Message + Emotion Context to LLM
                                        ↓
                              AI Response Generated
                                        ↓
                            Convert Response to Speech
                                    (repeat)
```

---

## 📁 Project Structure

```
AICompanion/
├── main.py                    # CLI entry point with arguments
├── app.py                     # Flask web server (Socket.IO, vision)
├── groq_chat.py               # Groq API conversational AI
├── vision_analyzer.py         # LLaMA 4 Scout vision analysis (NEW)
├── emotion_detector.py        # Webcam + FER emotion detection
├── voice_input.py             # Microphone speech-to-text
├── voice_output.py            # pyttsx3 text-to-speech
├── language_detector.py       # Script + langdetect language detection
├── sentiment_analyzer.py      # Text sentiment analysis
├── session_logger.py          # Conversation logging
├── emotion_analytics.py       # Emotion distribution reports
├── settings.py                # Configuration from .env
├── logger_setup.py            # Logging setup
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API key
├── templates/index.html       # Web UI template
├── static/css/style.css       # Dark glassmorphic theme
├── static/js/app.js           # Three.js + Socket.IO + vision
└── README.md                  # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Webcam** (for emotion detection)
- **Microphone** (for voice input)
- **Internet connection** (for Groq API and Google Speech Recognition)

### 1. Clone & set up a virtual environment

```bash
cd AICompanion
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note (Windows):** If `pyaudio` fails to install, download the appropriate `.whl` file from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install it with `pip install <filename>.whl`.
>
> **Note (Linux):** You may need to install `portaudio` first:
> ```bash
> sudo apt install portaudio19-dev python3-pyaudio
> ```

### 3. Set up Groq API key

1. Sign up for a free account at [https://console.groq.com](https://console.groq.com).
2. Create an API key.
3. Copy the example env file and add your key:

```bash
cp .env.example .env
# Edit .env and paste your API key
```

### 4. Run the application

**Web Mode (Recommended):**
```bash
python app.py
# Open http://localhost:5000 in your browser
# Allow camera and microphone when prompted
```

**CLI Mode:**
```bash
# Full mode — camera + voice
python main.py

# Without camera (no emotion detection)
python main.py --no-camera

# Without voice (keyboard input, text output)
python main.py --no-voice

# Text-only mode (no camera, no voice)
python main.py --text-only
```

### 5. Deploy on Render

The repo includes a Render blueprint file: [render.yaml](e:/my_projects/AICompanion/render.yaml)

1. Push the repository to GitHub.
2. In Render, choose New + > Blueprint and select this repo.
3. Add the environment variable `GROQ_API_KEY` in Render.
4. Deploy.

Render uses this production start command:

```bash
gunicorn --worker-class gthread --threads 8 --timeout 120 --bind 0.0.0.0:$PORT app:app
```

Notes:
- Render installs `render-requirements.txt`, which excludes CLI-only dependencies such as `pyaudio`.
- Hosted deployments should use the browser camera and microphone only. Server-side camera access is not expected on Render.
- The app still supports browser emotion detection, multilingual chat, vision analysis, and proactive AI in production.

---

## 💬 Example Interaction

```
AI Companion: You look a little sad today. Do you want to talk about it?

You (speaking): Yeah, today was really stressful.

AI Companion: I'm sorry you're feeling that way. What happened today?

You (speaking): Work was overwhelming and I couldn't finish anything.

AI Companion: That sounds tough. Remember, it's okay to have days like that.
              Tomorrow is a fresh start. Is there anything specific I can help with?
```

Say **"goodbye"** at any time to end the conversation, or press **Ctrl+C** to quit.

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `Camera not found` | Check that a webcam is connected and not used by another app |
| `No microphone detected` | Connect a microphone and restart |
| `pyaudio install error` | See platform-specific notes above |
| `GROQ_API_KEY not set` | Create a `.env` file with your key (see step 3) |
| `Speech recognition failed` | Check internet connection (Google API requires it) |
| `pyttsx3 error on Linux` | Install espeak: `sudo apt install espeak` |

---

## 📄 License

This project uses only free and open-source tools. Use it freely for personal and educational purposes.
