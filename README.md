# 🆘 Offline Emergency Response Chatbot

An AI-powered emergency response assistant that works fully offline. This chatbot is capable of understanding user queries related to emergency first aid situations, classifying intent, disambiguating context, and returning relevant responses—all without requiring an internet connection once set up.

---

## 🚀 Features

- FastAPI-powered web API
- Offline semantic search using MiniLM embeddings
- Hybrid intent classification (TF-IDF + Sentence Transformers)
- Context disambiguation and feedback logging
- Fully local SQLite backend
- Render-compatible deployment

---

## 📁 Repository Structure

```
offline-emergency-chatbot/
├── chatbot.py              # Core chatbot logic
├── main_api.py             # FastAPI app entry point
├── config.ini              # Configuration file
├── data/                   # Contains SQLite DB and preprocessed intent data
├── models/                 # Folder for pre-trained embedding models (downloaded later)
├── requirements.txt        # Python package dependencies
├── download_models.py      # Script to download and set up required models
```

---

## ⚙️ Step-by-Step Offline Setup Instructions (Local Use)

You can use either Git or download the repo manually:

### 🧰 Option 1: Clone Using Git

Make sure Git is installed on your system. You can get it from:  
[https://git-scm.com/downloads](https://git-scm.com/downloads)

Then:

```bash
git clone https://github.com/VidyeshSathe/offline-emergency-chatbot.git
cd offline-emergency-chatbot
```

### 📦 Option 2: Download ZIP and Extract

If you don't want to use Git:

1. Go to [https://github.com/VidyeshSathe/offline-emergency-chatbot](https://github.com/VidyeshSathe/offline-emergency-chatbot)
2. Click the green **"Code"** button → **"Download ZIP"**
3. Extract the ZIP on your system
4. Open a terminal in the extracted folder

---

### 🔧 Environment Setup

#### 1. Create and Activate a Python Environment (Recommended)

Using conda:

```bash
conda create -n chatbot-ai python=3.10
conda activate chatbot-ai
```

Or using venv:

```bash
python -m venv chatbot-env
./chatbot-env/Scripts/activate  # Windows
source chatbot-env/bin/activate  # Mac/Linux
```

#### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

#### 3. Download the Model Files

Model files are hosted on Dropbox and excluded from the repo to reduce size. Run the helper script:

```bash
python download_models.py
```

This will place the model in:
```
./models/MiniLM-L6-v2
```

#### 4. Start the API

```bash
uvicorn main_api:app --reload
```

Then go to:
```
http://127.0.0.1:8000/docs
```
to interact with the API via Swagger UI.

---

## 🌐 Optional: Use Render Deployment Instead

### 🔗 Live Demo (Deployed on Render)
You can test the deployed chatbot live here:
[https://firstaidchatbot-e1478.web.app/](https://firstaidchatbot-e1478.web.app/)


This project is already configured for deployment using [Render](https://render.com), a free cloud hosting platform.

1. Fork or import this repo into your GitHub
2. Create a new **Web Service** on Render
3. Use the existing `render.yaml` to:
   - Auto-install dependencies
   - Auto-download model files via Dropbox
   - Launch the app with `uvicorn`

You’ll get a hosted URL to use/share the chatbot without needing to run it locally.

---

## 🧠 Notes

- This chatbot is designed to run fully offline once dependencies and model files are set up
- The same model download logic used on Render is replicated via `download_models.py`

---

## 📫 Contact

Developed by **Vidyesh Sathe**  
If you'd like a walkthrough or have questions, feel free to reach out!
