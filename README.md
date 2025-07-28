# 🚨 AI-Powered Offline Emergency Chatbot

A mission-critical AI assistant that provides **first aid and emergency response guidance** — **entirely offline**. Designed for disaster scenarios with no internet access, the chatbot uses a hybrid AI retrieval engine and local datasets to assist users in urgent medical situations.

---

## 🧠 What It Does

- Understands emergency-related natural language input
- Detects intent, filters out gibberish and non-emergency queries
- Retrieves matching emergency cases using TF-IDF + SentenceTransformer models
- Displays structured first aid steps, emergency warnings, and follow-up advice
- Supports disambiguation and feedback correction loop

---

## 💻 How It Works

The system uses a **modular architecture** with key components:

| Module | Purpose |
|--------|---------|
| `intent_classifier.py` | Lightweight NLP-based intent classification (LogReg + MiniLM) |
| `retrieval_engine.py` | Hybrid TF-IDF + MiniLM-L12/L6 + BERT embedding similarity |
| `chatbot.py` | Orchestrates everything (CLI + API logic) |
| `disambiguation.py` | Top-3 disambiguation if input is ambiguous |
| `feedback_logger.py` | Stores user feedback and learns from confirmed corrections |
| `response_generator.py` | Formats structured emergency responses |
| `main_api.py` | FastAPI-based REST API for integration with GUI/Flutter apps |

---

## 📦 Project Structure

```
.
├── chatbot.py
├── config.ini
├── data_handler.py
├── disambiguation.py
├── feedback_logger.py
├── intent_classifier.py
├── main_api.py
├── response_generator.py
├── retrieval_engine.py
├── requirements.txt
├── data/
│   ├── first_aid.db
│   ├── user_feedback.db
├── models/
│   ├── MiniLM-L12-v2/
│   ├── MiniLM-L6-v2/
│   ├── bert-base-uncased/
├── embeddings/
├── vectorizers/
├── intent_classifier/
├── nltk_data/
└── logs/
```

---

## 🔌 Fully Offline Design

| Component | Offline Capability |
|----------|--------------------|
| SQLite database | ✅ Preloaded |
| Embedding models | ✅ Stored in `models/` |
| NLTK resources | ✅ Loaded from `nltk_data/` |
| SentenceTransformers | ✅ Uses local paths |
| Pip dependencies | ✅ Supports `.whl` based installs |

No internet connection is required once dependencies are installed locally.

---

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare NLTK (offline)**:
   - Place `punkt`, `stopwords`, `wordnet` in `./nltk_data`
   - Add: `nltk.data.path.append('./nltk_data')` is already done in code

3. **Run the Chatbot (CLI)**:
   ```bash
   python chatbot.py
   ```

4. **Start the API (Optional)**:
   ```bash
   uvicorn main_api:app --host 0.0.0.0 --port 8000
   ```

---

## 🧪 Testing

- Intent classification and retrieval logic are tested on a manual suite
- Edge cases like vague, gibberish, or out-of-database inputs are handled
- Disambiguation fallback ensures user safety

---

## 📜 Disclaimer

> This tool is not a substitute for professional medical help.  
> It provides **offline first aid information** based on predefined data.  
> In critical situations, **always seek real emergency services**.

---

## 🧭 Future Plans

- 📱 Voice + Multilingual input
- 🌐 Satellite sync for updates
- 🧍‍♂️ Human-in-the-loop triage review

---

## 🔐 License

This project is intended for educational and humanitarian use.  
Please reach out before commercial use or external redistribution.

---
