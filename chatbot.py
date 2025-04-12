import configparser
import logging
import os
import csv
import hashlib
from os.path import exists
from datetime import datetime

from data_handler import DataHandler
from intent_classifier import IntentClassifier
from retrieval_engine import RetrievalEngine
from disambiguation import Disambiguator
from response_generator import ResponseGenerator
from feedback_logger import FeedbackLogger

# Setup Logging
def setup_logging(log_path, log_level):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def _get_vector_data_signature(df):
    relevant_text = df[['Symptoms', 'Keywords/Triggers']].astype(str).values.flatten()
    combined = ''.join(sorted(relevant_text))
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

class Chatbot:
    def __init__(self, config_path="config.ini", test_mode=False):
        self.config = self.load_config(config_path)
        setup_logging(self.config['log_file'], self.config['log_level'])
        logging.info("Initializing Offline Emergency Chatbot.")

        self.data_handler = DataHandler(self.config)
        self.intent_classifier = IntentClassifier(self.config)
        self.retrieval_engine = RetrievalEngine(self.config)
        self.disambiguator = Disambiguator(self.config)
        self.response_generator = ResponseGenerator()
        self.feedback_logger = FeedbackLogger(self.config['feedback_path'])

        self.df, self.intent_df = self.data_handler.load_data()
        self.setup_or_load_models()
        self.test_mode = test_mode
        self.diagnostics_log = self.config.get("log_diagnostics_path", "logs/diagnostics_log.csv")

        if not os.path.exists(self.diagnostics_log):
            with open(self.diagnostics_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_input", "intent", "confidence", "spread_score", "top_score", "reason"])

    def load_config(self, path):
        config = configparser.ConfigParser()
        config.read(path)
        return config['DEFAULT']

    def setup_or_load_models(self):
        logging.info("Checking intent classifier signature...")
        self.intent_classifier.setup_or_load(self.intent_df)

        signature_path = "vectorizers/vector_signature.txt"
        current_sig = _get_vector_data_signature(self.df)
        prev_sig = open(signature_path).read().strip() if exists(signature_path) else None

        tfidf_path = 'vectorizers/tfidf_vectorizer.pkl'
        emb_paths = [
            'embeddings/embeddings_l12.pkl',
            'embeddings/embeddings_l6.pkl',
            'embeddings/embeddings_bert.pkl'
        ]

        combined = self.df['Symptoms'] + ' ' + self.df['Keywords/Triggers']

        if prev_sig == current_sig and exists(tfidf_path) and all([exists(p) for p in emb_paths]):
            logging.info("✅ Loading cached vector models and embeddings.")
            import pickle
            with open(tfidf_path, 'rb') as f:
                self.retrieval_engine.tfidf_vectorizer = pickle.load(f)
            processed = combined.apply(DataHandler.preprocess_text)
            self.retrieval_engine.tfidf_matrix = self.retrieval_engine.tfidf_vectorizer.transform(processed)

            self.df['embedding_l12'] = list(self.retrieval_engine.model_l12.encode(combined, convert_to_numpy=True))
            self.df['embedding_l6'] = list(self.retrieval_engine.model_l6.encode(combined, convert_to_numpy=True))
            self.df['embedding_bert'] = list(self.retrieval_engine.model_bert.encode(combined, convert_to_numpy=True))
        else:
            logging.info("🔁 Recomputing TF-IDF and semantic embeddings.")
            self.retrieval_engine.initialize(self.df)
            import pickle
            os.makedirs("embeddings", exist_ok=True)
            os.makedirs("vectorizers", exist_ok=True)
            with open(tfidf_path, 'wb') as f:
                pickle.dump(self.retrieval_engine.tfidf_vectorizer, f)
            with open(emb_paths[0], 'wb') as f:
                pickle.dump(list(self.df['embedding_l12']), f)
            with open(emb_paths[1], 'wb') as f:
                pickle.dump(list(self.df['embedding_l6']), f)
            with open(emb_paths[2], 'wb') as f:
                pickle.dump(list(self.df['embedding_bert']), f)
            with open(signature_path, "w") as f:
                f.write(current_sig)

    def log_decision(self, user_input, intent, confidence, spread_score, top_score, reason):
        with open(self.diagnostics_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), user_input, intent, f"{confidence:.4f}", f"{spread_score:.4f}", f"{top_score:.4f}", reason])

    def run_single_query(self, user_input: str) -> dict:
        if user_input.startswith("disambiguation:") or user_input.startswith("correction:"):
            label = user_input.split(":", 1)[1].strip()
            match = self.df[self.df["Emergency Type"] == label]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "type": "normal",
                    "response": self.response_generator.render(row),
                    "feedback": True
                }
            else:
                return {
                    "type": "fallback",
                    "response": "⚠️ Sorry, I couldn’t find that emergency in my database. Please try describing it again.",
                    "feedback": True
                }

        result = self.intent_classifier.predict_intent(user_input)
        intent = result.intent
        confidence = result.confidence
        is_gibberish = result.is_gibberish
        reason = result.reason

        if intent == "distress":
            return {
                "type": "distress",
                "response": {
                    "text": (
                        "🧡 It sounds like you’re in serious emotional distress.\n"
                        "I'm just an offline assistant and can't provide direct help, but you are not alone.\n"
                        "Please speak with someone you trust or contact a local crisis support service.\n"
                        "If you're in danger, seek emergency help immediately."
                    )
                },
                "feedback": False
            }

        if is_gibberish or intent == "non_emergency":
            return {
                "type": "info",
                "response": {
                    "text": "ℹ️ Please describe a real emergency or symptom."
                },
                "feedback": False
            }

        if reason == "too_long":
            return {
                "type": "info",
                "response": {
                    "text": "⚠️ I wasn’t able to understand — could you describe your emergency in one or two short sentences?"
                },
                "feedback": False
            }

        feedback_match = self.feedback_logger.find_feedback_match(user_input)
        if feedback_match == "NONE":
            return {
                "type": "fallback",
                "response": {
                    "text": "⚠️ This seems serious, but I don't have relevant information in my database. Please contact emergency services."
                },
                "feedback": True
            }
        elif feedback_match:
            row = self.df[self.df["Emergency Type"] == feedback_match].iloc[0]
            return {
                "type": "normal",
                "response": self.response_generator.render(row),
                "feedback": True
            }

        matches, spread_score = self.retrieval_engine.retrieve(user_input, self.df)
        top_score = matches.iloc[0]["Score"]

        if top_score < float(self.config.get('fallback_threshold', 0.02)):
            return {
                "type": "fallback",
                "response": {
                    "text": "⚠️ This seems serious, but I don't have relevant information in my database. Please contact emergency services."
                },
                "feedback": True
            }

        if reason == "hybrid_agreement" or (
            top_score > 0.7 and top_score - matches.iloc[1]["Score"] > self.disambiguator.threshold
        ):
            selected = matches.iloc[0]
            return {
                "type": "normal",
                "response": self.response_generator.render(selected),
                "feedback": True
            }

        selected = self.disambiguator.resolve(matches, user_input, auto=True)
        if isinstance(selected, dict) and selected.get("disambiguation"):
            return {
                "type": "disambiguation",
                "response": selected["message"],
                "options": selected["options"],
                "feedback": True
            }

        return {
            "type": "normal",
            "response": self.response_generator.render(selected),
            "feedback": True
        }
