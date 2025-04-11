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
        self.test_mode = test_mode
        self.diagnostics_log = self.config.get("log_diagnostics_path", "logs/diagnostics_log.csv")

        self.setup_or_load_models()

        if not exists(self.diagnostics_log):
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

    def handle_query(self, user_input, feedback_override=None):
        if user_input.startswith("disambiguation:"):
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
        feedback_match = self.feedback_logger.find_feedback_match(user_input)
        if feedback_override:
            corrected = feedback_override.get("corrected")
            predicted = feedback_override.get("predicted")
            helpful = feedback_override.get("was_helpful")
            if helpful:
                self.feedback_logger.log_feedback(user_input, predicted, was_helpful=True)
            else:
                self.feedback_logger.log_feedback(user_input, predicted, was_helpful=False, confirmed_emergency=corrected)
                if corrected != "None of the above":
                    row = self.df[self.df["Emergency Type"] == corrected].iloc[0]
                    return {
                        "type": "normal",
                        "response": self.response_generator.render(row),
                        "feedback": True
                    }
                else:
                    return {
                        "type": "fallback",
                        "response": "⚠️ This seems serious, but it's not in my database. Please contact emergency services.",
                        "feedback": True
                    }
        
            # NEW: return correction options if not provided yet
            if corrected is None:
                options = sorted(self.df["Emergency Type"].unique().tolist())
                options.append("None of the above")
                return {
                    "type": "correction",
                    "response": "❌ Thanks for the feedback. What emergency were you actually referring to?",
                    "options": options,
                    "feedback": True
                }


        if feedback_match == "NONE":
            return {
                "type": "fallback",
                "response": "⚠️ This seems serious, but I don't have relevant information in my database. Please contact emergency services.",
                "feedback": True
            }
        elif feedback_match:
            row = self.df[self.df["Emergency Type"] == feedback_match].iloc[0]
            return {
                "type": "normal",
                "response": self.response_generator.render(row),
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
                "response": (
                    "🧡 It sounds like you’re in serious emotional distress.\n"
                    "I'm just an offline assistant and can't provide direct help, but you are not alone.\n"
                    "Please speak with someone you trust or contact a local crisis support service.\n"
                    "If you're in danger, seek emergency help immediately."
                ),
                "feedback": False
            }

        if is_gibberish or intent == "non_emergency":
            return {
                "type": "info",
                "response": "ℹ️ Please describe a real emergency or symptom.",
                "feedback": False
            }

        if reason == "too_long":
            return {
                "type": "info",
                "response": "⚠️ I wasn’t able to understand — could you describe your emergency in one or two short sentences?",
                "feedback": False
            }

        matches, spread_score = self.retrieval_engine.retrieve(user_input, self.df)
        top_score = matches.iloc[0]['Score']

        if intent == "vague_emergency":
            vague_min = float(self.config.get('vague_score_min', 0.005))
            vague_max = float(self.config.get('vague_score_max', 0.025))
            if vague_min < spread_score < vague_max:
                parts, suggestions = self.retrieval_engine.suggest_emergencies_by_body_part(user_input, self.df)
                if suggestions:
                    return {
                        "type": "clarify",
                        "response": f"⚠️ It sounds like something is affecting your {parts[0]}. Could you clarify if it might be related to: {', '.join(suggestions[:4])}?",
                        "feedback": False
                    }
                return {
                    "type": "clarify",
                    "response": "⚠️ I need more detail. What exactly is happening or which part of the body is affected?",
                    "feedback": False
                }

        if top_score < float(self.config.get('fallback_threshold', 0.02)):
            return {
                "type": "fallback",
                "response": "⚠️ This seems serious, but I don't have relevant information in my database. Please contact emergency services.",
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

        selected = self.disambiguator.resolve(matches, user_input, auto=not self.test_mode)

        # ✅ If disambiguation is triggered, return structured object
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

    def run_single_query(self, user_input: str) -> dict:
        return self.handle_query(user_input)

    def run(self):
        print("🚨 Offline Emergency Assistant Ready — I'm here to help with first aid in urgent situations. Type 'exit' anytime to leave.")
        print("⚠️ Disclaimer: This assistant is not a substitute for professional medical advice. Always consult professionals in critical situations.")
        logging.info("Chatbot session started.")

        while True:
            user_input = input("\nDescribe your emergency: ").strip()
            if user_input.lower() == "exit":
                print("👋 Stay safe! Exiting chatbot.")
                logging.info("Chatbot session ended.")
                break

            result = self.handle_query(user_input)
            print(result["response"])

            if result["type"] == "disambiguation":
                print("\nChoose the correct emergency:")
                for i, opt in enumerate(result["options"], 1):
                    print(f"{i}. {opt}")
                choice = input("Select number: ").strip()
                try:
                    idx = int(choice)
                    selected = result["options"][idx - 1]
                    result = self.handle_query(selected)
                    print(result["response"])
                except:
                    print("❌ Invalid selection.")

            if result["feedback"] and not self.test_mode:
                print("\n💬 Was this helpful?")
                print("1. Yes")
                print("2. No")
                feedback = input("Select 1 or 2: ").strip()
                if feedback == "1":
                    self.handle_query(user_input, {
                        "was_helpful": True,
                        "predicted": self.extract_emergency_name(result["response"]),
                        "corrected": None
                    })
                elif feedback == "2":
                    print("Let’s fix it. Which emergency were you referring to?")
                    types = sorted(self.df['Emergency Type'].unique())
                    for i, label in enumerate(types, 1):
                        print(f"{i}. {label}")
                    print(f"{len(types)+1}. None of the above")
                    correction = input("Select number: ").strip()
                    try:
                        idx = int(correction)
                        if idx <= len(types):
                            corrected = types[idx-1]
                        else:
                            corrected = "None of the above"
                        self.handle_query(user_input, {
                            "was_helpful": False,
                            "predicted": self.extract_emergency_name(result["response"]),
                            "corrected": corrected
                        })
                    except:
                        print("❌ Invalid input. Feedback not saved.")

    def extract_emergency_name(self, response_text):
        import re
        match = re.search(r'🚨 Emergency Detected: (.+)', response_text)
        return match.group(1).strip() if match else "Unknown"


if __name__ == "__main__":
    try:
        chatbot = Chatbot()
        chatbot.run()
    except Exception as e:
        logging.critical(f"Critical failure: {e}", exc_info=True)
        print(f"❌ Critical error encountered. Check logs for details: {e}")
