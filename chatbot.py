import configparser
import logging
import os
import csv
import hashlib
from os.path import exists
from data_handler import DataHandler
from intent_classifier import IntentClassifier
from retrieval_engine import RetrievalEngine
from disambiguation import Disambiguator
from response_generator import ResponseGenerator
from feedback_logger import FeedbackLogger
from datetime import datetime

# Setup Logging
def setup_logging(log_path, log_level):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Hash generator for vector model fingerprint
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
        logging.info("Feedback database initialized and verified.")

        self.df, self.intent_df = self.data_handler.load_data()
        self.setup_or_load_models()
        self.test_mode = test_mode

    def load_config(self, path):
        config = configparser.ConfigParser()
        config.read(path)
        return config['DEFAULT']

    def setup_or_load_models(self):
        logging.info("Checking intent data signature and loading or retraining intent classifier...")
        self.intent_classifier.setup_or_load(self.intent_df)

        logging.info("Checking embedding/vector model cache...")
        signature_path = "vectorizers/vector_signature.txt"
        tfidf_path = 'vectorizers/tfidf_vectorizer.pkl'
        emb_l12_path = 'embeddings/embeddings_l12.pkl'
        emb_l6_path = 'embeddings/embeddings_l6.pkl'
        emb_bert_path = 'embeddings/embeddings_bert.pkl'

        current_sig = _get_vector_data_signature(self.df)
        prev_sig = None
        if os.path.exists(signature_path):
            with open(signature_path, "r") as f:
                prev_sig = f.read().strip()

        combined = self.df['Symptoms'] + ' ' + self.df['Keywords/Triggers']

        if (
            prev_sig == current_sig and
            os.path.exists(tfidf_path) and
            os.path.exists(emb_l12_path) and
            os.path.exists(emb_l6_path) and
            os.path.exists(emb_bert_path)
        ):
            logging.info("✅ Loading cached TF-IDF and embedding files.")
            import pickle
            with open(tfidf_path, 'rb') as f:
                self.retrieval_engine.tfidf_vectorizer = pickle.load(f)
            processed = combined.apply(DataHandler.preprocess_text)
            self.retrieval_engine.tfidf_matrix = self.retrieval_engine.tfidf_vectorizer.transform(processed)

            # This part was missing before — REQUIRED
            self.df['embedding_l12'] = list(self.retrieval_engine.model_l12.encode(combined, convert_to_numpy=True))
            self.df['embedding_l6'] = list(self.retrieval_engine.model_l6.encode(combined, convert_to_numpy=True))
            self.df['embedding_bert'] = list(self.retrieval_engine.model_bert.encode(combined, convert_to_numpy=True))

        else:
            logging.info("🔁 Recomputing TF-IDF and embeddings — emergency data changed.")
            self.retrieval_engine.initialize(self.df)
            import pickle
            os.makedirs("embeddings", exist_ok=True)
            os.makedirs("vectorizers", exist_ok=True)
            with open(emb_l12_path, 'wb') as f:
                pickle.dump(list(self.retrieval_engine.model_l12.encode(combined)), f)
            with open(emb_l6_path, 'wb') as f:
                pickle.dump(list(self.retrieval_engine.model_l6.encode(combined)), f)
            with open(emb_bert_path, 'wb') as f:
                pickle.dump(list(self.retrieval_engine.model_bert.encode(combined)), f)
            with open(tfidf_path, 'wb') as f:
                pickle.dump(self.retrieval_engine.tfidf_vectorizer, f)
            with open(signature_path, "w") as f:
                f.write(current_sig)


        self.diagnostics_log = self.config.get("log_diagnostics_path", "logs/diagnostics_log.csv")
        if not os.path.exists(self.diagnostics_log):
            with open(self.diagnostics_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_input", "intent", "confidence", "spread_score", "top_score", "reason"])

    def log_decision(self, user_input, intent, confidence, spread_score, top_score, reason):
        with open(self.diagnostics_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), user_input, intent, f"{confidence:.4f}", f"{spread_score:.4f}", f"{top_score:.4f}", reason])

    def run(self):
        print("🚨 Offline Emergency Assistant Ready — I'm here to help with first aid in urgent situations. Type 'exit' anytime to leave.")
        print("⚠️ Disclaimer: This assistant is not a substitute for professional medical advice, diagnosis, or treatment. It provides offline first aid information based on a predefined dataset. Always consult emergency services in critical situations.")
        logging.info("Chatbot session started.")

        while True:
            user_input = input("\nDescribe your emergency: ").strip()
            if user_input.lower() == "exit":
                logging.info("Chatbot session ended by user.")
                print("👋 Stay safe! Exiting chatbot.")
                break

            feedback_match = self.feedback_logger.find_feedback_match(user_input)
            if feedback_match == "NONE":
                print("⚠️ This might be serious, but I don't have enough information in my database. Please contact your local emergency service immediately.")
                continue
            elif feedback_match:
                row = self.df[self.df["Emergency Type"] == feedback_match].iloc[0]
                self.response_generator.generate(row)
                continue

            result = self.intent_classifier.predict_intent(user_input)
            intent = result.intent
            confidence = result.confidence
            is_gibberish = result.is_gibberish
            reason = result.reason

            if intent == "distress":
                print("🧡 It sounds like you’re in serious emotional distress.")
                print("I'm just an offline assistant, so I can't provide direct help, but you are not alone.")
                print("Please speak with someone you trust or contact a local crisis support service as soon as possible.")
                print("If you are in immediate danger, seek emergency help right away.")
                self.log_decision(user_input, intent, confidence, 0.0, 0.0, reason)
                continue

            if is_gibberish:
                print("ℹ️ I’m trained to assist with real medical emergencies. Please describe what’s happening or a specific symptom.")
                self.log_decision(user_input, intent, confidence, 0.0, 0.0, reason)
                continue

            if intent == "non_emergency":
                print("ℹ️ I’m trained to assist with real medical emergencies. Please describe what’s happening or a specific symptom.")
                self.log_decision(user_input, intent, confidence, 0.0, 0.0, reason)
                continue

            if reason == "too_long":
                print("⚠️ I wasn’t able to understand — could you describe your emergency in one or two short sentences?")
                continue

            matches, spread_score = self.retrieval_engine.retrieve(user_input, self.df)
            top_score = matches.iloc[0]['Score']

            vague_min = float(self.config.get('vague_score_min', 0.005))
            vague_max = float(self.config.get('vague_score_max', 0.025))

            if intent == "vague_emergency":
                if vague_min < spread_score < vague_max:
                    parts, suggestions = self.retrieval_engine.suggest_emergencies_by_body_part(user_input, self.df)
                    if suggestions:
                        print(f"⚠️ It sounds like something is affecting your {parts[0]}. Could you clarify if it might be related to any of these issues: {', '.join(suggestions[:4])}?")
                        reason = "vague_with_body_part"
                    else:
                        print("⚠️ I need a little more detail to help properly. Could you tell me what’s happening or what part of the body is affected?")
                        reason = "vague_general"
                    self.log_decision(user_input, intent, confidence, spread_score, top_score, reason)
                    continue

            if top_score < float(self.config['fallback_threshold']):
                print("⚠️ This might be serious, but I don't have enough information in my database. Please contact your local emergency service immediately.")
                self.log_decision(user_input, intent, confidence, spread_score, top_score, reason or "fallback")
                continue

            # DISAMBIGUATION DECISION
            if reason == "hybrid_agreement" and (
                top_score > 0.7 and (top_score - matches.iloc[1]["Score"] > self.disambiguator.threshold)
            ):
                selected = matches.iloc[0]
                print(f"✅ High confidence (score={top_score:.2f}) — auto-selected top match.")
                self.response_generator.generate(selected)
            else:
                selected = self.disambiguator.resolve(matches, user_input, auto=self.test_mode)
                if selected is None:
                    print("🧭 Got it. Please try describing the emergency in a different way so I can better understand.")
                    self.log_decision(user_input, intent, confidence, spread_score, top_score, "disambiguation_none_selected")
                    continue
                self.response_generator.generate(selected)

            self.log_decision(user_input, intent, confidence, spread_score, top_score, reason)

            if not self.test_mode:
                print("\n💬 Was this the help you were looking for? Your feedback helps me improve.")
                print("1. Yes, this was helpful")
                print("2. No, I was looking for something else")
                feedback_choice = input("Select 1 or 2: ").strip()
                if feedback_choice == "1":
                    if self.config.getboolean('enable_feedback_logging', fallback=True):
                        self.feedback_logger.log_feedback(user_input, selected['Emergency Type'], was_helpful=True)
                elif feedback_choice == "2":
                    print("Sorry! Let me help you refine it.")
                    print("Here are some emergency types you can choose from:")
                    for i, label in enumerate(sorted(self.df['Emergency Type'].unique()), 1):
                        print(f"{i}. {label}")
                    print(f"{len(self.df['Emergency Type'].unique())+1}. None of the above – not listed here")
                    choice = input("Select the correct one (enter number): ").strip()
                    try:
                        idx = int(choice)
                        if idx <= len(self.df['Emergency Type'].unique()):
                            corrected = sorted(self.df['Emergency Type'].unique())[idx-1]
                        else:
                            corrected = "None of the above"
                        if self.config.getboolean('enable_feedback_logging', fallback=True):
                            self.feedback_logger.log_feedback(user_input, selected['Emergency Type'], was_helpful=False, confirmed_emergency=corrected)
                    except:
                        print("Invalid input. Feedback not saved.")

    def run_single_query(self, user_input: str) -> dict:
        result = self.intent_classifier.predict_intent(user_input)
        intent = result.intent
        reason = result.reason
    
        if result.is_gibberish or intent == "non_emergency":
            return {
                "type": "info",
                "response": "ℹ️ Please describe a real emergency or symptom.",
                "feedback": False
            }
    
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
    
        
        emergency_matches = self.df[self.df["Emergency Type"].str.lower() == user_input.lower()]
        if not emergency_matches.empty:
            row = emergency_matches.iloc[0]
            return {
                "type": "normal",
                "response": self.response_generator.render(row),
                "feedback": True
            }
    
        feedback_match = self.feedback_logger.find_feedback_match(user_input)
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
    
        matches, spread_score = self.retrieval_engine.retrieve(user_input, self.df)
        logging.info(f"Top match: {matches.iloc[0]['Emergency Type']} (score={top_score:.4f})")
        top_score = matches.iloc[0]["Score"]
    
        fallback_threshold = float(self.config.get('fallback_threshold', 0.02))
        if top_score < fallback_threshold:
            return {
                "type": "fallback",
                "response": "⚠️ This seems serious, but I don't have relevant information in my database. Please contact emergency services.",
                "feedback": True
            }
    
        # DISAMBIGUATION
        if reason == "hybrid_agreement" or (
            top_score > 0.7 and top_score - matches.iloc[1]["Score"] > self.disambiguator.threshold
        ):
            selected = matches.iloc[0]
            return {
                "type": "normal",
                "response": self.response_generator.render(selected),
                "feedback": True
            }
    
        # DISAMBIGUATION UI PATH
        selected = self.disambiguator.resolve(matches, user_input, auto=True)
        if selected is None:
            options = matches.head(3)['Emergency Type'].tolist()
            options.append("None of the above")
            return {
                "type": "disambiguation",
                "response": "⚠️ Multiple possible emergencies detected. Please choose the best match:",
                "options": options,
                "feedback": True
            }
    
        return {
            "type": "normal",
            "response": self.response_generator.render(selected),
            "feedback": True
        }




if __name__ == "__main__":
    try:
        chatbot = Chatbot()
        chatbot.run()
    except Exception as e:
        logging.critical(f"Critical failure: {e}", exc_info=True)
        print(f"❌ Critical error encountered. Check logs for details: {e}")
