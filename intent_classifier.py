import pickle
import numpy as np
import re
from data_handler import DataHandler
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import os
import hashlib

class IntentResult:
    def __init__(self, intent, confidence, is_gibberish, reason=""):
        self.intent = intent
        self.confidence = confidence
        self.is_gibberish = is_gibberish
        self.reason = reason

class IntentClassifier:
    def __init__(self, config):
        self.config = config
        self.threshold = float(config.get('intent_confidence_threshold', 0.02))
        self.semantic_threshold = float(config.get('semantic_confidence_threshold', 0.65))
        self.pipeline = None

        model_path = config.get('sentence_transformer_model_l6', 'models/MiniLM-L6-v2')
        self.semantic_model = SentenceTransformer(model_path)
        self.semantic_centroids = {}

    def is_gibberish(self, user_input):
        text = user_input.strip().lower()

        if len(text) > 300 or len(text) < 4:
            return True
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.4:
            return True
        if re.fullmatch(r"[^a-zA-Z0-9]+", text):
            return True
        if len(set(text.split())) == 1 and len(text.split()) > 3:
            return True
        return False

    def _get_data_signature(self, df):
        # Generate hash based on sorted intent data
        data_string = ''.join(df.sort_values(by=['Phrase', 'Intent']).astype(str).values.flatten())
        return hashlib.md5(data_string.encode('utf-8')).hexdigest()

    def setup_or_load(self, df,
                      model_path="intent_classifier/intent_model.pkl",
                      centroid_path="intent_classifier/centroids.pkl",
                      sig_path="intent_classifier/data_signature.txt"):
        current_sig = self._get_data_signature(df)

        prev_sig = None
        if os.path.exists(sig_path):
            with open(sig_path, "r") as f:
                prev_sig = f.read().strip()

        if prev_sig == current_sig and os.path.exists(model_path):
            self.load_model(model_path)
            self.load_centroids(centroid_path)
            print("✅ Loaded cached intent classifier model.")
        else:
            print("🔁 Training new intent classifier — dataset has changed.")
            self.train(df)
            self.save_model(model_path)
            self.save_centroids(centroid_path)
            with open(sig_path, "w") as f:
                f.write(current_sig)

    def train(self, df):
        processed_texts = df['Phrase'].apply(DataHandler.preprocess_text)
        labels = df['Intent']

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        self.pipeline.fit(processed_texts, labels)

        # Compute semantic centroids
        self.semantic_centroids = {}
        for label in labels.unique():
            samples = df[df['Intent'] == label]['Phrase'].apply(DataHandler.preprocess_text)
            embeddings = self.semantic_model.encode(samples.tolist())
            centroid = np.mean(embeddings, axis=0)
            self.semantic_centroids[label] = centroid

    def predict_intent(self, user_input):
        if len(user_input.strip()) > 300:
            return IntentResult("non_emergency", 0.0, True, reason="too_long")
        if self.is_gibberish(user_input):
            return IntentResult("non_emergency", 0.0, True, reason="gibberish")

        processed_input = DataHandler.preprocess_text(user_input)
        probs = self.pipeline.predict_proba([processed_input])[0]
        tfidf_confidence = max(probs)
        tfidf_pred = self.pipeline.classes_[np.argmax(probs)]

        # Semantic prediction
        emb_input = self.semantic_model.encode([processed_input])[0]
        similarities = {
            label: cosine_similarity([emb_input], [centroid])[0][0]
            for label, centroid in self.semantic_centroids.items()
        }
        semantic_pred = max(similarities, key=similarities.get)
        semantic_confidence = similarities[semantic_pred]

        # Hybrid fusion logic
                # Hybrid fusion logic
        if tfidf_pred == semantic_pred:
            return IntentResult(tfidf_pred, tfidf_confidence, False, reason="hybrid_agreement")
        elif tfidf_confidence > 0.6 and semantic_confidence < self.semantic_threshold:
            return IntentResult(tfidf_pred, tfidf_confidence, False, reason="tfidf_confidence_high")
        elif semantic_confidence > self.semantic_threshold:
            return IntentResult(semantic_pred, semantic_confidence, False, reason="semantic_override")

        # Soft vague detection from config
        vague_tfidf_max = float(self.config.get("vague_tfidf_max", 0.4))
        vague_semantic_max = float(self.config.get("vague_semantic_max", 0.5))

        if tfidf_confidence < vague_tfidf_max and semantic_confidence < vague_semantic_max:
            return IntentResult("vague_emergency", semantic_confidence, False, reason="soft_vague_low_confidence")

        # Default vague fallback
        return IntentResult("vague_emergency", semantic_confidence, False, reason="disagreement_vague")


    def save_model(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.pipeline, file)

    def load_model(self, path):
        with open(path, 'rb') as file:
            self.pipeline = pickle.load(file)

    def save_centroids(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.semantic_centroids, f)

    def load_centroids(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.semantic_centroids = pickle.load(f)
        else:
            print(f"⚠️ Centroid file not found at {path}. Prediction may fallback.")
            self.semantic_centroids = {}
