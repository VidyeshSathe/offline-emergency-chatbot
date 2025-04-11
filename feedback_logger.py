import sqlite3
import datetime
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FeedbackLogger:
    def __init__(self, db_path):
        self.db_path = db_path
        self.semantic_model = SentenceTransformer("models/MiniLM-L6-v2")
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                predicted_emergency TEXT,
                confirmed_emergency TEXT,
                was_helpful INTEGER,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_feedback(self, user_input, predicted_emergency, was_helpful, confirmed_emergency=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO user_feedback (user_input, predicted_emergency, confirmed_emergency, was_helpful, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (user_input.strip(), predicted_emergency.strip(), 
             confirmed_emergency.strip() if confirmed_emergency and confirmed_emergency != "None of the above" else None, 
             int(was_helpful), datetime.datetime.now())
        )
        conn.commit()
        conn.close()

    def find_feedback_match(self, user_input, threshold=0.85):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_input, predicted_emergency, confirmed_emergency, was_helpful
            FROM user_feedback
            WHERE confirmed_emergency IS NOT NULL OR was_helpful = 1
        ''')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        inputs = [r[0] for r in rows]
        embeddings = self.semantic_model.encode(inputs)
        new_embedding = self.semantic_model.encode([user_input])[0]

        scores = cosine_similarity([new_embedding], embeddings)[0]
        max_idx = scores.argmax()
        max_score = scores[max_idx]

        if max_score >= threshold:
            _, pred, confirmed, helpful = rows[max_idx]
            if helpful == 1:
                return pred
            elif confirmed:
                return confirmed
            else:
                return "NONE"
        return None

    def get_corrections(self, user_input):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT confirmed_emergency FROM user_feedback
            WHERE user_input = ? AND was_helpful = 0 AND confirmed_emergency IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
            ''',
            (user_input.strip(),)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
