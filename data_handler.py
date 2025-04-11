import sqlite3
import pandas as pd
import re
import nltk
import os

# Explicitly point to your offline nltk_data directory
nltk.data.path.append(os.path.abspath("./nltk_data"))

# Verify resources load correctly
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class DataHandler:
    def __init__(self, config):
        self.db_path = config['database_path']

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM first_aid_data", conn)
        intent_df = pd.read_sql("SELECT * FROM intent_data", conn)
        conn.close()
        return df.fillna(""), intent_df.fillna("")

    @staticmethod
    def preprocess_text(text):
        lemmatizer = WordNetLemmatizer()
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        from nltk.tokenize import TreebankWordTokenizer
        tokens = TreebankWordTokenizer().tokenize(text)


        if not tokens:
            return ""  # Handle empty result after filtering

        tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum()]
        return " ".join(tokens)
