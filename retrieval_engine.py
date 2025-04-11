from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from data_handler import DataHandler

class RetrievalEngine:
    def __init__(self, config):
        self.config = config
        self.model_l12 = SentenceTransformer(config['sentence_transformer_model_l12'])
        self.model_l6 = SentenceTransformer(config['sentence_transformer_model_l6'])
        self.model_bert = SentenceTransformer(config['sentence_transformer_model_bert'])
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def initialize(self, df):
        combined = df['Symptoms'] + ' ' + df['Keywords/Triggers']
        processed = combined.apply(DataHandler.preprocess_text)
        self.tfidf_vectorizer = TfidfVectorizer().fit(processed)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(processed)

    def retrieve(self, user_input, df):
        input_clean = DataHandler.preprocess_text(user_input)

        # STEP 1: Enrich query using database triggers and symptoms
        enrichment_terms = []
        for _, row in df.iterrows():
            trigger_words = row["Keywords/Triggers"].lower().split(",")
            if any(term.strip() in input_clean for term in trigger_words):
                enrichment_terms.extend(trigger_words)

        if enrichment_terms:
            enriched_input = input_clean + " " + " ".join(set(enrichment_terms))
        else:
            enriched_input = input_clean

        # STEP 2: Compute retrieval scores using enriched input
        tfidf_input = self.tfidf_vectorizer.transform([enriched_input])
        scores_tfidf = cosine_similarity(tfidf_input, self.tfidf_matrix)[0]

        emb_input_l12 = self.model_l12.encode(enriched_input)
        emb_input_l6 = self.model_l6.encode(enriched_input)
        emb_input_bert = self.model_bert.encode(enriched_input)

        scores_l12 = cosine_similarity([emb_input_l12], list(df['embedding_l12']))[0]
        scores_l6 = cosine_similarity([emb_input_l6], list(df['embedding_l6']))[0]
        scores_bert = cosine_similarity([emb_input_bert], list(df['embedding_bert']))[0]

        final_scores = (
            float(self.config['tfidf_weight']) * scores_tfidf +
            float(self.config['sem_l12_weight']) * scores_l12 +
            float(self.config['sem_l6_weight']) * scores_l6 +
            float(self.config['bert_weight']) * scores_bert
        )

        df['Score'] = final_scores

        # Use top-N DB embeddings for spread calculation
        top_l12_embeddings = list(df['embedding_l12'][:10])
        top_bert_embeddings = list(df['embedding_bert'][:10])

        # Cosine similarity with top entries    
        dist_l12 = [cosine_similarity([emb_input_l12], [e])[0][0] for e in top_l12_embeddings]
        dist_bert = [cosine_similarity([emb_input_bert], [e])[0][0] for e in top_bert_embeddings]

        # Compute spread (standard deviation)
        spread_l12 = np.std(dist_l12)
        spread_bert = np.std(dist_bert)

        # Final spread score (average or max — your choice)
        spread_score = (spread_l12 + spread_bert) / 2
        # Or use: spread_score = max(spread_l12, spread_bert)

        return df.sort_values(by='Score', ascending=False).reset_index(drop=True), spread_score
        
    def suggest_emergencies_by_body_part(self, user_input, df):
        input_clean = user_input.lower()
        matched_parts = []

        # Ensure 'Body Part' column exists and is not null
        if 'Body Part' not in df.columns:
            return [], []

        for body_part in df['Body Part'].dropna().unique():
            if body_part.lower() in input_clean:
                matched_parts.append(body_part.lower())

        if matched_parts:
            suggestions_df = df[df['Body Part'].str.lower().isin(matched_parts)]
            emergency_types = suggestions_df['Emergency Type'].unique().tolist()
            return matched_parts, emergency_types

        return [], []


