# **Group 16**

# Adham Mohmed Elsaied Elwakel 222100195,
# Samaa Khaled Eltaky 222100761,
# Habiba Ahmed Abdelnapy 222100471, 
# Youssef Hussieny 222101943

# collaborative.py

# content_based.py
import pandas as pd
import numpy as np
import re
from collections import Counter

# --- Helper Class: TF-IDF From Scratch ---
class TFIDF_Scratch:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocab = {}  # word -> index
        self.idf = None
        # Basic English stop words list (since we aren't using sklearn's list)
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
            'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 
            'doesn', "doesn't"
        ])

    def _tokenize(self, text):
        # Lowercase, remove non-alphanumeric, split by whitespace
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = text.split()
        return [t for t in tokens if t not in self.stop_words]

    def fit_transform(self, raw_documents):
        # 1. Tokenize all documents
        docs_tokens = [self._tokenize(doc) for doc in raw_documents]
        n_docs = len(docs_tokens)

        # 2. Build Vocabulary (Top K frequent words)
        all_words = [word for doc in docs_tokens for word in doc]
        word_counts = Counter(all_words)
        # Select top max_features words
        most_common = word_counts.most_common(self.max_features)
        
        self.vocab = {word: i for i, (word, count) in enumerate(most_common)}
        n_vocab = len(self.vocab)
        
        # 3. Compute TF (Term Frequency) Matrix
        # Using NumPy for speed (Dense matrix)
        tf_matrix = np.zeros((n_docs, n_vocab))
        
        for doc_idx, tokens in enumerate(docs_tokens):
            doc_len = len(tokens)
            if doc_len == 0: continue
            
            for token in tokens:
                if token in self.vocab:
                    word_idx = self.vocab[token]
                    tf_matrix[doc_idx, word_idx] += 1
            
            # Normalize TF by document length (Standard TF definition)
            # Alternatively: Just raw count. Here we use raw count similar to default sklearn TfidfVectorizer
            # But sklearn does raw counts then L2 normalizes the result.
            pass 

        # 4. Compute IDF (Inverse Document Frequency)
        # DF: Number of documents containing the term
        # (tf_matrix > 0) creates a boolean mask, sum cols to get document freq
        df_counts = np.sum(tf_matrix > 0, axis=0)
        
        # IDF Formula: log((N + 1) / (DF + 1)) + 1  (Smoothed IDF)
        self.idf = np.log((n_docs + 1) / (df_counts + 1)) + 1
        
        # 5. Compute TF-IDF
        tfidf_matrix = tf_matrix * self.idf
        
        # 6. L2 Normalization (Euclidean Norm)
        # Important for Cosine Similarity to work as a simple Dot Product later
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        tfidf_normalized = tfidf_matrix / norms
        
        return tfidf_normalized

# --- Helper Function: Cosine Similarity From Scratch ---
def compute_cosine_similarity_scratch(matrix):
    """
    Computes Cosine Similarity manually: Dot Product of L2-normalized vectors.
    Matrix shape: (N_items, N_features)
    Result shape: (N_items, N_items)
    """
    print("[Content-Based] Computing Cosine Similarity (Dot Product)...")
    # Since vectors are already L2 normalized in our TFIDF class,
    # Cosine Similarity(A, B) = A . B
    similarity = np.dot(matrix, matrix.T)
    return similarity

def build_content_model(df):
    """
    Generates Content-Based Similarity Matrix.
    """
    print("[Content-Based] Building TF-IDF Features (From Scratch)...")
    
    # Create Item Profiles (unique items)
    item_df = df.drop_duplicates(subset='item_id').set_index('item_id')
    
    # Combine relevant text features
    # Ensure strings to avoid tokenization errors
    text_features = item_df['category'].fillna('') + " " + item_df.get('rented for', '').fillna('')
    text_features = text_features.astype(str).tolist()
    
    # TF-IDF Vectorization (Manual)
    tfidf_scratch = TFIDF_Scratch(max_features=1000)
    tfidf_matrix = tfidf_scratch.fit_transform(text_features)
    
    print(f"[Content-Based] TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    
    # Compute Cosine Similarity (Manual)
    cosine_sim = compute_cosine_similarity_scratch(tfidf_matrix)
    
    # Convert to DataFrame
    sim_df = pd.DataFrame(
        cosine_sim, 
        index=item_df.index.astype(str), 
        columns=item_df.index.astype(str)
    )
    
    return sim_df

if __name__ == "__main__":
    # Load Data
    try:
        df = pd.read_csv('../data/preprocessed_data.csv')
    except FileNotFoundError:
        print("Error: '../data/preprocessed_data.csv' not found.")
        exit()
        
    df['item_id'] = df['item_id'].astype(str)
    
    # Build Model
    sim_df = build_content_model(df)
    
    # Save
    sim_df.to_csv('../results/item_similarity.csv')
    print("[Content-Based] Saved to ../results/item_similarity.csv")