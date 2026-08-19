import os
import re
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

class TurkishBM25:
    """Türkçe uyumlu, hafif ve yerel bir BM25 sınıfı."""
    def __init__(self, corpus, b=0.75, k1=1.5):
        self.b = b
        self.k1 = k1
        self.corpus_size = len(corpus)
        self.avg_dl = 0
        self.doc_lengths = []
        self.doc_freqs = {}
        self.idf = {}
        self.tokenized_corpus = []
        
        self._initialize(corpus)

    def _tokenize(self, text):
        text = text.replace('I', 'ı').replace('İ', 'i').lower()
        words = re.findall(r'[a-zA-Z0-9çğıöşüçĞIİÖŞÜ]+', text)
        return words

    def _initialize(self, corpus):
        total_length = 0
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.tokenized_corpus.append(tokens)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                
        self.avg_dl = total_length / self.corpus_size if self.corpus_size > 0 else 1
        
        for token, df in self.doc_freqs.items():
            self.idf[token] = max(0.0001, (self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query):
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            
            word_counts = {}
            for token in doc_tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
                
            for q_token in query_tokens:
                if q_token in word_counts:
                    tf = word_counts[q_token]
                    idf_val = self.idf.get(q_token, 0.0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_dl))
                    score += idf_val * (numerator / denominator)
            scores.append(score)
        return scores

@st.cache_resource
def load_rag_assets():
    """Vektör tabanını ve E5 modelini önbellekte bir kez yükler."""
    try:
        db_path = "./gazali_chroma_db"
        chroma_client = chromadb.PersistentClient(path=db_path)
        embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
        collection = chroma_client.get_collection(name="gazali_kulliyati")
        
        all_data = collection.get()
        if not all_data or not all_data["documents"]:
            return None, None, None, None
            
        bm25_engine = TurkishBM25(all_data["documents"])
        return embed_model, collection, bm25_engine, all_data
    except Exception as e:
        st.error(f"Veritabanı yüklenirken hata oluştu: {e}")
        return None, None, None, None
