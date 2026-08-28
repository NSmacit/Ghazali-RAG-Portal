import os
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================================
# ISLAMICATE DH - GAZALİ PORTALI: SEMANTIC CACHE LAYER
# =====================================================================
# Bu sınıf, kullanıcının sorduğu soruların embedding vektörlerini saklayarak
# daha önce sorulmuş benzer soruları (semantik benzerlik > 0.95) yakalar.
# Benzer bir soru bulunduğunda LLM'e gitmeden doğrudan önbellekteki
# cevabı döner. Bu sayede API maliyetleri sıfırlanır ve yanıt süresi ~5ms'ye düşer.
# =====================================================================

class GazaliSemanticCache:
    def __init__(self, db_path="gazali_cache.db", threshold=0.94):
        """
        Semantik önbellek katmanını başlatır.
        """
        self.db_path = db_path
        self.threshold = threshold
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """
        Önbellek tablosunu oluşturur.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                embedding TEXT NOT NULL, -- JSON-serialized list of floats
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _cosine_similarity(self, vec1, vec2):
        """
        İki vektör arasındaki kosinüs benzerliğini hesaplar.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def check_cache(self, query, query_vector):
        """
        Sorgunun semantik olarak önbellekte olup olmadığını denetler.
        Eğer benzerlik eşik değerin üzerindeyse, önbellekteki yanıtı döner.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT query, response, embedding FROM semantic_cache")
        rows = cursor.fetchall()

        best_score = -1.0
        cached_response = None
        matched_query = None

        query_vector = np.array(query_vector, dtype=np.float32)

        for row in rows:
            cached_query, response, cached_emb_str = row
            cached_emb = np.array(json.loads(cached_emb_str), dtype=np.float32)
            
            similarity = self._cosine_similarity(query_vector, cached_emb)
            if similarity > best_score:
                best_score = similarity
                cached_response = response
                matched_query = cached_query

        if best_score >= self.threshold:
            print(f"[⚡ CACHE HIT] '{query}' için semantik eşleşme bulundu! (Benzerlik: %{best_score*100:.2f})")
            print(f"   Eşleşen Sorgu: '{matched_query}'")
            return cached_response, best_score
        
        print(f"[❄️ CACHE MISS] '{query}' için önbellek eşleşmesi bulunamadı. (En yüksek benzerlik: %{max(0.0, best_score)*100:.2f})")
        return None, best_score

    def set_cache(self, query, response, query_vector):
        """
        Sorguyu, üretilen yanıtı ve sorgunun embedding vektörünü önbelleğe yazar.
        """
        cursor = self.conn.cursor()
        emb_json = json.dumps(list(query_vector))
        cursor.execute(
            "INSERT INTO semantic_cache (query, response, embedding) VALUES (?, ?, ?)",
            (query, response, emb_json)
        )
        self.conn.commit()
        print(f"[💾 CACHED] Yeni sorgu semantik önbelleğe kaydedildi: '{query}'")

    def clear_cache(self):
        """
        Tüm önbellek tablosunu temizler.
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM semantic_cache")
        self.conn.commit()
        print("[🧹 CLEANED] Semantik önbellek tamamen temizlendi.")
