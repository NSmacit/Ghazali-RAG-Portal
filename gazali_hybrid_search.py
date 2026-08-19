import os
import sys
import math
import re
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =====================================================================
# ISLAMICATE DH - GAZALİ HİBRİT ARAMA VE YAPAY ZEKÂ SORGULAMA MOTORU
# =====================================================================
# Bu modül, İslami Dijital Beşerî Bilimler (Digital Humanities) alanında
# büyük külliyatları (örneğin el-Munkız'ın tamamını) taramak için en ileri
# arama mimarisi olan "Hibrit Arama" (Hybrid Search) altyapısını kurar.
#
# MİMARİ BİLEŞENLER:
# 1. Klasik Anahtar Kelime Araması (BM25 - Sparse Retrieval)
# 2. Anlamsal Vektör Araması (Dense Retrieval - Multilingual E5-Large)
# 3. Sıralama Sentezi (Reciprocal Rank Fusion - RRF Algoritması)
# 4. Dinamik Gemini 3.x Yapay Zekâ Üretim Katmanı (Zero-Hallucination)
# =====================================================================

class PureBM25:
    """
    Sıfır harici kütüphane bağımlılığı ile çalışan, Türkçe karakter uyumlu,
    yüksek performanslı yerel BM25 (Best Matching 25) arama algoritması sınıfı.
    """
    def __init__(self, documents, b=0.75, k1=1.5):
        self.b = b
        self.k1 = k1
        self.documents = documents
        self.corpus_size = len(documents)
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.doc_lengths = []
        self.df = {}
        self.idf = {}
        self.initialize()

    def tokenize(self, text):
        # Türkçe kelimeleri küçük harfe çevirme ve noktalama işaretlerini temizleme
        text = text.lower()
        text = text.replace('ı', 'i').replace('ğ', 'g').replace('ü', 'u').replace('ş', 's').replace('ö', 'o').replace('ç', 'c')
        words = re.findall(r'\b\w+\b', text)
        return words

    def initialize(self):
        total_len = 0
        for doc in self.documents:
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            
            # Kelime frekanslarını hesapla
            freqs = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            
            # Belge sıklığı (Document Frequency - DF) güncelle
            for token in freqs.keys():
                self.df[token] = self.df.get(token, 0) + 1
                
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0
        
        # IDF (Inverse Document Frequency) hesapla
        for word, df in self.df.items():
            # BM25 standart IDF formülü
            self.idf[word] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def get_score(self, query_tokens, doc_idx):
        score = 0.0
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        for token in query_tokens:
            if token not in doc_freq:
                continue
            tf = doc_freq[token]
            # BM25 TF-Doygunluk formülü
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += self.idf.get(token, 0.0) * (numerator / denominator)
        return score

    def search(self, query, top_n=5):
        query_tokens = self.tokenize(query)
        scores = []
        for idx in range(self.corpus_size):
            score = self.get_score(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))
        # Skora göre azalan sırada sırala
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


class GazaliHybridSearchEngine:
    def __init__(self, db_path="./gazali_chroma_db"):
        self.db_path = db_path
        self.selected_model = None
        self.available_models_list = []
        self.all_documents = []
        self.all_metadatas = []
        self.all_ids = []
        self.bm25_index = None
        
        # 1. Yerel Veri Tabanı Bağlantısı ve Tüm Verinin Çekilmesi
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"[!] '{self.db_path}' dizini bulunamadı! Lütfen önce veritabanını inşa edin."
            )
            
        print("[*] Yerel ChromaDB veritabanına bağlanılıyor...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        try:
            self.collection = self.chroma_client.get_collection(name="gazali_kulliyati")
        except Exception as e:
            raise ValueError(f"[!] Koleksiyon alınamadı: {e}")

        # Vektör veri tabanındaki tüm kayıtları BM25 dizini oluşturmak için belleğe çekelim
        print("[*] Veritabanındaki tüm notlar indeksleniyor...")
        db_data = self.collection.get()
        self.all_documents = db_data['documents']
        self.all_metadatas = db_data['metadatas']
        self.all_ids = db_data['ids']
        
        if not self.all_documents:
            print("[!] UYARI: Veritabanı şu an tamamen boş! Lütfen önce pipeline'ı çalıştırıp veri ekleyin.")
        else:
            # BM25 indeksini oluştur
            print(f"[+] {len(self.all_documents)} adet belge BM25 anahtar kelime motoruna yüklendi.")
            self.bm25_index = PureBM25(self.all_documents)

        # 2. Embedding Modelini Yükle
        print("[*] intfloat/multilingual-e5-large anlamsal modeli önbellekten çağrılıyor...")
        self.embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

        # 3. Gemini API Yapılandırması (v7 Dinamik Keşif Mekanizması)
        self.setup_gemini_api_safely()

    def setup_gemini_api_safely(self):
        """API Anahtarını doğrular ve aktif Gemini 3.x modellerini canlı tarar."""
        while True:
            api_key = os.environ.get("GEMINI_API_KEY")
            
            if api_key:
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
                print("\n" + "="*50)
                print(f"[*] Terminal oturumunda kayıtlı anahtar tespit edildi: {masked_key}")
                choice = input("👉 Bu kayıtlı anahtarı kullanmak istiyor musunuz? \n   (Kullanmak için Enter'a basın, YENİ girmek için 'y' yazın): ").strip().lower()
                
                if choice == 'y':
                    api_key = None
                    os.environ.pop("GEMINI_API_KEY", None)
            
            if not api_key:
                print("\n" + "="*50)
                print("[!] Lütfen Google AI Studio'dan aldığınız GÜNCEL Gemini API anahtarınızı girin.")
                print("[i] Ekran güvenliği için yazdıklarınız terminalde görünmeyecektir:")
                import getpass
                api_key = getpass.getpass("API Key: ").strip()
                if not api_key:
                    continue
                os.environ["GEMINI_API_KEY"] = api_key

            genai.configure(api_key=api_key)
            print("\n[*] Google sunucularına bağlanılıyor ve aktif modelleriniz listeleniyor...")
            try:
                raw_models = genai.list_models()
                self.available_models_list = []
                
                for m in raw_models:
                    if "generateContent" in m.supported_generation_methods:
                        m_name = m.name if m.name.startswith("models/") else f"models/{m.name}"
                        # Eski ve kısıtlı modelleri eliyoruz
                        if "gemini-pro" in m_name or "gemini-2.5" in m_name:
                            continue
                        self.available_models_list.append(m_name)
                
                if not self.available_models_list:
                    print("[❌] HATA: API anahtarınız başarılı fakat aktif bir model bulunamadı!")
                    os.environ.pop("GEMINI_API_KEY", None)
                    api_key = None
                    continue
                
                # Yeni nesil modelleri önceliklendirelim (Gemini 3.x ve Flash sürümleri)
                def model_priority(name):
                    name_lower = name.lower()
                    if "flash-latest" in name_lower: return 1
                    if "flash" in name_lower and "3" in name_lower: return 2
                    if "flash" in name_lower: return 3
                    if "pro" in name_lower and "3" in name_lower: return 4
                    return 10
                
                self.available_models_list.sort(key=model_priority)
                self.selected_model = self.available_models_list[0]
                
                # Bağlantı testi
                test_model = genai.GenerativeModel(model_name=self.selected_model)
                test_model.generate_content("test", generation_config={"max_output_tokens": 1})
                
                print(f"\n[+] API Bağlantısı BAŞARIYLA Doğrulandı! 🎉")
                print(f"[+] Seçilen Çalışma Modeli: {self.selected_model}")
                print("="*50 + "\n")
                break
                
            except Exception as e:
                print(f"\n[❌] API DOĞRULAMA HATASI: {e}")
                os.environ.pop("GEMINI_API_KEY", None)
                api_key = None

    def retrieve_hybrid(self, query, top_k=3, k_rrf=60):
        """
        BM25 ve Vektör aramalarını RRF (Reciprocal Rank Fusion) ile sentezler.
        RRF Formülü: RRF_Score(d) = sum(1 / (k_rrf + r_m(d)))
        """
        if not self.all_documents:
            return [], []

        # 1. VEKTÖR (DENSE) ARAMASI YAP
        formatted_query = f"query: {query}"
        query_vector = self.embed_model.encode(formatted_query).tolist()
        
        vector_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=len(self.all_documents) # Tüm adayları sırala
        )
        
        vector_ranking = []
        if vector_results and vector_results['ids'] and len(vector_results['ids'][0]) > 0:
            for i in range(len(vector_results['ids'][0])):
                doc_id = vector_results['ids'][0][i]
                vector_ranking.append(doc_id)

        # 2. BM25 (SPARSE) ARAMASI YAP
        bm25_raw_results = self.bm25_index.search(query, top_n=len(self.all_documents))
        bm25_ranking = []
        for idx, score in bm25_raw_results:
            bm25_ranking.append(self.all_ids[idx])

        # 3. RECIPROCAL RANK FUSION (RRF) SENTEZİ
        rrf_scores = {}
        
        # Vektör sıralama puanlarını ekle
        for rank, doc_id in enumerate(vector_ranking):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + (rank + 1)))
            
        # BM25 sıralama puanlarını ekle
        for rank, doc_id in enumerate(bm25_ranking):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + (rank + 1)))

        # RRF skoruna göre azalan sırada sırala
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # En iyi top_k belgeyi seçip detaylarını hazırlayalım
        retrieved_documents = []
        sources_info = []
        
        for doc_id, rrf_score in sorted_rrf[:top_k]:
            idx = self.all_ids.index(doc_id)
            doc_text = self.all_documents[idx]
            metadata = self.all_metadatas[idx]
            
            # Bu belgenin vektör ve BM25 sıralarını bulalım (Açıklayıcı raporlama için)
            v_rank = vector_ranking.index(doc_id) + 1 if doc_id in vector_ranking else "N/A"
            b_rank = bm25_ranking.index(doc_id) + 1 if doc_id in bm25_ranking else "N/A"
            
            retrieved_documents.append({
                "title": metadata['title'],
                "content": doc_text,
                "links": metadata.get('links', ''),
                "rrf_score": rrf_score
            })
            
            sources_info.append(
                f"[[{metadata['title']}]] (RRF Skoru: {rrf_score:.4f} | Vektör Sırası: {v_rank} | BM25 Sırası: {b_rank})"
            )
            
        return retrieved_documents, sources_info

    def generate_answer(self, query):
        """Hibrit aramadan beslenen kısıtlayıcı yapay zeka cevap motoru."""
        docs, sources_info = self.retrieve_hybrid(query, top_k=3)
        
        if not docs:
            return "Kütüphanenizde arama sorgusunu karşılayacak yeterli bilgi bulunamadı.", []
            
        context_str = ""
        for idx, doc in enumerate(docs):
            context_str += f"\n--- KAYNAK NOT: [[{doc['title']}]] ---\n"
            context_str += f"İlişkili Diğer Notlar: {doc['links']}\n"
            context_str += f"İçerik:\n{doc['content']}\n"
            context_str += "--------------------------------------\n"

        system_instruction = """Sen İmam Gazali felsefesi üzerine akademik derinliğe sahip bir yapay zeka araştırma asistanısın.
Görevin, kullanıcının sorusuna SADECE sana sunulan 'KAYNAK NOTLAR' kapsamındaki verileri kullanarak yanıt vermektir.

Uyman Gereken Kesin Kurallar:
1. Sadece sana verilen kaynaklardaki bilgileri kullan. Dışarıdan genel kültür veya internet bilgisi kesinlikle ekleme.
2. Eğer sorunun cevabı kaynaklarda geçmiyorsa kesinlikle uydurma. "Bu bilgi kaynaklarımızda bulunmamaktadır." de.
3. Cevap verirken her akademik iddiadan sonra hangi belgeden alındığını çift köşeli parantezle belirt. (Örn: "...kalbe inen ilahi bir nurdur [[Şüphe (Epistemik Kriz)]].")
4. Son derece saygın, ağırbaşlı ve akademik bir Türkçe üslubu kullan.
"""

        user_message = f"""Kullanıcı Sorusu: {query}

Sana Sunulan Kaynak Notlar (BM25 + Vektör Hibrit Arama Sonuçları):
==================================================
{context_str}
==================================================

Yukarıdaki kaynak notlara ve kurallara tamamen bağlı kalarak soruyu yanıtla:"""

        model = genai.GenerativeModel(
            model_name=self.selected_model,
            system_instruction=system_instruction
        )
        
        response = model.generate_content(
            contents=user_message,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        
        return response.text, sources_info


# =====================================================================
# HİBRİT ETKİLEŞİMLİ KONSOL (CLI)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🕌 GAZALİ KÜLLİYATI HİBRİT (BM25 + VEKTÖR) ARAMA MOTORU (v1.0)")
    print("="*60)
    
    try:
        engine = GazaliHybridSearchEngine()
        print("\n[+] Hibrit Arama Sistemi hazır! Büyük metinleri taramaya başlayabilirsiniz.")
        print("[i] Çıkmak için 'q' yazabilirsiniz.\n")
        
        while True:
            soru = input("👉 Aramak İstediğiniz Kavram veya Soru: ").strip()
            
            if not soru:
                continue
                
            if soru.lower() in ['q', 'exit', 'quit']:
                print("\n[*] Oturum kapatılıyor. Başarılar dileriz!")
                break
                
            try:
                cevap, kaynaklar = engine.generate_answer(soru)
                
                print("\n🔬 [HİBRİT SENTEZ (RRF) SIRALAMASI VE ÇEKİLEN BAĞLAMLAR]")
                print("-" * 70)
                for k in kaynaklar:
                    print(f"   • {k}")
                print("-" * 70)
                
                print(f"\n🤖 [YAPAY ZEKA CEVABI ({engine.selected_model}) (SIFIR HALÜSİNASYON)]")
                print("-" * 70)
                print(cevap)
                print("-" * 70 + "\n")
                
            except Exception as e:
                print(f"\n[!] Bir hata oluştu: {e}\n")
                
    except Exception as e:
        print(f"\n[!] Sistem başlatılamadı: {e}")
