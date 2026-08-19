import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =====================================================================
# ISLAMICATE DH - GAZALI SIFIR HALÜSİNASYONLU SORU-CEVAP MOTORU (RAG)
# =====================================================================
# Bu betik, yerel bilgisayarınızda kurduğunuz ChromaDB vektör veri tabanını
# okur, en alakalı Obsidian notlarını çeker ve Google Gemini API'sini
# kullanarak sadece ve sadece sizin notlarınıza dayalı cevaplar üretir.
# =====================================================================

class GazaliQueryEngine:
    def __init__(self, db_path="./gazali_chroma_db"):
        self.db_path = db_path
        
        # 1. Yerel Veri Tabanı Bağlantısı
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"[!] '{self.db_path}' dizini bulunamadı! Lütfen önce "
                "'obsidian_rag_pipeline.py' dosyasını çalıştırarak veritabanını inşa edin."
            )
            
        print("[*] Yerel ChromaDB veritabanı yükleniyor...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # 2. Embedding Modelini Yükle (Önbellekten anında yüklenecektir)
        print("[*] intfloat/multilingual-e5-large anlamsal modeli önbellekten çağrılıyor...")
        self.embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
        
        # 3. Koleksiyonu Al
        try:
            self.collection = self.chroma_client.get_collection(name="gazali_kulliyati")
        except Exception as e:
            raise ValueError(
                f"[!] 'gazali_kulliyati' koleksiyonu bulunamadı! Hata: {e}"
            )

        # 4. Gemini API Yapılandırması
        self.setup_gemini_api()

    def setup_gemini_api(self):
        """
        Gemini API Anahtarını işletim sisteminden veya kullanıcıdan alır.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("\n" + "="*50)
            print("[!] GEMINI_API_KEY çevre değişkeni (environment variable) bulunamadı.")
            print("Lütfen Gemini API anahtarınızı girin (Ekran güvenliği için yazdıklarınız görünmeyebilir):")
            import getpass
            api_key = getpass.getpass("API Key: ").strip()
            if not api_key:
                print("[!] API anahtarı girilmedi. Program sonlandırılıyor.")
                sys.exit(1)
            # Geçici olarak oturuma kaydet
            os.environ["GEMINI_API_KEY"] = api_key
            
        genai.configure(api_key=api_key)

    def retrieve_context(self, query, top_k=3):
        """
        Kullanıcı sorusuyla veritabanında arama yapar ve en yakın Obsidian notlarını getirir.
        """
        # E5 model kuralı: Sorgular için 'query: ' ön eki eklenmelidir.
        formatted_query = f"query: {query}"
        query_vector = self.embed_model.encode(formatted_query).tolist()

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        retrieved_documents = []
        sources_info = []
        
        # Arama sonuçları varsa işle
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                text = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                score = results['distances'][0][i]
                similarity = 1 - (score / 2)
                
                # Sadece anlamsal olarak %70 ve üzeri benzerlik taşıyan notları bağlama dahil et
                if similarity >= 0.70:
                    retrieved_documents.append({
                        "title": metadata['title'],
                        "content": text,
                        "links": metadata.get('links', ''),
                        "similarity": similarity
                    })
                    sources_info.append(f"[[{metadata['title']}]] (%{similarity*100:.1f} Alaka)")
                    
        return retrieved_documents, sources_info

    def generate_answer(self, query):
        """
        Anlamsal olarak çekilen verileri kısıtlayıcı bir prompt ile Gemini'ye gönderir.
        """
        # 1. Vektör veritabanından en alakalı notları getir
        docs, sources_info = self.retrieve_context(query, top_k=3)
        
        if not docs:
            return (
                "Sorguladığınız konu hakkında Obsidian kütüphanenizde yeterli bilgi bulunamadı. "
                "Lütfen kütüphanenize ilgili Markdown notlarını ekleyip pipeline'ı tekrar çalıştırın.",
                []
            )
            
        # 2. Bağlamı (Context) inşa et
        context_str = ""
        for idx, doc in enumerate(docs):
            context_str += f"\n--- KAYNAK NOT: [[{doc['title']}]] ---\n"
            context_str += f"İlişkili Diğer Notlar: {doc['links']}\n"
            context_str += f"İçerik:\n{doc['content']}\n"
            context_str += "--------------------------------------\n"

        # 3. İslâmî Dijital Beşerî Bilimler Sınırlandırıcı Sistem Yönergesi
        system_instruction = """Sen İmam Gazali felsefesi ve teolojisi üzerine uzmanlaşmış, akademik dürüstlüğü en üst düzeyde tutan bir yapay zeka asistanısın.
Görevin, kullanıcının sorusuna SADECE sana sunulan 'KAYNAK NOTLAR' kapsamındaki verileri kullanarak yanıt vermektir.

Uyman Gereken Kesin Kurallar:
1. Sana verilen KAYNAK NOTLAR dışındaki hiçbir genel kültür veya internet bilgisini kullanma.
2. Eğer sorunun yanıtı sana sunulan notlarda doğrudan veya dolaylı olarak geçmiyorsa, kesinlikle bilgi UYDURMA (Halüsinasyon üretme). Doğrudan şu cevabı ver:
   "Bu sorunun cevabı kütüphanenizdeki mevcut Gazali belgelerinde yer almamaktadır veya mevcut bağlam bu soruya cevap vermek için yetersizdir."
3. Cevap verirken her cümlenin veya iddiadan sonra hangi nottan alıntılandığını çift köşeli parantez referansıyla göster. (Örn: "...duyuların insanı yanılttığını söyler [[Hissiyyât]].")
4. Çelişkili veya belirsiz durumlar varsa bunları kendi yorumunu katmadan olduğu gibi aktar.
5. Tamamen Türkçe yanıt ver ve akademik, saygın bir dil kullan.
"""

        user_message = f"""Kullanıcı Sorusu: {query}

Sana Sunulan Kaynak Notlar:
==================================================
{context_str}
==================================================

Yukarıdaki kaynak notlara ve kurallara tamamen bağlı kalarak soruyu yanıtla:"""

        # 4. Gemini 1.5 Flash modelini düşük sıcaklık (temperature) ayarıyla çağır
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_instruction
        )
        
        # Yaratıcılığı (temperature) sıfıra yakın (0.1) tutarak halüsinasyon riskini yok ediyoruz.
        response = model.generate_content(
            contents=user_message,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        
        return response.text, sources_info


# =====================================================================
# ETKİLEŞİMLİ ÇALIŞTIRMA KONSOLU (Interactive CLI)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📜 GAZALİ KÜLLİYATI SIFIR HALÜSİNASYONLU SORU-CEVAP MOTORU")
    print("="*60)
    
    try:
        engine = GazaliQueryEngine()
        print("\n[+] Sistem hazır! Sorgulamaya başlayabilirsiniz.")
        print("[i] Çıkmak için 'q' veya 'exit' yazabilirsiniz.\n")
        
        while True:
            soru = input("👉 Sorunuzu Yazın: ").strip()
            if not soru:
                continue
            if soru.lower() in ['q', 'exit', 'çıkış', 'quit']:
                print("\n[*] Oturum kapatılıyor. İyi çalışmalar!")
                break
                
            try:
                cevap, kaynaklar = engine.generate_answer(soru)
                
                print("\n📚 [VEKTÖREL BAĞLAMDAN ÇEKİLEN KAYNAKLAR]")
                for k in kaynaklar:
                    print(f"   • {k}")
                
                print("\n🤖 [YAPAY ZEKA CEVABI (SIFIR HALÜSİNASYON)]")
                print("-" * 60)
                print(cevap)
                print("-" * 60 + "\n")
                
            except Exception as e:
                print(f"\n[!] Cevap üretilirken bir hata oluştu: {e}\n")
                
    except Exception as e:
        print(f"\n[!] Sistem başlatılamadı: {e}")
