import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =====================================================================
# ISLAMICATE DH - GAZALI SIFIR HALÜSİNASYONLU SORU-CEVAP MOTORU (v7)
# =====================================================================
# Bu sürüm (v7), Google'ın dinamik model değişikliklerini ve kısıtlamalarını
# aşmak için geliştirilmiş "Dinamik Model Keşif ve Tanı Asistanı" içerir.
# Sabit model isimleri kullanmak yerine, API anahtarınızın Google sunucularında
# bizzat izinli olduğu TÜM modelleri canlı olarak listeler ve en uygununu seçer.
# =====================================================================

class GazaliQueryEngine:
    def __init__(self, db_path="./gazali_chroma_db"):
        self.db_path = db_path
        self.selected_model = None
        self.available_models_list = []
        
        # 1. Yerel Veri Tabanı Bağlantısı
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"[!] '{self.db_path}' dizini bulunamadı! Lütfen önce "
                "'obsidian_rag_pipeline.py' dosyasını çalıştırarak veritabanını inşa edin."
            )
            
        print("[*] Yerel ChromaDB veritabanı yükleniyor...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # 2. Embedding Modelini Yükle
        print("[*] intfloat/multilingual-e5-large anlamsal modeli önbellekten çağrılıyor...")
        self.embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
        
        # 3. Koleksiyonu Al
        try:
            self.collection = self.chroma_client.get_collection(name="gazali_kulliyati")
        except Exception as e:
            raise ValueError(
                f"[!] 'gazali_kulliyati' koleksiyonu bulunamadı! Hata: {e}"
            )

        # 4. Gemini API Yapılandırması ve Güvenli Doğrulama
        self.setup_gemini_api_safely()

    def setup_gemini_api_safely(self):
        """
        API Anahtarını alır, doğrular ve anahtarın izin verdiği TÜM modelleri canlı tarar.
        """
        while True:
            api_key = os.environ.get("GEMINI_API_KEY")
            
            if api_key:
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
                print("\n" + "="*50)
                print(f"[*] Terminal oturumunda kayıtlı bir anahtar tespit edildi: {masked_key}")
                choice = input("👉 Bu kayıtlı anahtarı kullanmak istiyor musunuz? \n   (Kullanmak için Enter'a basın, YENİ anahtar girmek için 'y' yazın): ").strip().lower()
                
                if choice == 'y':
                    print("[*] Oturumdaki eski anahtar göz ardı ediliyor. Lütfen yeni anahtarınızı girin.")
                    api_key = None
                    os.environ.pop("GEMINI_API_KEY", None)
            
            if not api_key:
                print("\n" + "="*50)
                print("[!] Lütfen Google AI Studio'dan aldığınız GÜNCEL Gemini API anahtarınızı girin.")
                print("[i] Ekran güvenliği için yazdıklarınız terminalde görünmeyecektir (CMD+V yapıp Enter'a basabilirsiniz):")
                import getpass
                api_key = getpass.getpass("API Key: ").strip()
                if not api_key:
                    print("[!] Anahtar girilmedi. Tekrar deneniyor...")
                    continue
                os.environ["GEMINI_API_KEY"] = api_key

            # API Yapılandır
            genai.configure(api_key=api_key)
            
            # Bağlantıyı test et ve canlı modelleri listele
            print("\n[*] Google sunucularına bağlanılıyor ve API yetkileriniz sorgulanıyor...")
            try:
                # 1. Google sunucularından bu anahtara izin verilen tüm modelleri çekelim
                raw_models = genai.list_models()
                self.available_models_list = []
                
                print("\n📋 [API ANAHTARINIZIN ERİŞEBİLDİĞİ TÜM MODELLER]:")
                for m in raw_models:
                    if "generateContent" in m.supported_generation_methods:
                        # 'models/' ön ekini temizleyelim veya koruyalım
                        m_name = m.name if m.name.startswith("models/") else f"models/{m.name}"
                        # Bilinen eski/kullanımdan kalkan modelleri eliyoruz (gemini-pro, gemini-2.5 gibi kalkanlar)
                        if "gemini-pro" in m_name or "gemini-2.5" in m_name:
                            print(f"   • {m_name} (Sistem tarafından elendi - Eski/Kısıtlı)")
                            continue
                        self.available_models_list.append(m_name)
                        print(f"   • {m_name} [Erişilebilir ve Aktif ✅]")
                
                if not self.available_models_list:
                    print("\n[❌] HATA: API anahtarınız başarılı şekilde doğrulandı fakat içerik üretimi (generateContent) yapabilen hiçbir aktif model bulunamadı!")
                    os.environ.pop("GEMINI_API_KEY", None)
                    api_key = None
                    continue
                
                # Modelleri öncelik sırasına göre sıralayalım (Flash modelleri daha hızlı ve stabil olduğu için önceliklidir)
                # 2026 yılı standartlarında çalışan yeni modelleri de kapsayacak dinamik bir önceliklendirme
                def model_priority(name):
                    name_lower = name.lower()
                    if "flash-latest" in name_lower: return 1
                    if "flash" in name_lower and "1.5" in name_lower: return 2
                    if "flash" in name_lower: return 3
                    if "pro-latest" in name_lower: return 4
                    if "pro" in name_lower and "1.5" in name_lower: return 5
                    if "pro" in name_lower: return 6
                    return 10
                
                self.available_models_list.sort(key=model_priority)
                
                # İlk sıradaki modeli varsayılan olarak test edelim
                self.selected_model = self.available_models_list[0]
                
                # Küçük bir deneme üretimiyle bağlantıyı kesinleştirelim
                test_model = genai.GenerativeModel(model_name=self.selected_model)
                test_model.generate_content("test", generation_config={"max_output_tokens": 1})
                
                print(f"\n[+] API Bağlantısı BAŞARIYLA Doğrulandı! 🎉")
                print(f"[+] Otomatik Seçilen En Uygun Model: {self.selected_model}")
                print("="*50 + "\n")
                break
                
            except Exception as e:
                print(f"\n[❌] API BAĞLANTI VEYA DOĞRULAMA HATASI!")
                print(f"Detaylı Hata Mesajı: {e}")
                print("\n[*] Lütfen girilen anahtarın doğru olduğundan ve internet bağlantınızın (varsa VPN kısıtlamasının) sorunsuz olduğundan emin olun.")
                os.environ.pop("GEMINI_API_KEY", None)
                api_key = None

    def retrieve_context(self, query, top_k=3):
        """
        Kullanıcı sorusuyla veritabanında arama yapar ve en yakın Obsidian notlarını getirir.
        """
        formatted_query = f"query: {query}"
        query_vector = self.embed_model.encode(formatted_query).tolist()

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        retrieved_documents = []
        sources_info = []
        
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                text = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                score = results['distances'][0][i]
                similarity = 1 - (score / 2)
                
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
        docs, sources_info = self.retrieve_context(query, top_k=3)
        
        if not docs:
            return (
                "Sorguladığınız konu hakkında Obsidian kütüphanenizde yeterli bilgi bulunamadı. "
                "Lütfen kütüphanenize ilgili Markdown notlarını ekleyip pipeline'ı tekrar çalıştırın.",
                []
            )
            
        context_str = ""
        for idx, doc in enumerate(docs):
            context_str += f"\n--- KAYNAK NOT: [[{doc['title']}]] ---\n"
            context_str += f"İlişkili Diğer Notlar: {doc['links']}\n"
            context_str += f"İçerik:\n{doc['content']}\n"
            context_str += "--------------------------------------\n"

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

        # En kararlı çalışan modeli kullanarak cevap üret
        last_error = None
        for model_name in self.available_models_list:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
                )
                
                response = model.generate_content(
                    contents=user_message,
                    generation_config=genai.types.GenerationConfig(temperature=0.1)
                )
                
                # Başarılı olan modeli sabitleyelim
                self.selected_model = model_name
                return response.text, sources_info
            except Exception as e:
                last_error = e
                continue
                
        raise RuntimeError(f"Canlı modellerin hiçbiriyle cevap üretilemedi. Son hata: {last_error}")


# =====================================================================
# ETKİLEŞİMLİ ÇALIŞTIRMA KONSOLU (Interactive CLI)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📜 GAZALİ KÜLLİYATI SIFIR HALÜSİNASYONLU SORU-CEVAP MOTORU (v7)")
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
                
                print(f"\n🤖 [YAPAY ZEKA CEVABI (Model: {engine.selected_model}) (SIFIR HALÜSİNASYON)]")
                print("-" * 60)
                print(cevap)
                print("-" * 60 + "\n")
                
            except Exception as e:
                print(f"\n[!] Cevap üretilirken bir hata oluştu: {e}\n")
                
    except Exception as e:
        print(f"\n[!] Sistem başlatılamadı: {e}")
