import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

# =====================================================================
# ISLAMICATE DH - OBSIDIAN VAULT RAG INGESTION PIPELINE
# =====================================================================
# This script reads your Obsidian Markdown files, extracts bidirectionally
# linked concepts (e.g. [[Yakin]], [[Nefs]]), and stores them along with 
# the semantic embeddings in a local ChromaDB instance.
#
# Model used: intfloat/multilingual-e5-large (Best-in-class for Arabic/Turkish)
# =====================================================================

class ObsidianRAGPipeline:
    def __init__(self, vault_path, db_path="./gazali_chroma_db"):
        self.vault_path = vault_path
        self.db_path = db_path
        
        # Initialize Local Chroma Client (Persistent Storage)
        print("[*] Local Chroma DB başlatılıyor...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Initialize Multilingual Embedding Model (Mevzu 2026 makalesinde önerilen SOTA model)
        print("[*] intfloat/multilingual-e5-large modeli yükleniyor...")
        print("[i] Not: Bu model ilk kez çalıştırıldığında yerel bilgisayarınıza indirilecektir (~2.24 GB).")
        self.embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
        
        # Create or Get Collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="gazali_kulliyati",
            metadata={"hnsw:space": "cosine"} # Kosinüs benzerliği (Cosine Similarity) kullanılarak sorgulama yapılır
        )

    def extract_obsidian_links(self, content):
        """
        Regex kullanarak metindeki [[Çift Köşeli Parantez]] içindeki Obsidian iç bağlantılarını bulur.
        """
        # [[Kavram]] veya [[Kavram|Görünen Ad]] formatlarını yakalar
        links = re.findall(r'\[\[(.*?)\]\]', content)
        cleaned_links = []
        for link in links:
            # Eğer alias (|) varsa, sadece hedef not adını al
            target = link.split('|')[0].strip()
            cleaned_links.append(target)
        return list(set(cleaned_links))

    def clean_text_for_embedding(self, content):
        """
        Vektörleştirme işleminin daha temiz olması için Obsidian wikilink parantezlerini kaldırır:
        [[Yakin]] -> Yakin
        """
        # [[Kavram|Görünen Ad]] -> Görünen Ad
        content = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', content)
        # [[Kavram]] -> Kavram
        content = re.sub(r'\[\[(.*?)\]\]', r'\1', content)
        return content

    def ingest_vault(self):
        """
        Obsidian klasöründeki tüm .md dosyalarını okur, anlamsal özelliklerini ve bağlantılarını çıkarıp
        veritabanına yükler.
        """
        if not os.path.exists(self.vault_path):
            raise FileNotFoundError(f"[!] Belirtilen Obsidian Vault dizini bulunamadı: {self.vault_path}")

        md_files = [f for f in os.listdir(self.vault_path) if f.endswith(".md")]
        print(f"\n[*] Klasörde {len(md_files)} adet Markdown dosyası tespit edildi.")

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for file_name in md_files:
            file_path = os.path.join(self.vault_path, file_name)
            title = os.path.splitext(file_name)[0]
            
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            # 1. Obsidian iç bağlantılarını ayıkla
            links = self.extract_obsidian_links(raw_content)
            
            # 2. Embedding kalitesi için metni temizle ([[Parantezleri]] kaldır)
            clean_content = self.clean_text_for_embedding(raw_content)
            
            # E5 modeli için girdi ön eki: E5 modelleri arama kalitesi için 'passage: ' veya 'query: ' ön eki gerektirir.
            # Dökümanlar için 'passage: ', sorgular için 'query: ' eklenir.
            formatted_passage = f"passage: {clean_content}"
            
            # 3. Embedding (Vektör) Üret
            print(f"[*] '{title}' belgesi vektörleştiriliyor...")
            vector = self.embed_model.encode(formatted_passage).tolist()

            # 4. Listelere Ekle
            documents.append(clean_content)
            embeddings.append(vector)
            
            # Metadata alanında ilişkisel veriyi (Graph verisini) saklıyoruz!
            metadatas.append({
                "title": title,
                "links": ", ".join(links),
                "source": file_name
            })
            ids.append(title)  # ID olarak notun kendi başlığını kullanıyoruz (benzersizdir)

        # 5. ChromaDB'ye Toplu Yükleme (Upsert)
        if ids:
            print("\n[*] Veriler ChromaDB'ye yazılıyor...")
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            print("[+] Yükleme işlemi BAŞARIYLA tamamlandı! 🚀")
        else:
            print("[!] Klasörde işlenecek döküman bulunamadı.")

    def semantic_search(self, query, top_k=3):
        """
        Doğal dil sorusuyla veritabanında kosinüs benzerliği üzerinden anlamsal arama yapar.
        """
        # E5 model kuralı: Sorgular için 'query: ' ön eki eklenmelidir.
        formatted_query = f"query: {query}"
        query_vector = self.embed_model.encode(formatted_query).tolist()

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        print(f"\n🔍 '{query}' için en alakalı {top_k} sonuç:\n" + "="*50)
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            score = results['distances'][0][i] # Cosine distance (0 en yakın, 2 en uzak)
            similarity = 1 - (score / 2) # Benzerlik oranı
            metadata = results['metadatas'][0][i]
            text = results['documents'][0][i]

            print(f"📌 Başlık: {metadata['title']} (Benzerlik Oranı: %{similarity*100:.2f})")
            print(f"🔗 İlişkili Kavramlar: [{metadata['links']}]")
            print(f"📖 Alıntı:\n{text[:250]}...")
            print("-"*50)


# =====================================================================
# ÇALIŞTIRMA ÖRNEĞİ
# =====================================================================
if __name__ == "__main__":
    # Kendi yerel bilgisayarınızdaki Obsidian klasör yolunu buraya yazın
    VAULT_YOLU = "./"
    
    # Adım 1: Pipeline sınıfını başlat (Modeli yükle ve Veritabanı bağlantısı kur)
    try:
        pipeline = ObsidianRAGPipeline(vault_path=VAULT_YOLU)
        
        # Adım 2: Obsidian notlarını tarayıp veritabanını oluştur/güncelle
        # Not: Yerel dizinde ./gazaliprojesi klasörünü oluşturup içine notları koyduktan sonra bu satırı çalıştırın.
        if os.path.exists(VAULT_YOLU):
            pipeline.ingest_vault()
            
            # Adım 3: Test Sorguları Yap (Anlamsal Arama)
            pipeline.semantic_search("İmam Gazali duyusal algıların yanılgısı hakkında ne diyor?")
            pipeline.semantic_search("Aklın kesin gördüğü zorunlu ilkelerin şüpheye düşme aşaması")
        else:
            print(f"\n[!] Lütfen önce bilgisayarınızda '{VAULT_YOLU}' klasörünü oluşturup içine Obsidian notlarınızı kopyalayın.")
            print("[*] Ardından bu betiği terminalde çalıştırarak Obsidian verilerinizi saniyeler içinde RAG beynine aktarabilirsiniz.")
            
    except Exception as e:
        print(f"[!] Hata oluştu: {e}")
