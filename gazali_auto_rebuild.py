import os
import sys

# Proje dizinini yola ekle
sys.path.append(os.getcwd())

# Mevcut yükleme aracından sınıfımızı içe aktarıyoruz
try:
    from gazali_book_ingester import GazaliBookIngester
except ImportError:
    try:
        # Eğer v3 dosyası ana dosya ise ondan dene
        from gazali_book_ingester_v3 import GazaliBookIngester
    except ImportError:
        print("[❌] Hata: Kitap yükleme sınıfı bulunamadı!")
        sys.exit(1)

def auto_rebuild():
    print("\n" + "="*60)
    print("⚡ GAZALİ PORTALI - OTOMATİK VERİTABANI YENİDEN İNŞA ARACI")
    print("="*60)

    # 1. Yükleyiciyi başlat
    ingester = GazaliBookIngester()

    # 2. Klasördeki tüm dosyaları tara
    all_files = os.listdir(".")
    
    # Yüklenecek hedef dosya uzantıları
    target_extensions = [".docx", ".pdf", ".md", ".txt"]
    # Hariç tutulacak sistem/kod dosyaları (hepsi küçük harfle; karşılaştırma da küçük harfle yapılır)
    ignored_files = {
        "requirements.txt", "readme.md", "github-readme.md",
        "github-readme-v2.md", "github-readme-v3.md",
        "gazali-rag-rehberi.md", "gazali-veri-envanteri.md",
        # Üretilmiş çıktı / config dosyaları (korpusa girmemeli, yoksa retrieval'ı kirletir)
        "rag_evaluation_report.md", "license", "license.md",
    }
    # Üretilmiş/yardımcı dosyaları desen bazında da ele: requirements*.txt, *_report.md
    def is_ignored(name_lower):
        if name_lower in ignored_files:
            return True
        if name_lower.startswith("requirements"):
            return True
        if name_lower.endswith("_report.md") or name_lower.endswith("-report.md"):
            return True
        return False

    print("\n[*] Klasör taranıyor ve kitaplar tespit ediliyor...")
    files_to_ingest = []

    for file in all_files:
        name_lower = file.lower()
        ext = os.path.splitext(name_lower)[1]

        # Eğer dosya bir kitap/makale ise ve hariç tutulacaklar listesinde değilse ekle
        if ext in target_extensions and not is_ignored(name_lower) and not name_lower.startswith("."):
            files_to_ingest.append(file)

    if not files_to_ingest:
        print("[⚠️] Klasörde yüklenecek uygun kitap veya makale bulunamadı!")
        return

    print(f"[+] Toplam {len(files_to_ingest)} adet kaynak bulundu:")
    for f in files_to_ingest:
        print(f"  • {f}")

    # 3. Hepsini sırayla ve otomatik olarak yükle
    print("\n[🚀] Toplu yükleme işlemi başlatılıyor...")
    for idx, file in enumerate(files_to_ingest, 1):
        print(f"\n[{idx}/{len(files_to_ingest)}] Yükleniyor: {file}")
        try:
            # Kitap başlığını dosya adından temiz bir şekilde türetelim
            clean_title = os.path.splitext(file)[0].replace("_", " ").title()
            ingester.ingest_book(file_path=file, book_title=clean_title)
            print(f"[✅] Başarıyla tamamlandı: {file}")
        except Exception as e:
            print(f"[❌] Hata oluştu ({file}): {e}")

    print("\n" + "="*60)
    print("🎉 TERTEMİZ YENİ VERİTABANINIZ HAZIR!")
    print("="*60 + "\n")

if __name__ == "__main__":
    auto_rebuild()
