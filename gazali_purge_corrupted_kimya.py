"""
Bozuk Kimyâ-yı Saâdet Chunk Temizleyici
========================================
Eldeki Kimyâ-yı Saâdet PDF'inin metin katmanı bozuk (harfler boşluklarla kopuk;
"k ıy am ette b u n d a n" gibi). Standart çıkarıcılar (pypdf/PyMuPDF) bunu
düzeltemiyor. Bu script, korpustaki Kimya chunk'larından tek-harf token oranı
yüksek (okunamaz) olanları siler; okunabilir olanları korur.

Temiz metin katmanlı bir Kimya PDF bulununca eser yapı-farkında yeniden yüklenir.
Silinenler o zaman zaten üzerine yazılır.
"""
import chromadb

DB_PATH = "gazali_chroma_db"
COLLECTION = "gazali_kulliyati"
THRESHOLD = 0.15  # tek-harf token oranı bunun üstündeyse "okunamaz" say


def single_char_ratio(text):
    toks = text.split()
    return (sum(1 for w in toks if len(w) == 1) / len(toks)) if toks else 0.0


def main():
    col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)
    d = col.get(include=["documents", "metadatas"])

    to_delete = [
        i for i, t, m in zip(d["ids"], d["documents"], d["metadatas"])
        if m.get("book", "").lower().startswith("kimya") and single_char_ratio(t) > THRESHOLD
    ]
    if not to_delete:
        print("[i] Silinecek bozuk Kimya chunk bulunamadı.")
        return

    before = col.count()
    col.delete(ids=to_delete)
    after = col.count()
    print(f"[✓] Bozuk Kimya temizlendi: {len(to_delete)} chunk silindi.")
    print(f"    Korpus: {before} → {after} chunk")


if __name__ == "__main__":
    main()
