"""
Gazâlî Korpus Temizleme Aracı (Contaminant Purge)
=================================================
ChromaDB korpusundan, kaynak eser olmayan üretilmiş/config artefaktlarını
(RAG değerlendirme raporu, README, requirements vb.) siler. Bu chunk'lar
retrieval'ı kirletir: örn. rag_evaluation_report.md testin sorgu kelimelerini
birebir içerdiği için TC-001'de gerçek kaynağı (Kimyâ-yı Saâdet) üst sıradan iter.

Silinmeden önce yedek alın:  cp -R gazali_chroma_db gazali_chroma_db.backup
"""
import sys
import chromadb

DB_PATH = "gazali_chroma_db"
COLLECTION = "gazali_kulliyati"
# Korpustan silinecek 'book' metadata değerleri (kaynak eser DEĞİL)
CONTAMINANT_BOOKS = {"Rag Evaluation Report", "Readme", "Requirements Copy"}


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_collection(COLLECTION)

    data = col.get(include=["metadatas"])
    ids_to_del = [i for i, m in zip(data["ids"], data["metadatas"])
                  if m.get("book") in CONTAMINANT_BOOKS]

    if not ids_to_del:
        print("[i] Silinecek kirli chunk bulunamadı. Korpus zaten temiz.")
        return

    before = col.count()
    print(f"[*] Silinecek chunk sayısı: {len(ids_to_del)}")
    for b in sorted(CONTAMINANT_BOOKS):
        n = sum(1 for m in data["metadatas"] if m.get("book") == b)
        if n:
            print(f"    • {b!r}: {n} chunk")

    col.delete(ids=ids_to_del)
    after = col.count()
    print(f"[✓] Tamamlandı. Önce {before} → Sonra {after} chunk (silinen: {before - after})")


if __name__ == "__main__":
    main()
