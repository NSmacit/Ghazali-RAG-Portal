"""
İhyâu Ulûmi'd-Dîn — Yapı-farkında Ingester (1. Cilt)
=====================================================
Ali Arslan tercümesi (günümüz Türkçesiyle) 1272 sayfalık PDF'i:
  • Sayfa aralığına göre her paragrafı ait olduğu KİTABA (Kitâbü'l-İlim,
    Kavâidü'l-Akâid, ...) etiketler — böylece kullanıcı "sadece İhyâ'nın
    İlim kitabında ara" diyebilir ve eserde boğulmaz.
  • Toplu (batch) embedding ile hızlı vektörleştirir.
  • Zengin metadata (eser / cilt / kitap / sayfa) ile ChromaDB'ye ekler.

Kullanım:
  python3 gazali_ihya_ingester.py --dry-run   # sadece önizleme (model yüklenmez)
  python3 gazali_ihya_ingester.py             # gerçek yükleme
Kaldırmak için: ID öneki 'ihya_c1_' olan chunk'ları silmek yeterlidir.
"""
import os
import re
import sys
import argparse

PDF_PATH = "/Users/macit/Desktop/İHYAU ULUMİDDİN CİLD 1 İMAM GAZALİ - TERCÜME ALİ ARSLAN - GÜNÜMÜZ TÜRKÇESİYLE.pdf"
ESER = "İhyâu Ulûmi'd-Dîn"
CILT = 1
DB_PATH = "./gazali_chroma_db"
COLLECTION = "gazali_kulliyati"
MAX_CHUNK = 1200
OVERLAP = 200

# PDF sayfa indeksine (0-tabanlı) göre kanonik kitap aralıkları.
# Değerler, başlıkların ilk tespit edildiği sayfalardan türetildi.
BOOK_RANGES = [
    (0,    "Giriş ve Mukaddime"),
    (70,   "Kitâbü'l-İlim"),
    (354,  "Kitâbu Kavâidi'l-Akâid"),
    (482,  "Kitâbu Esrâri't-Tahâre"),
    (546,  "Kitâbu Esrâri's-Salât"),
    (748,  "Kitâbu Esrâri'z-Zekât"),
    (832,  "Kitâbu Esrâri's-Savm"),
    (860,  "Kitâbu Esrâri'l-Hac"),
    (968,  "Kitâbu Âdâbı Tilâveti'l-Kur'ân"),
    (1042, "Kitâbu'z-Zikr ve'd-Da'avât"),
    (1162, "Kitâbu Tertîbi'l-Evrâd"),
]


def book_for_page(idx):
    name = BOOK_RANGES[0][1]
    for start, bname in BOOK_RANGES:
        if idx >= start:
            name = bname
        else:
            break
    return name


def build_chunks():
    """PDF'i okur, kitap sınırlarına saygılı ~MAX_CHUNK boy paragraflara böler."""
    import pypdf
    reader = pypdf.PdfReader(PDF_PATH)
    n_pages = len(reader.pages)

    chunks = []
    buf, buf_page, buf_book = "", None, None

    def flush():
        nonlocal buf, buf_page, buf_book
        if buf.strip():
            chunks.append({"text": buf.strip(), "page": buf_page, "kitap": buf_book})
        buf = ""

    def add_piece(piece, page, book):
        """Bir metin parçasını (paragraf/cümle) buffer'a ekler; taşarsa flush eder."""
        nonlocal buf, buf_page, buf_book
        if len(buf) + len(piece) + 2 <= MAX_CHUNK:
            buf = (buf + "\n\n" + piece) if buf else piece
        else:
            flush()
            ov = " ".join(buf.split()[-OVERLAP // 6:]) if buf else ""
            buf = (ov + "\n\n" + piece) if ov else piece
            buf_page, buf_book = page, book

    for idx in range(n_pages):
        text = (reader.pages[idx].extract_text() or "").strip()
        if not text:
            continue
        book = book_for_page(idx)
        page = idx + 1  # basılı sayfa numarası (0-tabanlı indeks + 1)

        if buf_book is None:
            buf_book, buf_page = book, page
        if book != buf_book:  # kitap değişti → chunk'ı bölme
            flush()
            buf_page, buf_book = page, book

        for para in re.split(r'\n\s*\n', text):
            para = para.strip()
            if not para:
                continue
            if len(para) <= MAX_CHUNK:
                add_piece(para, page, book)
            else:  # aşırı büyük paragrafı cümlelere böl (e5 512-token sınırı için)
                for sent in re.split(r'(?<=[.!?])\s+', para):
                    sent = sent.strip()
                    if sent:
                        add_piece(sent, page, book)
    flush()
    return chunks, n_pages


def print_summary(chunks, n_pages):
    from collections import Counter, OrderedDict
    by_book = OrderedDict()
    for _, name in BOOK_RANGES:
        by_book[name] = 0
    for c in chunks:
        by_book[c["kitap"]] = by_book.get(c["kitap"], 0) + 1
    print("\n" + "=" * 62)
    print(f"📖 {ESER} — {CILT}. Cilt  |  {n_pages} sayfa  →  {len(chunks)} paragraf (chunk)")
    print("=" * 62)
    for name, cnt in by_book.items():
        if cnt:
            print(f"  {cnt:5d}  {name}")
    print("=" * 62)


def ingest(chunks):
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Apple Silicon'da MPS (Metal GPU) ile CPU'ya göre 10-30x hızlanma
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    print(f"\n[*] E5 modeli yükleniyor (intfloat/multilingual-e5-large) — cihaz: {device}…")
    model = SentenceTransformer("intfloat/multilingual-e5-large", device=device)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

    passages = [f"passage: {c['text']}" for c in chunks]
    print(f"[*] {len(passages)} paragraf toplu olarak vektörleştiriliyor…")
    vectors = model.encode(passages, batch_size=48, show_progress_bar=True, convert_to_numpy=True).tolist()

    filename = os.path.basename(PDF_PATH)
    ids, docs, metas = [], [], []
    for i, c in enumerate(chunks):
        ids.append(f"ihya_c{CILT}_chunk_{i + 1:05d}")
        docs.append(c["text"])
        metas.append({
            "title": f"{ESER} · {c['kitap']} (s.{c['page']})",
            "book": f"İhyâ ({CILT}. Cilt) · {c['kitap']}",
            "eser": ESER,
            "cilt": CILT,
            "kitap": c["kitap"],
            "page": str(c["page"]),
            "chunk_index": i + 1,
            "source": filename,
            "links": "",
        })

    print("[*] ChromaDB'ye yazılıyor (batch upsert)…")
    B = 100
    for i in range(0, len(ids), B):
        collection.upsert(
            ids=ids[i:i + B], embeddings=vectors[i:i + B],
            documents=docs[i:i + B], metadatas=metas[i:i + B],
        )
    print(f"\n[🎉] BAŞARILI! {ESER} {CILT}. Cilt yüklendi: {len(ids)} paragraf.")
    print(f"[i] Toplam koleksiyon boyutu: {collection.count()} chunk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Sadece önizleme; model yüklenmez, DB'ye yazılmaz.")
    args = ap.parse_args()

    if not os.path.exists(PDF_PATH):
        print(f"[❌] PDF bulunamadı: {PDF_PATH}")
        sys.exit(1)

    print("[*] PDF okunuyor ve yapı-farkında parçalama yapılıyor…")
    chunks, n_pages = build_chunks()
    print_summary(chunks, n_pages)

    if args.dry_run:
        print("\n[i] DRY-RUN: hiçbir şey yazılmadı. Gerçek yükleme için --dry-run olmadan çalıştırın.")
        return
    ingest(chunks)


if __name__ == "__main__":
    main()
