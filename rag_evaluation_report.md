# 🛡️ Gazâlî RAG Sistemi Kalite ve Performans Raporu (RAG Evaluation Report)

Bu rapor, **İslâmî Beşerî Bilimler & Yapay Zekâ Portalı** bünyesinde çalışan yerel **Hybrid Search (BM25 + Multilingual-E5-Large)** arama motorunun retrieval hassasiyetini ve performans metriklerini doğrulamak amacıyla otomatik olarak üretilmiştir.

## 📊 Özet Değerlendirme Metrikleri

| Metrik | Değer | Durum |
| :--- | :--- | :--- |
| **Toplam Paragraf Kapsamı** | 3883 Paragraf (Chunk) | ✅ Enterprise |
| **Genel Test Başarı Oranı** | %100.0 (4/4) | ✅ MÜKEMMEL |
| **Ortalama Sorgu Gecikmesi** | 543.9 ms | ⚠️ İYİLEŞTİRİLMELİ |
| **Sözlük Genişletme Kalitesi** | %100 Çapraz Dilli Entegrasyon | ✅ Aktif |

---

## 🔍 Detaylı Test Senaryoları Raporu

### [TC-001] Nefs ve Kalp İlişkisi (Tasavvuf Epistemolojisi)
* **Sorgu Terimi:** `kalbin hakikati ve marifetü'n-nefs`
* **Test Durumu:** **✅ BAŞARILI (PASSED)**
* **Sorgu Gecikmesi (Latency):** `1168.68 ms` (Eşik Değeri < 800ms: ❌)
* **Yakalanan Anahtar Terimler:** `2` adet terim yakalandı. (nefs, ruh)
* **Beklenen Eser Eşleşmesi:** ✅ Başarılı

---
### [TC-002] Arapça-Türkçe Çapraz Dilli Arama (Cross-Lingual)
* **Sorgu Terimi:** `الشك`
* **Test Durumu:** **✅ BAŞARILI (PASSED)**
* **Sorgu Gecikmesi (Latency):** `422.78 ms` (Eşik Değeri < 800ms: ✅)
* **Yakalanan Anahtar Terimler:** `2` adet terim yakalandı. (şüphe, akıl)
* **Beklenen Eser Eşleşmesi:** ✅ Başarılı

---
### [TC-003] Eyyühe'l-Veled (Nasihat ve Amel)
* **Sorgu Terimi:** `ey oğul ilim ve amel`
* **Test Durumu:** **✅ BAŞARILI (PASSED)**
* **Sorgu Gecikmesi (Latency):** `304.16 ms` (Eşik Değeri < 800ms: ✅)
* **Yakalanan Anahtar Terimler:** `3` adet terim yakalandı. (oğul, ilim, amel)
* **Beklenen Eser Eşleşmesi:** ✅ Başarılı

---
### [TC-004] Filozofların Tutarsızlığı (Yeni Entegre Edilen Eser)
* **Sorgu Terimi:** `filozofların tutarsızlığı ve metafizik iddialar`
* **Test Durumu:** **✅ BAŞARILI (PASSED)**
* **Sorgu Gecikmesi (Latency):** `279.91 ms` (Eşik Değeri < 800ms: ✅)
* **Yakalanan Anahtar Terimler:** `3` adet terim yakalandı. (filozof, tutarsızlık, tehafüt)
* **Beklenen Eser Eşleşmesi:** ✅ Başarılı

---

## 🔬 Metodoloji ve Teknik Değerlendirme

1. **Hibrit Arama (Dense + Sparse Sentezi):** 
   Sistem, kelime eşleşmelerini yakalamak için **BM25 (Sparse)** ve anlamsal ilişkileri kurmak için **intfloat/multilingual-e5-large (Dense)** modelini kullanır. Skorlar **Reciprocal Rank Fusion (RRF)** formülü ile birleştirilir.
   
2. **Çapraz Dilli Arama (Cross-Lingual Retrieval):**
   Arapça aramalarda sistem hem yerel sözlük genişletmesi hem de Gemini destekli dinamik terim eşleştirmesi yaparak arama uzayını genişletir ve Türkçe kitaplardan da doğru paragrafları çeker.

3. **Mükerrerlik ve Güvenlik:**
   Tüm veritabanı sorguları yerel diskte (`gazali_chroma_db`) izole çalıştırılır. API anahtarları `.env` kılıfıyla korunmakta olup, kodlar **AGPL-3.0** lisansı ile IP hırsızlığına karşı mühürlenmiştir.
