import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
from config.settings import CUSTOM_CSS
from src.database import load_rag_assets
from src.search import run_hybrid_search
from src.ui import generate_docx_stream
from src.ner import extract_and_display_ner

# Streamlit sayfa konfigurasyonu
st.set_page_config(
    page_title="İmam Gazâlî Dijital Beşerî Bilimler Portalı",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme uygula
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Varlıkları yükle
embed_model, collection, bm25_engine, all_data = load_rag_assets()

# Session State'leri Başlat (Tüm durumlar bellekli!)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "available_models" not in st.session_state:
    st.session_state.available_models = []
if "writer_article" not in st.session_state:
    st.session_state.writer_article = ""
if "writer_topic" not in st.session_state:
    st.session_state.writer_topic = ""
if "ner_results" not in st.session_state:
    st.session_state.ner_results = None
if "ner_topic_state" not in st.session_state:
    st.session_state.ner_topic_state = ""
if "ner_docs" not in st.session_state:
    st.session_state.ner_docs = []

# =====================================================================
# YAN PANEL (SIDEBAR) - GÜVENLİK VE AYARLAR
# =====================================================================
with st.sidebar:
    avatar_filename = "gazali_avatar_v2.jpg"
    if os.path.exists(avatar_filename):
        try:
            st.image(avatar_filename, use_container_width=True)
        except TypeError:
            st.image(avatar_filename, use_column_width=True)
    else:
        try:
            st.image("https://lh3.googleusercontent.com/notebooklm/AKYWMX-1Hvp_4i-e2m2xMO2yCZQvlu0Dq6YH3N82xKnYbzuVmBO_a5leTcjlypD_WLjym7j4ZSLDCeXSgOyurMbZkWei8w59lZ0eclF1eBJtknDRkophMYKIXkMq9X2xmBRZgYb--mb3S5GDIYZOt2hQohhT-Oct2Q", use_container_width=True)
        except TypeError:
            st.image("https://lh3.googleusercontent.com/notebooklm/AKYWMX-1Hvp_4i-e2m2xMO2yCZQvlu0Dq6YH3N82xKnYbzuVmBO_a5leTcjlypD_WLjym7j4ZSLDCeXSgOyurMbZkWei8w59lZ0eclF1eBJtknDRkophMYKIXkMq9X2xmBRZgYb--mb3S5GDIYZOt2hQohhT-Oct2Q", use_column_width=True)
            
    st.title("🕌 Gazâlî Portal")
    st.caption("Digital Humanities & RAG Platform v2.0 (Enterprise)")
    st.write("---")
    
    api_key_input = st.text_input(
        "🔑 Gemini API Key:",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Google AI Studio'dan aldığınız API anahtarı."    )
    
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        genai.configure(api_key=api_key_input)
        
        if not st.session_state.api_key_valid:
            try:
                raw_models = genai.list_models()
                models_list = []
                for m in raw_models:
                    if "generateContent" in m.supported_generation_methods:
                        m_name = m.name if m.name.startswith("models/") else f"models/{m.name}"
                        if not any(x in m_name for x in ["gemini-pro", "gemini-2.5"]):
                            models_list.append(m_name)
                
                def model_prio(name):
                    return 1 if "flash" in name.lower() else 5
                models_list.sort(key=model_prio)
                
                st.session_state.available_models = models_list
                st.session_state.selected_model = models_list[0] if models_list else None
                st.session_state.api_key_valid = True
                st.success("API Bağlantısı Başarılı! 🎉")
            except Exception as e:
                st.error(f"Bağlantı Hatası: {e}")
                st.session_state.api_key_valid = False
                
    if st.session_state.api_key_valid:
        st.selectbox(
            "🤖 Aktif Gemini Sürümü:",
            options=st.session_state.available_models,
            index=0,
            key="selected_model_box"
        )
        st.session_state.selected_model = st.session_state.selected_model_box
    
    st.write("---")
    if all_data:
        st.metric(label="📚 Toplam Paragraf Sayısı", value=len(all_data["documents"]))
        st.info("Sisteminizde kavram notları, orijinal e-kitaplar ve Tehâfüt yüklüdür.")
    else:
        st.warning("Veritabanınız boş! Lütfen önce pipeline veya ingester'ı çalıştırın.")

# =====================================================================
# ANA SAYFA VE TAB TASARIMI
# =====================================================================
st.title("🕌 İmam Gazâlî Akademik Araştırma Portalı")
st.write("Klasik İslâmî teoloji, felsefe ve epistemoloji araştırmalarını yapay zekâ ile buluşturan kurumsal bilgi yönetim paneli.")

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Akademik Sohbet (Chatbot)", 
    "✍️ Otomatik Co-Writer", 
    "📊 İnteraktif Kavram Ağ Haritası", 
    "📜 Ayet & Hadis & Şahıs Analitiği"
])

# ---------------------------------------------------------------------
# TAB 1: AKADEMİK SOHBET (CHATBOT)
# ---------------------------------------------------------------------
with tab1:
    st.subheader("🤖 Hibrit Arama Destekli Soru-Cevap Motoru")
    st.caption("Gelişmiş BM25 kelime araması ve E5-Large anlamsal aramayı birleştirerek sıfır halüsinasyonla çalışır.")
    
    if not st.session_state.api_key_valid:
        st.info("👉 Devam etmek için lütfen sol panelden geçerli bir Gemini API anahtarı girin.")
    else:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Grounding (Vektörel & BM25 Kaynak Atıfları)"):
                        for doc in message["sources"]:
                            source_header = f"📌 {doc['book']} | Sayfa: {doc['page']}" if doc['book'] else f"📌 Obsidian Notu: [[{doc['title']}]]"
                            st.markdown(f"**{source_header}** *(RRF Skoru: {doc['rrf_score']:.4f}, Vektör Sırası: {doc['v_rank']}, BM25 Sırası: {doc['b_rank']})*")
                            st.caption(f"\\\"{doc['text']}\\\"")
        
        user_query = st.chat_input("İmam Gazali felsefesi veya e-kitaplarınız hakkında sorun...")
        
        if user_query:
            st.chat_message("user").write(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            with st.spinner("📚 Yerel arşiviniz taranıyor..."):
                retrieved_docs = run_hybrid_search(embed_model, collection, bm25_engine, all_data, user_query, top_k=4)
                
            if not retrieved_docs:
                response_text = "Arşivinizde bu konuyla ilişkili hiçbir belge veya kitap bulunamadı."
                st.chat_message("assistant").write(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text, "sources": []})
            else:
                context_str = ""
                for doc in retrieved_docs:
                    source_label = f"[{doc['book']} (Sayfa {doc['page']})]" if doc['book'] else f"[[{doc['title']}]]"
                    context_str += f"\n--- KAYNAK: {source_label} ---\nİçerik: {doc['text']}\n------------------------\n"
                    
                system_instruction = """Sen İmam Gazali felsefesi ve teolojisi üzerine uzmanlaşmış, akademik dürüstlüğü en üst düzeyde tutan bir yapay zeka asistanısın.
Görevin, kullanıcının sorusuna SADECE sana sunulan 'KAYNAK NOTLAR' kapsamındaki verileri kullanarak yanıt vermektir.

Uyman Gereken Kesin Kurallar:
1. Sana verilen KAYNAK NOTLAR dışındaki hiçbir genel kültür veya internet bilgisini kullanma.
2. Eğer sorunun yanıtı sana sunulan notlarda doğrudan veya dolaylı olarak geçmiyorsa, kesinlikle bilgi UYDURMA (Halüsinasyon üretme).
3. Cevap verirken her cümlenin veya iddiadan sonra hangi belgeden/sayfadan alındığını belirt. (Örn: "...duyuların insanı yanılttığını söyler [[Hissiyyât]] veya [Eyyühel Veled (Sayfa 5)].")
4. Çelişkili veya belirsiz durumlar varsa bunları kendi yorumunu katmadan olduğu gibi aktar.
5. Tamamen Türkçe yanıt ver ve akademik, saygın bir dil kullan.
"""

                user_prompt = f"Kullanıcı Sorusu: {user_query}\n\nSana Sunulan Kaynak Notlar:\n==================================================\n{context_str}\n==================================================\n\nYukarıdaki kaynak notlara ve kurallara tamamen bağlı kalarak soruyu yanıtla:"

                try:
                    with st.spinner("🔮 Gemini anlamsal sentez gerçekleştiriyor..."):
                        model = genai.GenerativeModel(
                            model_name=st.session_state.selected_model,
                            system_instruction=system_instruction
                        )
                        response = model.generate_content(
                            contents=user_prompt,
                            generation_config=genai.types.GenerationConfig(temperature=0.1)
                        )
                        response_text = response.text
                        
                        with st.chat_message("assistant"):
                            st.markdown(response_text)
                            with st.expander("📚 Grounding (Vektörel & BM25 Kaynak Atıfları)"):
                                for doc in retrieved_docs:
                                    source_header = f"📌 {doc['book']} | Sayfa: {doc['page']}" if doc['book'] else f"📌 Obsidian Notu: [[{doc['title']}]]"
                                    st.markdown(f"**{source_header}** *(RRF Skoru: {doc['rrf_score']:.4f}, Vektör Sırası: {doc['v_rank']}, BM25 Sırası: {doc['b_rank']})*")
                                    st.caption(f"\\\"{doc['text']}\\\"")
                                    
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": retrieved_docs
                        })
                except Exception as e:
                    st.error(f"Yapay zeka cevap üretirken hata oluştu: {e}")

# ---------------------------------------------------------------------
# TAB 2: AKADEMİK CO-WRITER (YAZAR) - Bellek Korumalı!
# ---------------------------------------------------------------------
with tab2:
    st.subheader("✍️ Akıllı Makale ve Tez Taslağı Hazırlayıcı")
    st.caption("Obsidian kütüphaneniz ve yüklediğiniz e-kitaplar üzerinden yapılandırılmış, dipnotlu akademik metinler hazırlar.")
    
    if not st.session_state.api_key_valid:
        st.info("👉 Devam etmek için lütfen sol panelden geçerli bir Gemini API anahtarı girin.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            article_topic = st.text_input("📝 Makale / Tez Taslağı Konusu:", value=st.session_state.writer_topic, placeholder="Örn: Şüphe krizinden duyulara güvenin sarsılması")
        with col2:
            output_format = st.selectbox("📂 İndirme Formatı:", ["Microsoft Word (.docx)", "Markdown (.md)"])
            
        if st.button("🚀 Makaleyi Yazmaya Başla"):
            if not article_topic:
                st.warning("Lütfen yazılmasını istediğiniz konuyu girin.")
            else:
                st.session_state.writer_topic = article_topic
                with st.spinner("🔍 ChromaDB ve BM25 üzerinden ilişkili kaynaklar taranıyor..."):
                    docs = run_hybrid_search(embed_model, collection, bm25_engine, all_data, article_topic, top_k=5)
                    
                if not docs:
                    st.error("Girdiğiniz konuyla ilişkili hiçbir yerel kaynak bulunamadı.")
                else:
                    context_str = ""
                    for doc in docs:
                        label = f"[{doc['book']} (Sayfa {doc['page']})]" if doc['book'] else f"[[{doc['title']}]]"
                        context_str += f"\n--- KAYNAK: {label} ---\n{doc['text']}\n--------------------\n"
                        
                    system_instruction = """Sen İmam Gazâlî felsefesi üzerine uzmanlaşmış, akademik yayın kurallarına hakim bir asistan yazarsın.
Görevin, sana sunulan kaynakları sentezleyerek üst düzey, derinlemesine bir akademik makale taslağı hazırlamaktır.

Makale Yapısı Aynen Şu Şekilde Olmalıdır:
1. AKADEMİK BAŞLIK (İçerikle uyumlu, saygın)
2. ÖZET (Abstract) (Türkçe ve İngilizce - en az 100'er kelime)
3. GİRİŞ (Konunun önemi, ana problemi ve Gazali epistemolojisindeki yeri)
4. GELİŞME (Sana verilen kaynak notlardaki felsefi tartışmaların sentezlenerek alt başlıklarla derinleştirilmesi)
5. SONUÇ (Metinden çıkarılan ana sentez ve bulgular)
6. KAYNAKÇA (Kullandığın kaynak belgelerin ve kitapların listesi)

Kurallar:
- Metinde bilgi uydurma. Gelişme bölümündeki iddiaların sonuna kaynak referanslarını [Kitap Adı (Sayfa No)] şeklinde yaz.
- Akademik, akıcı, zengin ve kusursuz bir Türkçe kullan.
"""

                    user_prompt = f"Yazılacak Makale Konusu: {article_topic}\n\nSana Sunulan Kaynak Notlar:\n==================================================\n{context_str}\n==================================================\n\nYukarıdaki kurallara ve şablona tamamen bağlı kalarak kapsamlı akademik makale taslağını yaz:"

                    try:
                        with st.spinner("✍️ Yapay zekâ akademik taslağınızı yazıyor. Bu işlem 15-30 saniye sürebilir..."):
                            model = genai.GenerativeModel(
                                model_name=st.session_state.selected_model,
                                system_instruction=system_instruction
                            )
                            response = model.generate_content(
                                contents=user_prompt,
                                generation_config=genai.types.GenerationConfig(temperature=0.2)
                            )
                            st.session_state.writer_article = response.text
                            st.success("Makale Taslağı Başarıyla Üretildi! 🎉")
                    except Exception as e:
                        st.error(f"Yazım sırasında bir hata oluştu: {e}")
        
        # Eğer hafızada makale varsa göster (Sayfadan çıksak bile kaybolmaz!)
        if st.session_state.writer_article:
            st.write("---")
            st.markdown(st.session_state.writer_article)
            st.write("---")
            
            if "Word" in output_format:
                file_stream = generate_docx_stream(st.session_state.writer_topic, st.session_state.writer_article)
                st.download_button(
                    label="📥 Word Dosyası Olarak İndir (.docx)",
                    data=file_stream,
                    file_name=f"gazali_{st.session_state.writer_topic.replace(' ', '_').lower()}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.download_button(
                    label="📥 Markdown Dosyası Olarak İndir (.md)",
                    data=st.session_state.writer_article,
                    file_name=f"gazali_{st.session_state.writer_topic.replace(' ', '_').lower()}.md",
                    mime="text/markdown"
                )

# ---------------------------------------------------------------------
# TAB 3: İNTERAKTİF KAVRAM AĞ HARİTASI
# ---------------------------------------------------------------------
with tab3:
    st.subheader("📊 Obsidian Kavramsal İlişki Haritası")
    st.write("Obsidian notlarınız arasındaki anlamsal wikitext bağlantılarını ve ilişkileri gösteren etkileşimli haritanız:")
    
    html_path = "gazali_interaktif_ag.html"
    png_path = "gazali_network_graph.png"
    
    if os.path.exists(html_path):
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=750, scrolling=True)
            st.success("🌐 İnteraktif haritanız yüklendi! Mouse ile sürükleyip düğümleri oynatabilirsiniz.")
        except Exception as e:
            st.error(f"HTML harita yüklenirken hata oluştu: {e}")
    elif os.path.exists(png_path):
        st.image(png_path, use_column_width=True)
        st.warning("Ölçeklenebilir interaktif haritayı göremiyorsanız lütfen önce 'gazali_network_analysis.py' dosyasını çalıştırın.")
    else:
        st.warning("Henüz oluşturulmuş bir ağ görseli bulunamadı. Lütfen önce 'gazali_network_analysis.py' betiğini çalıştırın.")

# ---------------------------------------------------------------------
# TAB 4: AYET & HADİS & ŞAHIS ANALİTİĞİ (NER) - Bellek Korumalı!
# ---------------------------------------------------------------------
with tab4:
    st.subheader("📜 Yapay Zekâ Destekli Ayet, Hadis ve Şahıs Analitiği")
    st.caption("Külliyatta geçen ayetleri, hadis-i şerifleri ve felsefi/tarihi şahsiyetleri yapay zekâ ile otomatik ayıklar ve listeler.")
    
    if not st.session_state.api_key_valid:
        st.info("👉 Devam etmek için lütfen sol panelden geçerli bir Gemini API anahtarı girin.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            ner_topic = st.text_input("🔍 Taranacak Akademik Konu:", value=st.session_state.ner_topic_state, placeholder="Örn: ilim ve amel, kalbin askerleri, nefs riyazeti, yaratılış")
        with col2:
            num_docs = st.slider("Tarama Kapsamı (Paragraf Sayısı):", min_value=3, max_value=8, value=5)
            
        if st.button("🚀 Tarama ve Analizi Başlat"):
            if not ner_topic:
                st.warning("Lütfen tarama yapılacak bir konu girin.")
            else:
                st.session_state.ner_topic_state = ner_topic
                with st.spinner("🔍 Yerel kitaplar taranıyor ve ilgili paragraflar alınıyor..."):
                    docs = run_hybrid_search(embed_model, collection, bm25_engine, all_data, ner_topic, top_k=num_docs)
                    st.session_state.ner_docs = docs
                    
                if not docs:
                    st.error("Girdiğiniz konuyla ilişkili hiçbir kaynak bulunamadı.")
                else:
                    st.info(f"📚 Konuyla en ilişkili {len(docs)} paragraf başarıyla çekildi. Yapay zekâ ayıklama (NER) motoru çalıştırılıyor...")
                    
                    with st.spinner("🔮 Yapay zekâ anlamsal varlık ayıklama (NER) gerçekleştiriyor..."):
                        ner_data, success, err_msg = extract_and_display_ner(docs)
                        if success:
                            st.session_state.ner_results = ner_data
                            st.success("Analiz Başarıyla Tamamlandı! 🎉")
                        else:
                            st.error(f"NER analizi sırasında bir hata oluştu: {err_msg}")
                            
        # Hafızada analiz sonucu varsa göster! (Sayfadan çıksak bile kaybolmaz!)
        if st.session_state.ner_results:
            data = st.session_state.ner_results
            
            # 1. AYETLER
            st.markdown("### 📖 Ayet-i Kerîmeler")
            ayet_list = data.get("ayetler", [])
            if ayet_list:
                df_ayet = pd.DataFrame(ayet_list)
                df_ayet.columns = ["Sure / Ayet No", "Ayet Meali / Metni", "Geçtiği Kaynak"]
                st.dataframe(df_ayet, use_container_width=True)
            else:
                st.info("Taranan paragraflarda doğrudan ayet-i kerîme atfı tespit edilemedi.")
                
            # 2. HADİSLER
            st.markdown("### 💬 Hadîs-i Şerîfler")
            hadis_list = data.get("hadisler", [])
            if hadis_list:
                df_hadis = pd.DataFrame(hadis_list)
                df_hadis.columns = ["Hadis-i Şerif Metni", "Ravi / Kaynak", "Geçtiği Kaynak"]
                st.dataframe(df_hadis, use_container_width=True)
            else:
                st.info("Taranan paragraflarda doğrudan hadîs-i şerîf atfı tespit edilemedi.")
                
            # 3. ŞAHISLAR
            st.markdown("### 👤 Tarihi Şahsiyetler & Alimler")
            sahis_list = data.get("sahislar", [])
            if sahis_list:
                df_sahis = pd.DataFrame(sahis_list)
                df_sahis.columns = ["İsim", "Metindeki Rolü / Bağlam", "Geçtiği Kaynak"]
                st.dataframe(df_sahis, use_container_width=True)
            else:
                st.info("Taranan paragraflarda özel şahıs/alim atfı tespit edilemedi.")
                
            # Kaynak Metinler
            if st.session_state.ner_docs:
                with st.expander("📝 Taranan Paragrafların Ham Metinleri"):
                    for d in st.session_state.ner_docs:
                        lbl = f"📌 {d['book']} | Sayfa: {d['page']}" if d['book'] else f"📌 Obsidian: [[{d['title']}]]"
                        st.markdown(f"**{lbl}**")
                        st.caption(f'"{d["text"]}"')
