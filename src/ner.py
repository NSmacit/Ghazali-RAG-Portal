import json
import streamlit as st
import google.generativeai as genai

SYSTEM_INSTRUCTION_NER = """Sen klasik İslami metinler ve teoloji üzerinde uzmanlaşmış, son derece hassas bir metin analitiği (Named Entity Recognition) asistanısın.
Görevin, sana sunulan kaynak metinleri analiz ederek içindeki:
1. AYETLERİ (Eğer varsa; sure adı, ayet numarası, Türkçe meali ve geçtiği kaynak ile birlikte)
2. HADİSLERİ (Eğer varsa; hadisin içeriği, ravisi veya kaynağı, geçtiği yer ile birlikte)
3. ŞAHISLARI (Metinde adı geçen tarihi alimler, filozoflar, peygamberler, sahabeler veya şahsiyetler)
bulup ayıklamaktır.

Çıktıyı KESİNLİKLE ama KESİNLİKLE sadece aşağıdaki JSON şablonuna göre üretmelisin. JSON dışında hiçbir giriş, açıklama, markdown kodu veya düz metin yazmamalısın. Çıktı doğrudan geçerli bir JSON olmalıdır:
{
  "ayetler": [
    {"sure_ayet": "Sure Adı ve Ayet No", "metin": "Ayeti kerimenin meali veya içeriği", "kaynak": "Geçtiği kitap/not ve sayfa"}
  ],
  "hadisler": [
    {"metin": "Hadis-i şerifin meali veya içeriği", "ravi_kaynak": "Zikredilen ravi veya hadis kaynağı", "kaynak": "Geçtiği kitap/not ve sayfa"}
  ],
  "sahislar": [
    {"isim": "Kişinin Adı", "rol_baglam": "Metindeki rolü veya hangi bağlamda zikredildiği", "kaynak": "Geçtiği kitap/not ve sayfa"}
  ]
}

Eğer metinde ayet, hadis veya şahıs yoksa, ilgili listeyi boş bırak: []
"""

def extract_and_display_ner(docs):
    """Paragraflardan Ayet, Hadis ve Şahıs varlıklarını Gemini kullanarak ayıklar, JSON formatında döndürür."""
    context_str = ""
    for doc in docs:
        label = f"[{doc['book']} (Sayfa {doc['page']})]" if doc['book'] else f"[[{doc['title']}]]"
        context_str += f"\n--- KAYNAK: {label} ---\n{doc['text']}\n--------------------\n"
        
    user_prompt = f"Analiz Edilecek Kaynak Metinler:\n==================================================\n{context_str}\n==================================================\n\nYukarıdaki kaynakları tara ve JSON formatında ayıklama sonuçlarını döndür:"
    
    try:
        model = genai.GenerativeModel(
            model_name=st.session_state.selected_model,
            system_instruction=SYSTEM_INSTRUCTION_NER
        )
        response = model.generate_content(
            contents=user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        raw_json = response.text.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json.split("```json", 1)[1]
        if raw_json.endswith("```"):
            raw_json = raw_json.rsplit("```", 1)[0]
        raw_json = raw_json.strip()
        
        data = json.loads(raw_json)
        return data, True, ""
    except Exception as e:
        return None, False, str(e)
