# =====================================================================
# GAZALİ ENTERPRISE SYSTEM CONFIGURATION
# =====================================================================

CUSTOM_CSS = """
<style>
    .reportview-container {
        background: #121212;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #00adb5 !important;
        font-family: 'Georgia', serif;
    }
    .stButton>button {
        background-color: #00adb5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #007a80;
        border: none;
        box-shadow: 0 4px 12px rgba(0,173,181,0.3);
    }
    .source-box {
        background-color: #1e1e1e;
        border-left: 4px solid #00adb5;
        padding: 12px;
        margin: 8px 0px;
        border-radius: 4px;
    }
</style>
"""

ARABIC_TO_TURKISH_DICT = {
    "العلم": "ilim bilgi muallim taallüm",
    "العمل": "amel eylem pratik ibadet",
    "القلب": "kalp gönül cevher tasfiye",
    "النفس": "nefis nefs ruh kendini benlik",
    "العقل": "akıl rasyonel düşünce fehim",
    "السعادة": "saadet mutluluk kurtuluş necât",
    "الشك": "şüphe tereddüt şüphecilik istidlal",
    "اليقين": "yakin yakinî kesin bilgi hakikat",
    "الحس": "hissiyyat duyular his duyu organları",
    "الرياضة": "riyazet nefs terbiyesi tefekkür",
    "الوسوسة": "vesvese kuruntu şeytan vesvesesi",
    "معرفة": "marifet bilmek tanımak irfan",
    "معرفة النفس": "kendini bilmek marifet-i nefs nefsini bilmek",
    "معرفة الله": "Allah'ı bilmek marifetullah",
    "الدنيا": "dünya hayatı fani geçici alem",
    "الآخرة": "ahiret beka alemi ebediyet",
    "الولد": "veled çocuk oğul ey oğul",
    "نصيحة": "nasihat öğüt vasiyet",
    "تصوف": "tasavvuf ahlak zühd takva",
    "طهارة": "taharet temizlik kalb tasfiyesi",
    "عشق": "aşk muhabbet sevgi",
    "نور": "nur ışık ilahi aydınlanma"
}
