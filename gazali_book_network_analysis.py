import os
import re
import sys

# =====================================================================
# ISLAMICATE DH - GAZALİ KİTAPLARI OTOMATİK SEMANTİK AĞ ANALİZİ (v1.0)
# =====================================================================
# Bu betik, yerel RAG veritabanınızdaki (ChromaDB) tüm kitap verilerini
# (Kimyâ-yı Saâdet ve Eyyühe'l-Veled dahil) tarar.
# İmam Gazâlî felsefesinin en kritik 18 temel kavramının metin içindeki
# ortak geçiş (co-occurrence) sıklıklarını analiz eder.
# Grafik Teorisi kullanarak kavramların merkezilik değerlerini hesaplar
# ve Streamlit web portalınıza gömülmek üzere göz kamaştırıcı, 
# interaktif bir HTML ağ grafiği oluşturur.
# =====================================================================

def check_dependencies():
    """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder."""
    missing = []
    try:
        import chromadb
    except ImportError:
        missing.append("chromadb")
    try:
        import networkx
    except ImportError:
        missing.append("networkx")
    try:
        import pyvis
    except ImportError:
        missing.append("pyvis")
        
    if missing:
        print("\n" + "="*60)
        print("[!] EKSİK KÜTÜPHANE TESPİT EDİLDİ!")
        print("Otomatik kitap ağ analizi için şu kütüphaneler gereklidir:")
        for m in missing:
            print(f"   • {m}")
        print("\nYüklemek için terminale şu komutu yazıp Enter'a basın:")
        print(f"pip3 install {' '.join(missing)}")
        print("="*60 + "\n")
        sys.exit(1)

check_dependencies()

import chromadb
import networkx as nx
from pyvis.network import Network

class GazaliBookNetworkAnalyzer:
    def __init__(self, db_path="./gazali_chroma_db"):
        self.db_path = db_path
        self.graph = nx.Graph()
        
        # Gazâlî felsefesinin en temel 18 kavramı ve bunları yakalayacak Regex kalıpları
        self.concepts = {
            "Kalb (Gönül)": r"kalb|kalp|gönül",
            "Nefs (Nefis)": r"nefs|nefis",
            "Akıl": r"akıl|aklın|akla",
            "Ruh": r"ruh|ruhun|ruha",
            "İlim (Bilgi)": r"ilim|bilgi|ilmin",
            "Amel (Eylem)": r"amel|eylem|ibadet|davranış",
            "Dünya": r"dünya|dünyevî",
            "Ahiret": r"ahiret|kabir|haşr",
            "Marifet (İrfan)": r"marifet|arif|bilmek|tanımak",
            "Şüphe": r"şüphe|şek|tereddüt",
            "Yakîn (Kesin Bilgi)": r"yakîn|yakin|kesin bilgi",
            "Riyâzet (Nefs Terbiyesi)": r"riyâzet|riyazet|mücahede",
            "Hissiyyât (Duyular)": r"hissiyyât|hissiyyat|duyu|göz|kulak",
            "Nasihat (Öğüt)": r"nasihat|öğüt|vasiyet",
            "Saâdet (Mutluluk)": r"saâdet|saadet|mutluluk",
            "Şehvet": r"şehvet|arzu|istek",
            "Gazab (Öfke)": r"gazab|öfke|gazap",
            "Vesvese": r"vesvese|şeytan|vesveseler"
        }
        
    def run_analysis(self):
        print("[*] Gazâlî RAG Veritabanı (ChromaDB) yükleniyor...")
        if not os.path.exists(self.db_path):
            print(f"[!] Hata: '{self.db_path}' klasörü bulunamadı!")
            print("Lütfen bu betiği 'gazali_chroma_db' klasörünüzün olduğu ana dizinde çalıştırın.")
            return False
            
        try:
            # 1. ChromaDB'den verileri çekelim
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            collection = chroma_client.get_collection(name="gazali_kulliyati")
            all_data = collection.get()
            
            documents = all_data.get("documents", [])
            metadatas = all_data.get("metadatas", [])
            
            if not documents:
                print("[!] Hata: Veritabanında hiçbir paragraf bulunamadı!")
                return False
                
            print(f"[+] Veritabanındaki toplam {len(documents)} paragraf analiz ediliyor...")
            
            # Kavramların tekil frekanslarını ve ikili ortak geçiş sıklıklarını (co-occurrence) tutmak için
            concept_frequencies = {concept: 0 for concept in self.concepts}
            co_occurrence = {c1: {c2: 0 for c2 in self.concepts} for c1 in self.concepts}
            
            # Türkçe küçük harfe dönüştürme fonksiyonu (OCR pürüzlerini eritmek için)
            def normalize_text(text):
                text = text.replace('I', 'ı').replace('İ', 'i').lower()
                return text
                
            # 2. Her bir paragrafı tarayalım
            for doc in documents:
                norm_doc = normalize_text(doc)
                
                # Bu paragrafta hangi kavramlar var?
                active_concepts = []
                for concept, pattern in self.concepts.items():
                    if re.search(pattern, norm_doc):
                        active_concepts.append(concept)
                        concept_frequencies[concept] += 1
                        
                # Bulunan kavramlar arasında ikili ilişkiler kuralım (Co-occurrence)
                for i in range(len(active_concepts)):
                    for j in range(i + 1, len(active_concepts)):
                        c1, c2 = active_concepts[i], active_concepts[j]
                        co_occurrence[c1][c2] += 1
                        co_occurrence[c2][c1] += 1
                        
            # 3. Grafik (NetworkX) yapısını kuralım
            # Sadece en az 1 kez geçen kavramları düğüm olarak ekleyelim
            for concept, freq in concept_frequencies.items():
                if freq > 0:
                    self.graph.add_node(concept, frequency=freq)
                    
            # Ortak geçişleri kenar (edge) olarak ekleyelim
            min_co_occurrence_threshold = 2 # En az 2 farklı paragrafta yan yana geçmiş olmalılar
            for c1 in self.graph.nodes():
                for c2 in self.graph.nodes():
                    if c1 != c2 and self.graph.has_node(c1) and self.graph.has_node(c2):
                        weight = co_occurrence[c1][c2]
                        if weight >= min_co_occurrence_threshold:
                            # Aynı kenarın iki kez eklenmesini önleyelim
                            if not self.graph.has_edge(c1, c2):
                                self.graph.add_edge(c1, c2, weight=weight)
                                
            print(f"[+] Semantik ağ başarıyla kuruldu. Düğüm: {self.graph.number_of_nodes()}, Kenar: {self.graph.number_of_edges()}")
            return True
            
        except Exception as e:
            print(f"[!] Veritabanı taranırken beklenmedik hata oluştu: {e}")
            return False

    def calculate_centralities(self):
        """Grafik teorisi analizlerini gerçekleştirir ve rapor hazırlar."""
        print("[*] Grafik teorisi algoritmaları (Merkezilik Analizi) çalıştırılıyor...")
        
        # 1. Derece Merkeziliği (Degree Centrality): En çok kavramla ilişkili olan
        deg_centrality = nx.degree_centrality(self.graph)
        
        # 2. Arasındalık Merkeziliği (Betweenness Centrality): Kavramsal köprü rolü
        bet_centrality = nx.betweenness_centrality(self.graph)
        
        results = []
        for node in self.graph.nodes():
            results.append({
                "concept": node,
                "frequency": self.graph.nodes[node]["frequency"],
                "connections": self.graph.degree(node),
                "degree_centrality": deg_centrality[node],
                "betweenness_centrality": bet_centrality[node]
            })
            
        # Toplam frekansa göre sırala
        results.sort(key=lambda x: x["frequency"], reverse=True)
        
        # Raporlama
        print("\n" + "="*85)
        print("🕌 GAZÂLÎ KÜLLİYATI SEMANTİK AĞ ANALİZ RAPORU (KİTAP ODAKLI)")
        print("="*85)
        print(f"{'Temel Kavram':<25} | {'Frekans (Paragraf)':<18} | {'Bağlantı Derecesi':<18} | {'Köprü Gücü':<12}")
        print("-" * 85)
        for r in results[:12]:
            print(f"{r['concept']:<25} | {r['frequency']:<18} | {r['connections']:<18} | {r['betweenness_centrality']:.4f}")
        print("="*85)
        print("[i] Frekans: Kavramın tüm kitaplarda kaç farklı paragrafta (chunk) geçtiğini gösterir.")
        print("[i] Köprü Gücü: Kavramın farklı felsefi konular arasında kurduğu anlamsal bağın gücüdür.\n")
        return results

    def generate_interactive_html(self, output_path="gazali_interaktif_ag.html"):
        """Streamlit portalına gömülmek üzere muhteşem bir PyVis interaktif HTML grafiği çizer."""
        print(f"[*] Göz kamaştırıcı koyu temalı interaktif HTML üretiliyor: '{output_path}'")
        
        # Koyu temalı PyVis Network nesnesi oluşturma
        net = Network(
            notebook=False, 
            height="700px", 
            width="100%", 
            bgcolor="#121212", 
            font_color="#e0e0e0"
        )
        
        # Fizik ayarlarını akışkan hale getirelim
        net.toggle_physics(True)
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -60,
              "centralGravity": 0.01,
              "springLength": 150,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": { "iterations": 150 }
          }
        }
        """)
        
        # Merkezilik değerlerini alalım
        deg_centrality = nx.degree_centrality(self.graph)
        
        # Düğümleri (kavramları) ekleyelim
        for node in self.graph.nodes():
            freq = self.graph.nodes[node]["frequency"]
            centrality = deg_centrality[node]
            
            # Düğüm boyutu frekansına ve merkeziliğine bağlı olsun
            node_size = 15 + (centrality * 40)
            
            # Renk paleti: En popüler/merkezi kavramlar neon sarı/amber, diğerleri neon turkuaz
            if centrality > 0.6:
                node_color = "#ffc107" # Altın/Amber (En merkezi olanlar - kalb, nefs vb.)
                font_style = "16px Georgia gold bold"
            elif centrality > 0.4:
                node_color = "#00adb5" # Neon Turkuaz (Oldukça merkezi olanlar)
                font_style = "14px Georgia #00adb5 bold"
            else:
                node_color = "#007a80" # Koyu Teal (Diğerleri)
                font_style = "12px Georgia #e0e0e0"
                
            hover_title = (
                f"<b>{node}</b><br>"
                f"• Toplam Sıklık: {freq} paragraf<br>"
                f"• Ağ Bağlantısı: {self.graph.degree(node)} kavram<br>"
                f"• Anlamsal Merkezilik: {centrality:.3f}"
            )
            
            net.add_node(
                node, 
                label=node, 
                size=node_size, 
                color=node_color,
                title=hover_title,
                font={"size": 13, "face": "Georgia", "color": "#e0e0e0"}
            )
            
        # Kenarları (İlişki çizgilerini) ekleyelim
        max_weight = max([self.graph[u][v]['weight'] for u, v in self.graph.edges()]) if self.graph.edges() else 1
        
        for u, v in self.graph.edges():
            weight = self.graph[u][v]["weight"]
            # Çizgi kalınlığı ortak geçiş sıklığına göre artsın
            edge_width = 1 + (weight / max_weight) * 8
            
            # Opaklık da ilişki gücüne bağlı olsun (Daha güçlü ilişkiler daha parlak)
            opacity = 0.15 + (weight / max_weight) * 0.65
            edge_color = f"rgba(0, 173, 181, {opacity})"
            
            hover_edge_title = f"{u} ─── {v}<br>Ortak Geçiş Sıklığı: {weight} paragraf"
            
            net.add_edge(
                u, v, 
                width=edge_width, 
                color=edge_color,
                title=hover_edge_title
            )
            
        net.save_graph(output_path)
        print(f"[🎉] MUHTEŞEM! Web portalınızla %100 entegre interaktif kavram haritanız '{output_path}' başarıyla oluşturuldu!")

if __name__ == "__main__":
    analyzer = GazaliBookNetworkAnalyzer()
    if analyzer.run_analysis():
        analyzer.calculate_centralities()
        analyzer.generate_interactive_html()
