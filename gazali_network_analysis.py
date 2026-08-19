import os
import re
import sys

# =====================================================================
# ISLAMICATE DH - GAZALI KAVRAMSAL İLİŞKİ AĞI ANALİZİ (NETWORK ANALYSIS)
# =====================================================================
# Bu betik, yerel bilgisayarınızdaki Obsidian klasörünü tarar,
# notlar arasındaki çift köşeli parantezli [[Bağlantıları]] otomatik bulur,
# grafik teorisi (Graph Theory) algoritmalarını kullanarak kavramların
# "Merkezilik" (Centrality) değerlerini hesaplar ve görselleştirir.
# =====================================================================

def check_dependencies():
    """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder."""
    missing = []
    try:
        import networkx
    except ImportError:
        missing.append("networkx")
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    if missing:
        print("\n" + "="*60)
        print("[!] EKSİK KÜTÜPHANE TESPİT EDİLDİ!")
        print("Grafik analizi ve görselleştirme için şu kütüphaneler gereklidir:")
        for m in missing:
            print(f"   • {m}")
        print("\nYüklemek için terminale şu komutu yazıp Enter'a basın:")
        print(f"pip3 install {' '.join(missing)}")
        print("="*60 + "\n")
        sys.exit(1)

check_dependencies()

import networkx as nx
import matplotlib.pyplot as plt

class GazaliNetworkAnalyzer:
    def __init__(self, vault_path="./"):
        self.vault_path = vault_path
        self.graph = nx.Graph()
        self.concepts = []
        
    def scan_vault(self):
        """Klasördeki tüm .md dosyalarını tarar ve kavramlar arası bağlantıları çıkarır."""
        print("[*] Obsidian klasörünüz taranıyor...")
        
        # 1. Önce tüm geçerli Markdown dosyalarını (kavramları) bulalım
        md_files = [f for f in os.listdir(self.vault_path) if f.endswith(".md") and f != "README.md"]
        
        if not md_files:
            print("[!] Hata: Klasörde hiç .md dosyası bulunamadı!")
            print("Lütfen bu betiği Obsidian notlarınızın olduğu klasörde çalıştırın.")
            return False
            
        self.concepts = [os.path.splitext(f)[0] for f in md_files]
        print(f"[+] Klasörde {len(self.concepts)} adet kavram belgesi tespit edildi.")
        
        # 2. Her bir dosyanın içeriğini okuyup bağlantıları ([[Link]]) çıkaralım
        # Obsidian Wikilink yapısını yakalayan Regex
        wikilink_pattern = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
        
        for file_name in md_files:
            concept_name = os.path.splitext(file_name)[0]
            self.graph.add_node(concept_name) # Düğümü ekle
            
            file_path = os.path.join(self.vault_path, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Bağlantıları bul
                links = wikilink_pattern.findall(content)
                for link in links:
                    link = link.strip()
                    # Eğer bağlantı verilen dosya da bizim kütüphanemizde varsa kenar (edge) ekle
                    if link in self.concepts and link != concept_name:
                        self.graph.add_edge(concept_name, link)
            except Exception as e:
                print(f"[!] {file_name} okunurken hata oluştu: {e}")
                
        return True

    def calculate_metrics(self):
        """Grafik teorisi analizlerini gerçekleştirir."""
        print("\n[*] Grafik teorisi analizleri hesaplanıyor...")
        
        # 1. Derece Merkeziliği (Degree Centrality): Bir kavramın doğrudan kaç bağlantısı var?
        degree_cent = nx.degree_centrality(self.graph)
        
        # 2. Arasındalık Merkeziliği (Betweenness Centrality): Kavramlar arası köprü olma gücü nedir?
        between_cent = nx.betweenness_centrality(self.graph)
        
        # 3. Yakınlık Merkeziliği (Closeness Centrality): Diğer tüm kavramlara en hızlı erişen kim?
        closeness_cent = nx.closeness_centrality(self.graph)
        
        # Sonuçları birleştirip sıralayalım
        results = []
        for node in self.graph.nodes():
            deg = self.graph.degree(node)
            results.append({
                "concept": node,
                "connections": deg,
                "degree_centrality": degree_cent[node],
                "betweenness_centrality": between_cent[node],
                "closeness_centrality": closeness_cent[node]
            })
            
        # Doğrudan bağlantı sayısına göre azalan sırada sıralayalım
        results.sort(key=lambda x: x["connections"], reverse=True)
        return results

    def print_report(self, metrics):
        """Hesaplanan metrikleri terminale şık bir tablo halinde basar."""
        print("\n" + "="*80)
        print("🕌 İMAM GAZALİ KAVRAMSAL AĞ MERKEZİLİK RAPORU (DIGITAL HUMANITIES)")
        print("="*80)
        print(f"{'Kavram Adı':<25} | {'Bağlantı':<8} | {'Derece Mer.':<12} | {'Köprü Gücü (Arasındalık)':<24}")
        print("-" * 80)
        for m in metrics[:10]: # En önemli ilk 10 kavramı gösterelim
            print(f"{m['concept']:<25} | {m['connections']:<8} | {m['degree_centrality']:.4f}     | {m['betweenness_centrality']:.4f}")
        print("="*80)
        print("[i] Derece Merkeziliği: Kavramın ağ içindeki doğrudan popülaritesini gösterir.")
        print("[i] Köprü Gücü: Kavramın farklı felsefi akımlar veya fikirler arasında kurduğu entelektüel köprüyü gösterir.")

    def visualize_network(self, output_img="gazali_network_graph.png"):
        """Görsel bir grafik haritası çizer ve bilgisayara kaydeder."""
        print(f"\n[*] Kavramsal ağ haritası oluşturuluyor ve '{output_img}' olarak kaydediliyor...")
        
        plt.figure(figsize=(14, 10))
        
        # Layout (Düğüm konumlandırma algoritması - Kamada-Kawai veya Spring)
        pos = nx.kamada_kawai_layout(self.graph)
        
        # Düğümlerin boyutlarını bağlantı sayılarına (degree) göre dinamik yapalım
        degrees = dict(self.graph.degree())
        node_sizes = [v * 350 for v in degrees.values()]
        
        # Düğüm renklerini bağlantı gücüne göre gradyan yapalım (Teal/Mavi tonları)
        node_colors = [degrees[node] for node in self.graph.nodes()]
        
        # Grafiği çizelim
        nx.draw_networkx_nodes(
            self.graph, pos, 
            node_size=node_sizes, 
            node_color=node_colors, 
            cmap=plt.cm.viridis,
            alpha=0.9
        )
        
        # Kenarları (Bağlantı çizgilerini) çizelim
        nx.draw_networkx_edges(
            self.graph, pos, 
            width=1.5, 
            edge_color="gray", 
            alpha=0.4
        )
        
        # Etiketleri (Yazıları) ekleyelim
        nx.draw_networkx_labels(
            self.graph, pos, 
            font_size=10, 
            font_weight="bold", 
            font_family="sans-serif"
        )
        
        plt.title(
            "İmam Gazali Epistemolojik Kavram Ağı Görselleştirmesi\n"
            "(Düğüm boyutu ve rengi bağlantı yoğunluğunu temsil eder)", 
            fontsize=14, fontweight="bold", pad=20
        )
        plt.axis("off") # Eksenleri gizle
        plt.tight_layout()
        
        # Kaydet
        plt.savefig(output_img, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Alternatif interaktif HTML (PyVis) denemesi
        try:
            from pyvis.network import Network
            net = Network(notebook=False, height="750px", width="100%", bgcolor="#1e1e1e", font_color="white")
            
            # networkx grafiğini pyvis formatına dönüştürelim
            for node in self.graph.nodes():
                size = degrees[node] * 5
                net.add_node(node, label=node, size=size, color="#00adb5")
            for edge in self.graph.edges():
                net.add_edge(edge[0], edge[1], color="#393e46")
                
            net.save_graph("gazali_interaktif_ag.html")
            print("[+] Muhteşem! Tarayıcınızda açıp inceleyebileceğiniz 'gazali_interaktif_ag.html' başarıyla oluşturuldu! 🌐")
        except ImportError:
            print("[i] Not: Eğer terminalde 'pip install pyvis' komutunu çalıştırırsanız,")
            print("    sistem tarayıcıda mouse ile oynatabileceğiniz interaktif bir HTML harita da üretecektir.")

if __name__ == "__main__":
    analyzer = GazaliNetworkAnalyzer()
    if analyzer.scan_vault():
        metrics = analyzer.calculate_metrics()
        analyzer.print_report(metrics)
        analyzer.visualize_network()
        print("\n[🎉] Tebrikler! Ağ analiz raporunuz ve görselleriniz hazır.")
