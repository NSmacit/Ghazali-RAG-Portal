# =====================================================================
# ISLAMICATE DH - GAZALİ PORTALI: ENTERPRISE DOCKERFILE (v2)
# =====================================================================
# Bu Dockerfile, Debian / trixie tabanlı slim imajlarda bulunmayan
# 'software-properties-common' paketini temizleyerek derleme hatasını çözer.
# =====================================================================

FROM python:3.12-slim

# Sistem gereksinimlerinin ve derleme araçlarının kurulması
# Debian-slim imajlarında 'software-properties-common' paketine ihtiyaç yoktur.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizininin belirlenmesi
WORKDIR /app

# Python kütüphane gereksinimlerinin kopyalanması ve yüklenmesi
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Proje dosyalarının konteynere aktarılması
COPY . .

# Streamlit varsayılan portunun dışarı açılması
EXPOSE 8501

# Konteyner sağlık kontrolü (Healthcheck)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Uygulamanın başlatılması
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
