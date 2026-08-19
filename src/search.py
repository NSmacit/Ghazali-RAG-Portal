import re
import streamlit as st
import google.generativeai as genai
from config.settings import ARABIC_TO_TURKISH_DICT

def translate_expand_query(query):
    """Sorguda Arapça karakterler varsa yerel sözlük ve Gemini yardımıyla Türkçe genişletme yapar."""
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
    if not is_arabic:
        return query, False, ""
        
    clean_q = re.sub(r'[^\w\s\u0600-\u06FF]', '', query).strip()
    translated = ARABIC_TO_TURKISH_DICT.get(clean_q, "")
    
    if not translated:
        words = clean_q.split()
        tr_words = []
        for w in words:
            tr_w = ARABIC_TO_TURKISH_DICT.get(w, "")
            if tr_w:
                tr_words.append(tr_w)
        if tr_words:
            translated = " ".join(tr_words)
            
    dynamic_translation = ""
    if st.session_state.get("api_key_valid", False):
        try:
            model = genai.GenerativeModel(st.session_state.get("selected_model"))
            prompt = f"Translate the following classical Arabic Islamic/philosophical term or question into clean Turkish search keywords for a book database. Return ONLY the translated Turkish keywords, no explanations:\n{query}"
            response = model.generate_content(prompt)
            dynamic_translation = response.text.strip().replace("\n", " ")
        except Exception:
            pass
            
    combined_translation = translated
    if dynamic_translation:
        if combined_translation:
            combined_translation += " " + dynamic_translation
        else:
            combined_translation = dynamic_translation
            
    if not combined_translation:
        combined_translation = query
        
    return combined_translation, True, clean_q

def run_hybrid_search(embed_model, collection, bm25_engine, all_data, query, top_k=4):
    """BM25 ve Vektör (E5) sıralamalarını RRF formülüyle sentezler. Arapça sorgularda çapraz dilli arama yapar."""
    if not collection or not bm25_engine or not all_data:
        return []
        
    bm25_query = query
    vector_query = query
    
    translated_q, is_arabic, original_arabic = translate_expand_query(query)
    if is_arabic:
        bm25_query = translated_q
        vector_query = f"{query} {translated_q}"
        st.info(f"🌐 **Arapça Arama Tespit Edildi:**\n• Orijinal Arapça: `{query}`\n• Türkçe Genişletme: `{translated_q}`\n\n*Çapraz Dilli (Cross-lingual) Hibrit motorumuz, hem orijinal terimi vektör uzayında tarar hem de otomatik çeviri üzerinden yerel BM25 indekslemesi gerçekleştirir.*")
        
    bm25_scores = bm25_engine.get_scores(bm25_query)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda k: bm25_scores[k], reverse=True)
    
    formatted_query = f"query: {vector_query}"
    query_vector = embed_model.encode(formatted_query).tolist()
    vector_results = collection.query(
        query_embeddings=[query_vector],
        n_results=len(all_data["documents"])
    )
    
    vector_order = vector_results["ids"][0]
    vector_ranked = [all_data["ids"].index(vid) for vid in vector_order]
    
    rrf_scores = {}
    k_constant = 60
    
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_constant + rank + 1))
        
    for rank, idx in enumerate(vector_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_constant + rank + 1))
        
    top_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    
    final_docs = []
    for idx in top_indices:
        metadata = all_data["metadatas"][idx]
        text = all_data["documents"][idx]
        
        v_rank = vector_ranked.index(idx) + 1 if idx in vector_ranked else "N/A"
        b_rank = bm25_ranked.index(idx) + 1
        
        final_docs.append({
            "text": text,
            "title": metadata.get("title", "Bilinmeyen"),
            "book": metadata.get("book", ""),
            "page": metadata.get("page", ""),
            "links": metadata.get("links", ""),
            "rrf_score": rrf_scores[idx],
            "v_rank": v_rank,
            "b_rank": b_rank
        })
        
    return final_docs
