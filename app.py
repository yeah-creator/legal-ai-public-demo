# --------------------------------------------------------------------------
# --- app.py (Версия 4.0: HyDE + Расширение + Переранжирование) ---
# --- Это самая продвинутая версия нашего поискового движка ---
# --------------------------------------------------------------------------

import streamlit as st
import faiss
import json
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- НАСТРОЙКИ ---
FAISS_INDEX_PATH = "document_index.faiss"
CHUNK_DATA_PATH = "chunk_data.json"
BI_ENCODER_MODEL = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- ИНИЦИАЛИЗАЦИЯ GEMINI ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    GEMINI_AVAILABLE = True
except Exception as e:
    st.error(f"Не удалось настроить Gemini API. Проверьте ваш секретный ключ в файле .streamlit/secrets.toml. Ошибка: {e}")
    GEMINI_AVAILABLE = False

# --- ЗАГРУЗКА РЕСУРСОВ ---
@st.cache_resource
def load_resources():
    """Загружает все необходимые ресурсы: модели, индекс и данные чанков."""
    print("Загрузка ресурсов для поиска...")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNK_DATA_PATH, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    print("Ресурсы успешно загружены.")
    return bi_encoder, cross_encoder, index, chunks_data

bi_encoder, cross_encoder, index, chunks_data = load_resources()

# --- ФУНКЦИИ ПРИЛОЖЕНИЯ ---

def generate_hypothetical_answer(query):
    """
    ✅ НОВАЯ ФУНКЦИЯ (HyDE): Генерирует гипотетический ответ на запрос для улучшения поиска.
    """
    if not GEMINI_AVAILABLE:
        st.warning("Gemini недоступен. Поиск будет выполнен по исходному запросу.")
        return query
    
    try:
        # Промпт, который просит модель сгенерировать идеальный ответ
        prompt = f"""
        Представь, что ты юрист-эксперт. В ответ на следующий вопрос пользователя, напиши один короткий, 
        идеальный абзац, который мог бы содержаться в мотивировочной части судебного решения и 
        который бы полностью отвечал на этот вопрос. Абзац должен быть насыщенным и по делу. 
        Не пиши ничего, кроме этого гипотетического абзаца.

        Вопрос пользователя: "{query}"
        """
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Не удалось сгенерировать гипотетический ответ. Поиск будет выполнен по исходному запросу. Ошибка: {e}")
        return query


def search(search_text, k=30):
    """Первичный поиск кандидатов в FAISS по предоставленному тексту."""
    query_vector = bi_encoder.encode([search_text], normalize_embeddings=True)
    query_vector_np = np.array(query_vector).astype('float32')
    distances, indices = index.search(query_vector_np, k)
    
    unique_parent_texts = set()
    results = []
    for i in indices[0]:
        # Проверяем, что индекс в пределах данных
        if i < len(chunks_data):
            parent_text = chunks_data[i]['parent_text']
            if parent_text not in unique_parent_texts:
                results.append(chunks_data[i])
                unique_parent_texts.add(parent_text)
    return results

def rerank(original_query, search_results, top_k=5):
    """Переранжирование результатов с помощью Cross-Encoder по ОРИГИНАЛЬНОМУ запросу."""
    if not search_results:
        return []
    # Важно: сравниваем найденные тексты с оригинальным запросом, чтобы не потерять фокус
    pairs = [[original_query, result['parent_text']] for result in search_results]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    
    for i in range(len(scores)):
        search_results[i]['rerank_score'] = scores[i]
        
    reranked_results = sorted(search_results, key=lambda x: x['rerank_score'], reverse=True)
    return reranked_results[:top_k]

# --- ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС ---

st.title("⚖️ Демо 4.0: Поиск с помощью HyDE")
st.write("Теперь система сначала просит Gemini написать 'идеальный' ответ на ваш вопрос, а затем ищет в базе документы, похожие на этот идеальный ответ.")

user_query = st.text_input("Задайте ваш вопрос:", "удорожание договора")

if st.button("Найти ответ"):
    if user_query:
        # Этап 1: Генерация гипотетического ответа (HyDE)
        with st.spinner("Этап 1: Прошу Gemini помочь сформулировать идеальный ответ..."):
            hypothetical_answer = generate_hypothetical_answer(user_query)
        
        st.info(f"**Гипотетический ответ для поиска:**\n\n{hypothetical_answer}")

        # Этап 2: Быстрый поиск по гипотетическому ответу
        with st.spinner("Этап 2: Ищу документы, похожие на 'идеальный ответ'..."):
            initial_results = search(hypothetical_answer)

        if initial_results:
            # Этап 3: Переранжирование по оригинальному запросу
            with st.spinner("Этап 3: 'Эксперт' выбирает 5 лучших, сверяясь с вашим исходным вопросом..."):
                final_results = rerank(user_query, initial_results)

            st.subheader("🏆 Главные результаты (после всех улучшений):")
            if final_results:
                for i, result in enumerate(final_results):
                    st.markdown(f"---")
                    st.write(f"**Результат #{i+1} (из файла: `{result['doc_name']}`)**")
                    st.info(f"**Оценка релевантности:** {result['rerank_score']:.2f}")
                    st.success(result['parent_text'])
            else:
                 st.error("После всех фильтров не осталось релевантных результатов.")
        else:
            st.warning("Не удалось найти релевантные фрагменты по вашему запросу.")
    else:
        st.error("Пожалуйста, введите вопрос.")