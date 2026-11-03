import os
import time
import json
import uuid
import torch
import pandas as pd
import streamlit as st
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig

# Настройка страницы Streamlit
st.set_page_config(page_title="Локальный генератор отзывов (FRED-T5)", layout="wide")

# Пути к файлам данных и конфигурации модели
CSV_FILE = "reviews.csv"
JSON_FILE = "generated_reviews.json"
BASE_MODEL_NAME = "ai-forever/FRED-T5-large"
LORA_WEIGHTS_DIR = "./lora_output"  # Папка, куда скрипт train.py сохраняет веса

# Инициализация состояний сессии
if "generated_review" not in st.session_state:
    st.session_state.generated_review = ""

@st.cache_resource
def load_model_and_tokenizer():
    """Загрузка токенизатора и модели FRED-T5 с поддержкой LoRA весов"""
    # Выбор устройства выполнения: GPU (CUDA) если доступно, иначе CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FRED-T5 использует GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_NAME, eos_token='<|endoftext|>', bos_token='<|endoftext|>')
    
    try:
        # Пытаемся загрузить модель с сохраненными LoRA адаптерами
        if os.path.exists(LORA_WEIGHTS_DIR):
            config = PeftConfig.from_pretrained(LORA_WEIGHTS_DIR)
            base_model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
            model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_DIR)
            st.info("🚀 Загружена локальная модель FRED-T5 с дообученными LoRA-адаптерами.")
        else:
            # Если папки с адаптерами нет, загружаем базовую предобученную модель
            model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
            st.warning("⚠️ Адаптеры LoRA не найдены. Загружена базовая модель FRED-T5-large.")
    except Exception as e:
        st.error(f"Ошибка при инициализации модели: {e}")
        return None, None, "cpu"
        
    model.to(device)
    model.eval()
    return model, tokenizer, device

def format_prompt(student_name, course_name, rating_text, comment, history_records):
    """Сборка структурированного шаблона входной последовательности согласно Этапу 1"""
    history_context = "Нет"
    if history_records:
        history_context = " | ".join([r.get('reviewText', '').replace('\n', ' ') for r in history_records[-2:]])
        
    # Формируем строгий промпт, к которому привыкла модель FRED-T5 при обучении
    prompt = (
        f"<lm>Студент: {student_name}. Курс: {course_name}. Оценка: {rating_text}. "
        f"Теги: {comment if comment else 'нет'.}. История: {history_context} </s>"
    )
    return prompt

def generate_review_local(model, tokenizer, device, prompt_text):
    """Локальная генерация отзыва с жесткими параметрами декодирования (Этап 2)"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            temperature=0.3,          # Низкая температура для строгого академического тона
            top_p=0.9,                # Ограничение хвоста распределения токенов
            repetition_penalty=2.5,   # Высокий штраф за циклические повторения текста
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Очистка возможных артефактов тегов разметки T5
    generated_text = generated_text.replace("<lm>", "").replace("</s>", "").strip()
    return generated_text

def load_student_history(student_name):
    """Загрузка истории отзывов по студенту из CSV"""
    if not os.path.exists(CSV_FILE):
        return []
    try:
        df = pd.read_csv(CSV_FILE, encoding="cp1251")
        df['studentName'] = df['studentName'].astype(str).str.strip()
        filtered = df[df['studentName'].str.lower() == student_name.strip().lower()]
        return filtered.to_dict(orient="records")
    except Exception as e:
        st.warning(f"Не удалось прочитать историю из CSV: {e}")
        return []

def save_confirmed_review(student_name, course_name, review_text, rating, comment):
    """Сохранение утвержденного отзыва в CSV и JSON файлы"""
    new_row = pd.DataFrame([{
        "studentName": student_name,
        "courseName": course_name,
        "reviewText": review_text,
        "rating": rating,
        "comment": comment,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE, encoding="cp1251")
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(CSV_FILE, index=False, encoding="cp1251")
        except Exception as e:
            st.error(f"Ошибка записи в CSV: {e}")
    else:
        try:
            new_row.to_csv(CSV_FILE, index=False, encoding="cp1251")
        except Exception as e:
            st.error(f"Ошибка создания CSV: {e}")

    json_data = []
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception:
            pass
            
    json_data.append({
        "studentName": student_name,
        "courseName": course_name,
        "reviewText": review_text,
        "rating": rating,
        "comment": comment,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    try:
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Ошибка записи в JSON: {e}")

# Загрузка локальных тяжелых ресурсов один раз при старте
model, tokenizer, device = load_model_and_tokenizer()

# Заголовок интерфейса
st.title("✍️ Локальный генератор отзывов на студентов")

if model is None:
    st.error("Критическая ошибка: Не удалось инициализировать нейросеть FRED-T5.")
else:
    col_form, col_history = st.columns([1.2, 0.8], gap="large")

    with col_form:
        st.subheader("Параметры для генерации")
        
        student_name = st.selectbox(
            "Студент", 
            ["Тимофеев Мирон", "Студент 1", "Студент 2", "Студент 3", "Студент 4", "Студент 5"],
            index=0
        )
        
        course_name = st.selectbox(
            "Курс",
            ["Майнкрафт", "Scratch", "Python", "Python 2", "Роблокс"],
            index=2
        )
        
        rating_options = {5: "5 — Отлично", 4: "4 — Хорошо", 3: "3 — Плохо"}
        rating = st.radio(
            "Оценка за спринт",
            options=list(rating_options.keys()),
            format_func=lambda x: rating_options[x],
            horizontal=True
        )
        
        comment = st.text_area(
            "Черновик комментария (теги преподавателя)", 
            placeholder="Например: занимался хорошо, сдал проект вовремя, нужно повторить циклы..."
        )
        
        if st.button("Сгенерировать отзыв локально", type="primary"):
            history_records = load_student_history(student_name)
            
            # 1. Форматирование строгого промпта для локальной модели
            prompt_text = format_prompt(
                student_name=student_name,
                course_name=course_name,
                rating_text=rating_options[rating],
                comment=comment,
                history_records=history_records
            )

            # 2. Локальный инференс без вызова сторонних API
            with st.spinner("Локальная модель FRED-T5 генерирует текст отзыва..."):
                try:
                    start_time = time.time()
                    st.session_state.generated_review = generate_review_local(model, tokenizer, device, prompt_text)
                    generation_time = time.time() - start_time
                    st.success(f"Готово! Время генерации: {generation_time:.2s} сек. (Устройство: {device.upper()})")
                except Exception as e:
                    st.error(f"Ошибка локального инференса: {e}")

        # Блок редактирования и сохранения результата
        if st.session_state.generated_review:
            st.markdown("---")
            st.subheader("🤖 Результат генерации")
            
            edited_review = st.text_area(
                "Редактирование черновика", 
                value=st.session_state.generated_review, 
                height=250
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Перегенерировать"):
                    st.session_state.generated_review = ""
                    st.rerun()
            with col_btn2:
                if st.button("✅ Подтвердить и сохранить", type="primary"):
                    save_confirmed_review(student_name, course_name, edited_review, rating, comment)
                    st.success("Отзыв сохранен!")
                    st.session_state.generated_review = ""
                    st.rerun()

    # Правая колонка: Просмотр истории из CSV
    with col_history:
        st.subheader("📚 Последние отзывы студента")
        history = load_student_history(student_name)
        
        if history:
            for idx, record in enumerate(reversed(history)):
                with st.expander(f"Отзыв от {record.get('timestamp', 'Неизвестно')} (Оценка: {record.get('rating', '-')})"):
                    st.caption(f"**Курс:** {record.get('courseName', '')} | **Заметки:** {record.get('comment', '')}")
                    st.write(record.get('reviewText', ''))
        else:
            st.info("История отзывов для этого студента в CSV пока отсутствует.")
