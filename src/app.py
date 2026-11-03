import os
import time
import json
import torch
import pandas as pd
import streamlit as st
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig

# Настройка страницы Streamlit
st.set_page_config(page_title="Локальный генератор отзывов (FRED-T5)", layout="wide")

# Константы путей и структуры данных
CSV_FILE = "reviews.csv"
JSON_FILE = "generated_reviews.json"
BASE_MODEL_NAME = "ai-forever/FRED-T5-large"
LORA_WEIGHTS_DIR = "./lora_output"

# Проверка наличия и автоматическое создание CSV-файла при старте приложения
if not os.path.exists(CSV_FILE):
    # Создаем пустой файл строго с теми колонками, которые вы оставили
    df_init = pd.DataFrame(columns=["studentName", "courseName", "total_hours", "reviewText", "timestamp"])
    df_init.to_csv(CSV_FILE, index=False, encoding="cp1251")

# Инициализация состояний сессии
if "generated_review" not in st.session_state:
    st.session_state.generated_review = ""

@st.cache_resource
def load_model_and_tokenizer():
    """Загрузка токенизатора и модели FRED-T5 с поддержкой LoRA весов"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_NAME, eos_token='<|endoftext|>', bos_token='<|endoftext|>')
    
    try:
        if os.path.exists(LORA_WEIGHTS_DIR):
            config = PeftConfig.from_pretrained(LORA_WEIGHTS_DIR)
            base_model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
            model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_DIR)
            st.info("🚀 Загружена локальная модель FRED-T5 с дообученными LoRA-адаптерами.")
        else:
            model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
            st.warning("⚠️ Адаптеры LoRA не найдены. Загружена базовая модель FRED-T5-large.")
    except Exception as e:
        st.error(f"Ошибка при инициализации модели: {e}")
        return None, None, "cpu"
        
    model.to(device)
    model.eval()
    return model, tokenizer, device

def format_prompt(student_name, course_name, total_hours):
    """Сборка структурированного шаблона входной последовательности под новый датасет"""
    prompt = f"<lm>Студент: {student_name}. Курс: {course_name}. Прогресс: {total_hours} ч. </s>"
    return prompt

def generate_review_local(model, tokenizer, device, prompt_text):
    """Локальная генерация отзыва по часам и курсу"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=2.5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs, skip_special_tokens=True)
    generated_text = generated_text.replace("<lm>", "").replace("</s>", "").strip()
    return generated_text

def load_student_history(student_name):
    """Загрузка истории отзывов по студенту из CSV"""
    try:
        df = pd.read_csv(CSV_FILE, encoding="cp1251")
        df['studentName'] = df['studentName'].astype(str).str.strip()
        filtered = df[df['studentName'].str.lower() == student_name.strip().lower()]
        return filtered.to_dict(orient="records")
    except Exception as e:
        st.warning(f"Не удалось прочитать историю из CSV: {e}")
        return []

def save_confirmed_review(student_name, course_name, total_hours, review_text):
    """Сохранение утвержденного отзыва в файлы"""
    new_row = pd.DataFrame([{
        "studentName": student_name,
        "courseName": course_name,
        "total_hours": total_hours,
        "reviewText": review_text,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    try:
        df = pd.read_csv(CSV_FILE, encoding="cp1251")
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_FILE, index=False, encoding="cp1251")
    except Exception as e:
        st.error(f"Ошибка записи в CSV: {e}")

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
        "total_hours": total_hours,
        "reviewText": review_text,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    try:
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Ошибка записи в JSON: {e}")

# Инициализация тяжелых ресурсов
model, tokenizer, device = load_model_and_tokenizer()

st.title("✍️ Локальный генератор отзывов на студентов")

if model is None:
    st.error("Критическая ошибка: Не удалось инициализировать нейросеть FRED-T5.")
else:
    col_form, col_history = st.columns([1.2, 0.8], gap="large")

    with col_form:
        st.subheader("Параметры для генерации")
        
        student_name = st.selectbox(
            "Студент", 
            ["Тимофеев Мирон", "Лебедев Ярослав", "Петрова Виктория", "Новикова Екатерина", "Смирнова Анастасия"],
            index=0
        )
        
        course_name = st.selectbox(
            "Курс",
            ["Python Ур.1", "Scratch", "3D", "Роблокс"],
            index=1
        )
        
        # Новый числовой инпут вместо текстовых комментариев
        total_hours = st.number_input(
            "Общий прогресс по курсу (в часах)", 
            min_value=1, 
            max_value=300, 
            value=16, 
            step=1
        )
        
        if st.button("Сгенерировать отзыв локально", type="primary"):
            prompt_text = format_prompt(
                student_name=student_name,
                course_name=course_name,
                total_hours=total_hours
            )

            with st.spinner("Локальная модель FRED-T5 генерирует текст отзыва..."):
                try:
                    start_time = time.time()
                    st.session_state.generated_review = generate_review_local(model, tokenizer, device, prompt_text)
                    generation_time = time.time() - start_time
                    st.success(f"Готово! Время генерации: {generation_time:.2f} сек.")
                except Exception as e:
                    st.error(f"Ошибка локального инференса: {e}")

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
                    save_confirmed_review(student_name, course_name, total_hours, edited_review)
                    st.success("Отзыв сохранен!")
                    st.session_state.generated_review = ""
                    st.rerun()

    with col_history:
        st.subheader("📚 Последние отзывы студента")
        history = load_student_history(student_name)
        
        if history:
            for record in reversed(history):
                with st.expander(f"Отзыв от {record.get('timestamp', 'Неизвестно')}"):
                    st.caption(f"**Курс:** {record.get('courseName', '')} | **Пройдено часов:** {record.get('total_hours', '0')}")
                    st.write(record.get('reviewText', ''))
        else:
            st.info("История отзывов для этого студента в CSV пока отсутствует.")
