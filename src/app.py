import os
import time
import json
import torch
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig

#### Настройка страницы Streamlit

st.set_page_config(page_title="Генератор отзывов", layout="wide")

CSV_FILE = "reviews.csv"
JSON_FILE = "generated_reviews.json"
BASE_MODEL_NAME = "ai-forever/FRED-T5-large"
LORA_WEIGHTS_DIR = "./lora_output"

if not os.path.exists(CSV_FILE):
    df_init = pd.DataFrame(columns=["studentName", "courseName", "total_hours", "reviewText", "timestamp"])
    df_init.to_csv(CSV_FILE, index=False, encoding="cp1251")

if "generated_review" not in st.session_state:
    st.session_state.generated_review = ""

@st.cache_resource
def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_NAME, eos_token='<|endoftext|>', bos_token='<|endoftext|>')
        if os.path.exists(LORA_WEIGHTS_DIR):
            config = PeftConfig.from_pretrained(LORA_WEIGHTS_DIR)
            base_model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
            model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_DIR)
        else:
            model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    except Exception as e:
        return None, None, "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

def load_student_history(student_name):
    try:
        if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
            return []
        df = pd.read_csv(CSV_FILE, encoding="cp1251")
        df['studentName'] = df['studentName'].astype(str).str.strip()
        filtered = df[df['studentName'].str.lower() == student_name.strip().lower()]
        return filtered.to_dict(orient="records")
    except Exception:
        return []

def format_prompt(student_name, course_name, total_hours, tags_str):
    history_records = load_student_history(student_name)
    history_text = history_records[-1].get('reviewText', 'Нет предыдущих отзывов') if history_records else "Нет предыдущих отзывов"

    prompt = (
        f"<lm>Студент: {student_name}. "
        f"Курс: {course_name}. "
        f"Прогресс: {total_hours} ч. "
        f"История: {history_text}. "
        f"Теги: {tags_str}. "
        f"Отзыв:</s>"
    )
    return prompt

def generate_review_local(model, tokenizer, device, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            temperature=0.4,
            top_p=0.85,
            repetition_penalty=2.5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()

model, tokenizer, device = load_model_and_tokenizer()

st.title("Написать отзыв")

col_ui, col_preview = st.columns([1, 1], gap="large")

with col_ui:
    #### Верстка оригинального модального окна на HTML/CSS с встраиванием скрипта Streamlit
    html_form = """
    <div style="font-family: sans-serif; background: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd;">
        <h3>Параметры отзыва</h3>
        <form id="reviewForm">
            <label>Имя студента:</label><br>
            <input type="text" id="studentName" name="studentName" style="width:100%; padding:8px; margin: 8px 0;" required><br>
            <label>Название курса:</label><br>
            <input type="text" id="courseName" name="courseName" style="width:100%; padding:8px; margin: 8px 0;" required><br>
            <label>Теги/Комментарии преподавателя:</label><br>
            <textarea id="teacherTags" name="teacherTags" style="width:100%; height:80px; padding:8px; margin: 8px 0;"></textarea><br>
            <button type="button" onclick="submitForm()" style="background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer;">Сгенерировать</button>
        </form>
    </div>

    <script>
        function submitForm() {
            const data = {
                isSubmitted: true,
                studentName: document.getElementById('studentName').value,
                courseName: document.getElementById('courseName').value,
                teacherTags: document.getElementById('teacherTags').value
            };
            // Взаимодействие со Streamlit (передача данных в форму)
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: data
            }, '*');
        }
    </script>
    """

    # Рендеринг HTML-окна как интерактивного компонента
    form_result = components.html(html_form, height=450, scroller=False)

with col_preview:
    st.subheader("Генерация...")

    #### Обработка данных формы, полученных от HTML-компонента
    if form_result and isinstance(form_result, dict) and form_result.get("isSubmitted"):
        s_name = form_result.get("studentName")
        c_name = form_result.get("courseName")
        t_tags = form_result.get("teacherTags", "успешно продолжает обучение")

        st.info(f"Получены данные формы для студента: **{s_name}**")

        # Симулируем расчет прогресса на основе истории или базового значения спринта
        calculated_hours = 16

        prompt_text = format_prompt(
            student_name=s_name,
            course_name=c_name,
            total_hours=calculated_hours,
            tags_str=t_tags
        )

        if model is None:
            st.warning("Модель находится в режиме симуляции интерфейса.")
            st.session_state.generated_review = f"Черновик развернутого отзыва для родителя {s_name} по курсу {c_name} на основе комментариев: {t_tags}."
        else:
            with st.spinner("Формируем идеальный текст по методическим правилам..."):
                try:
                    st.session_state.generated_review = generate_review_local(model, tokenizer, device, prompt_text)
                except Exception as e:
                    st.error(f"Ошибка инференса: {e}")

    if st.session_state.generated_review:
        st.markdown("---")
        st.write("Сгенерированный академический отзыв:")
        final_text = st.text_area("Вы можете отредактировать текст перед отправкой:", value=st.session_state.generated_review, height=200)

        if st.button("Отправить", type="primary"):
            # Логика сохранения
            if os.path.exists(CSV_FILE):
                try:
                    df = pd.read_csv(CSV_FILE, encoding="cp1251")
                except Exception:
                    df = pd.DataFrame(columns=["studentName", "courseName", "total_hours", "reviewText", "timestamp"])
            else:
                df = pd.DataFrame(columns=["studentName", "courseName", "total_hours", "reviewText", "timestamp"])
            
            new_row = {
                "studentName": form_result.get("studentName") if form_result else "Неизвестно",
                "courseName": form_result.get("courseName") if form_result else "Неизвестно",
                "total_hours": 16,
                "reviewText": final_text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(CSV_FILE, index=False, encoding="cp1251")
            
            st.success("Отзыв успешно сохранён!")
            st.session_state.generated_review = ""
