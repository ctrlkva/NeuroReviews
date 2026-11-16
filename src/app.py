import os
import torch
import streamlit as st
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Инициализация путей
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "data" / "history_teacher.csv"
MODEL_PATH = BASE_DIR.parent / "models" / "FRED-T5-large"

@st.cache_resource
def load_model_and_tokenizer():
    if MODEL_PATH.exists():
        model_path_str = str(MODEL_PATH.resolve().as_posix())
        tokenizer = AutoTokenizer.from_pretrained(model_path_str, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path_str,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=True
        ).to(device)
        return model, tokenizer, device
    return None, None, "cpu"

def load_data():
    if CSV_PATH.exists():
        try:
            return pd.read_csv(CSV_PATH)
        except Exception:
            return pd.DataFrame(columns=["studentName", "courseName", "rating", "comment", "review_text"])
    return pd.DataFrame(columns=["studentName", "courseName", "rating", "comment", "review_text"])

def main():
    st.set_page_config(page_title="NeuroReviews Editor", layout="wide")
    st.title("Управление отзывами NeuroReviews")

    df = load_data()

    if "review_comment" not in st.session_state:
        st.session_state["review_comment"] = ""
    if "generated_review" not in st.session_state:
        st.session_state["generated_review"] = ""

    col_form, col_tags = st.columns([2, 1])

    with col_form:
        with st.form(key="review_form"):
            st.subheader("Параметры отзыва")
            
            if not df.empty and "studentName" in df.columns:
                students_list = sorted(df["studentName"].dropna().unique().tolist())
                selected_student = st.selectbox("Студент:", students_list)
                
                filtered_courses = sorted(df[df["studentName"] == selected_student]["courseName"].dropna().unique().tolist())
                if not filtered_courses and "courseName" in df.columns:
                    filtered_courses = sorted(df["courseName"].dropna().unique().tolist())
                selected_course = st.selectbox("Курс:", filtered_courses if filtered_courses else ["Базовый курс"])
            else:
                selected_student = st.text_input("Имя студента (вручную):")
                selected_course = st.text_input("Название курса (вручную):")

            rating = st.slider("Оценка:", min_value=1, max_value=5, value=5)

            comment = st.text_area(
                "Комментарий преподавателя / Базовый текст:", 
                value=st.session_state["review_comment"],
                height=150,
                key="review_text_area"
            )
            
            submit_button = st.form_submit_button(label="🤖 Сгенерировать отзыв")

    with col_tags:
        st.subheader("Быстрые теги")
        available_tags = [
            'отличная теория', 'отличная практика', 'высокая вовлеченность', 
            'самостоятельная работа', 'высокая активность', 'навык самопроверки', 
            'требуется концентрация', 'разбор ошибок'
        ]
        
        for idx, tag in enumerate(available_tags):
            if st.button(f"➕ {tag}", key=f"tag_{idx}", use_container_width=True):
                current_text = st.session_state.get("review_text_area", "")
                if current_text.strip():
                    new_text = f"{current_text.strip()} #{tag.replace(' ', '_')} "
                else:
                    new_text = f"#{tag.replace(' ', '_')} "
                st.session_state["review_comment"] = new_text
                st.rerun()

    if submit_button:
        if not comment.strip():
            st.error("Пожалуйста, заполните поле комментария или добавьте теги!")
        else:
            with st.spinner("Генерация развернутого отзыва нейросетью..."):
                model, tokenizer, device = load_model_and_tokenizer()
                if model and tokenizer:
                    try:
                        prompt = f"Студент: {selected_student}. Курс: {selected_course}. Оценка: {rating}. Комментарий: {comment}. Сгенерируй развернутый отзыв для родителей:"
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                        st.session_state["generated_review"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    except Exception as e:
                        st.session_state["generated_review"] = f"Ошибка генерации модели: {str(e)}"
                else:
                    st.session_state["generated_review"] = f"Привет! Локальная модель FRED-T5 не найдена по пути {MODEL_PATH.resolve()}. Вот базовый отзыв: Студент {selected_student} успешно прошел курс {selected_course} с оценкой {rating}. Заметки: {comment}."

    if st.session_state["generated_review"]:
        st.write("---")
        st.subheader("✨ Результат генерации")
        
        final_review = st.text_area(
            label="Вы можете отредактировать готовый отзыв перед сохранением:",
            value=st.session_state["generated_review"],
            height=200,
            key="final_review_area"
        )
        
        if st.button("💾 Сохранить финальный отзыв в CSV", type="primary"):
            new_row = {
                "studentName": selected_student,
                "courseName": selected_course,
                "rating": rating,
                "comment": comment,
                "review_text": final_review
            }
            
            try:
                updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                updated_df.to_csv(CSV_PATH, index=False)
                st.success("Отзыв успешно сохранен в базу данных (history_teacher.csv)!")
                st.session_state["review_comment"] = ""
                st.session_state["generated_review"] = ""
                st.rerun()
            except Exception as e:
                st.error(f"Не удалось сохранить файл: {str(e)}")

if __name__ == "__main__":
    main()
