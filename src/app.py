import os
import torch
import streamlit as st
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Инициализация путей
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DEFAULT_CSV_PATH = DATA_DIR / "history_teacher.csv"
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


def get_csv_path() -> Path:
    """Ищет целевой CSV или берет первый доступный CSV в папке data."""
    if DEFAULT_CSV_PATH.exists():
        return DEFAULT_CSV_PATH
    if DATA_DIR.exists():
        first_csv = next(DATA_DIR.glob("*.csv"), None)
        if first_csv is not None:
            return first_csv
    return DEFAULT_CSV_PATH


def load_data(csv_path: Path):
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
            df.columns = df.columns.str.strip()
            return df, None
        except Exception as e:
            return pd.DataFrame(columns=["studentName", "courseName", "rating", "comment", "review_text"]), str(e)
    return pd.DataFrame(columns=["studentName", "courseName", "rating", "comment", "review_text"]), f"Файл не найден по пути: {csv_path}"


def add_tag_to_comment(tag: str):
    """Коллбэк для мгновенного добавления тега в поле ввода."""
    formatted_tag = f"#{tag.replace(' ', '_')}"
    current_text = st.session_state.get("review_text_area", "").strip()
    if current_text:
        st.session_state["review_text_area"] = f"{current_text} {formatted_tag} "
    else:
        st.session_state["review_text_area"] = f"{formatted_tag} "


def main():
    st.set_page_config(page_title="NeuroReviews Editor", layout="wide")
    st.title("Управление отзывами NeuroReviews")

    csv_path = get_csv_path()
    df, load_error = load_data(csv_path)

    if load_error:
        st.warning(f"⚠️ Проблема с загрузкой CSV ({csv_path.name}): {load_error}")

    if "review_text_area" not in st.session_state:
        st.session_state["review_text_area"] = ""
    if "generated_review" not in st.session_state:
        st.session_state["generated_review"] = ""

    col_form, col_tags = st.columns(2)

    with col_form:
        st.subheader("Параметры отзыва")

        # Выбор студента и курса вынесен НАД формой для мгновенной фильтрации курсов
        if not df.empty and "studentName" in df.columns:
            students_list = sorted(df["studentName"].dropna().astype(str).unique().tolist())
            selected_student = st.selectbox("Студент:", students_list)

            if "courseName" in df.columns:
                filtered_courses = sorted(
                    df[df["studentName"].astype(str) == selected_student]["courseName"]
                    .dropna().astype(str).unique().tolist()
                )
                if not filtered_courses:
                    filtered_courses = sorted(df["courseName"].dropna().astype(str).unique().tolist())
            else:
                filtered_courses = []

            selected_course = st.selectbox("Курс:", filtered_courses if filtered_courses else ["Базовый курс"])
        else:
            if not df.empty:
                st.info(f"Найдены столбцы: {', '.join(df.columns)}. Ожидался столбец 'studentName'.")
            selected_student = st.text_input("Имя студента (вручную):")
            selected_course = st.text_input("Название курса (вручную):")

        # Форма для оценки, комментария и отправки на генерацию
        with st.form(key="review_form"):
            rating = st.slider("Оценка:", min_value=1, max_value=5, value=5)

            comment = st.text_area(
                "Комментарий преподавателя / Базовый текст:",
                height=150,
                key="review_text_area"
            )

            submit_button = st.button(label="🤖 Сгенерировать отзыв", type="primary")


    with col_tags:
        st.subheader("Быстрые теги")
        available_tags = [
            "отличная теория", "отличная практика", "высокая вовлеченность",
            "самостоятельная работа", "высокая активность", "навык самопроверки",
            "требуется концентрация", "разбор ошибок"
        ]

        for idx, tag in enumerate(available_tags):
            st.button(
                f"➕ {tag}",
                key=f"tag_{idx}",
                use_container_width=True,
                on_click=add_tag_to_comment,
                args=(tag,)
            )

    if submit_button:
        if not comment.strip():
            st.error("Пожалуйста, заполните поле комментария или добавьте теги!")
        else:
            with st.spinner("Генерация развернутого отзыва нейросетью..."):
                model, tokenizer, device = load_model_and_tokenizer()
                if model and tokenizer:
                    try:
                        # Отделяем теги (#тег) от текстовых заметок
                        words = comment.split()
                        tags_list = [w.lstrip("#").replace("_", " ") for w in words if w.startswith("#")]
                        notes_list = [w for w in words if not w.startswith("#")]

                        teacher_notes = " ".join(notes_list).strip() or "освоение программы курса"
                        tags_str = ", ".join(tags_list) if tags_list else "успешно продолжает обучение"
                        tags_formatted = f"Оценка: {rating}/5, Метрики: {tags_str}"

                        # Достаем прогресс (часы) и прошлый отзыв студента из CSV
                        total_hours = 0
                        history_text = "Нет предыдущих отзывов"

                        if not df.empty and "studentName" in df.columns:
                            student_records = df[df["studentName"].astype(str) == str(selected_student)]
                            if not student_records.empty:
                                if "timestamp" in student_records.columns:
                                    student_records = student_records.sort_values("timestamp")
                                
                                last_row = student_records.iloc[-1]
                                if "total_hours" in student_records.columns and pd.notna(last_row["total_hours"]):
                                    total_hours = last_row["total_hours"]
                                
                                rev_col = "reviewText" if "reviewText" in student_records.columns else "review_text"
                                if rev_col in student_records.columns and pd.notna(last_row[rev_col]):
                                    history_text = str(last_row[rev_col]).strip()

                        # Промпт в точном формате обучения (dataset.py)
                        prompt = (
                            f"<lm>Студент: {selected_student}. "
                            f"Курс: {selected_course}. "
                            f"Прогресс: {total_hours} ч. "
                            f"Пройдено на уроке: {teacher_notes} "
                            f"Метрики: {tags_formatted}. "
                            f"История: {history_text}. "
                            f"Отзыв:</s>"
                        )

                        inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=512,
                                num_beams=4,
                                repetition_penalty=1.2,
                                early_stopping=True
                            )
                        st.session_state["generated_review"] = tokenizer.decode(outputs.squeeze(0), skip_special_tokens=True)
                    except Exception as e:
                        st.session_state["generated_review"] = f"Ошибка генерации модели: {str(e)}"
                else:
                    st.session_state["generated_review"] = (
                        f"Привет! Локальная модель FRED-T5 не найдена по пути {MODEL_PATH.resolve()}. "
                        f"Вот базовый отзыв: Студент {selected_student} успешно прошел курс {selected_course} "
                        f"с оценкой {rating}. Заметки: {comment}."
                    )

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
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                updated_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                st.success(f"Отзыв успешно сохранен в {csv_path.name}!")
                st.session_state["review_text_area"] = ""
                st.session_state["generated_review"] = ""
                st.rerun()
            except Exception as e:
                st.error(f"Не удалось сохранить файл: {str(e)}")
                
if __name__ == "__main__":
    main()
