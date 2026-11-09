import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from dataset import EdTechReviewDataset
from pathlib import Path

# Установка зеркала Hugging Face для обхода проблем со скачиванием в РФ
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    # Динамическое определение абсолютного пути относительно файла скрипта
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR.parent / "models" / "FRED-T5-large"
    DATASET_PATH = BASE_DIR.parent / "data" / "perfect_reviews_train.csv"
    OUTPUT_DIR = BASE_DIR.parent / "src" / "lora_output"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Запуск процесса обучения на устройстве: {device}")

    # Создаем папку для моделей, если ее нет
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Если локальной папки с моделью нет, скачиваем из репозитория и сохраняем локально
    if not MODEL_PATH.exists():
        print(f"Локальная папка {MODEL_PATH.resolve()} не найдена.")
        print("Скачивание модели из репозитория ai-forever/FRED-T5-large и сохранение на локальный диск...")
        
        # Скачиваем веса и токенизатор напрямую с хаба через зеркало
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRED-T5-large", eos_token='<|endoftext|>', bos_token='<|endoftext|>')
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "ai-forever/FRED-T5-large",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Сохраняем в локальную директорию для будущих запусков
        tokenizer.save_pretrained(str(MODEL_PATH.resolve().as_posix()))
        model.save_pretrained(str(MODEL_PATH.resolve().as_posix()))
        print(f"Модель успешно скачана и сохранена в: {MODEL_PATH.resolve()}")

    # Преобразуем путь в строку с прямыми слэшами для совместимости с HF Hub валидатором на Windows
    model_path_str = str(MODEL_PATH.resolve().as_posix())
    
    print(f"Загрузка токенизатора из: {model_path_str}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_str, 
        eos_token='<|endoftext|>', 
        bos_token='<|endoftext|>',
        local_files_only=True
    )
    
    print(f"Загрузка модели из: {model_path_str}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path_str,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False, 
        r=8,
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Обучающий датасет не найден по пути {DATASET_PATH.resolve()}.")
        
    train_dataset = EdTechReviewDataset(csv_path=str(DATASET_PATH), tokenizer=tokenizer, max_length=512)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR.resolve().as_posix()),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        num_train_epochs=10,
        logging_steps=2,
        save_strategy="no",
        fp16=(device == "cuda"),
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Начало дообучения модели...")
    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR.resolve().as_posix()))
    print(f"Обучение завершено! Веса сохранены в: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
