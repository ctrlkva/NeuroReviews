import re
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class EdTechReviewDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Класс датасета для подготовки данных к обучению NLG-модели FRED-T5.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Загружаем данные и сортируем по времени
        self.df = pd.read_csv(csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(by=['studentName', 'timestamp']).reset_index(drop=True)

        self.data = self._prepare_samples()

    def _prepare_samples(self) -> list:
        samples = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            
            # Парсим JSON-теги обратно в текстовый формат для модели
            try:
                tags_dict = json.loads(row['teacher_tags'])
                tags_formatted = f"Общая оценка: {tags_dict.get('общая_оценка', 'хорошо')}, Плюсы: {tags_dict.get('хорошо', 'нет')}, Минусы: {tags_dict.get('плохо', 'нет')}"
            except Exception:
                tags_formatted = f"Метрики: {row.get('tags', 'успешно продолжает обучение')}"
            
            # Формируем историю прошлых периодов
            history_text = "Нет предыдущих отзывов"
            student_group = self.df[self.df['studentName'] == row['studentName']].sort_values('timestamp')
            past_records = student_group[student_group['timestamp'] < row['timestamp']]
            if not past_records.empty:
                history_text = past_records.iloc[-1]['reviewText']
            
            # Сборка единого сквозного промпта: БД + Темы от учителя + Теги + История
            prompt = (
                f"<lm>Студент: {row['studentName']}. "
                f"Курс: {row['courseName']}. "
                f"Прогресс: {row['total_hours']} ч. "
                f"Пройдено на уроке: {row['teacher_notes']} "
                f"Метрики: {tags_formatted}. "
                f"История: {history_text}. "
                f"Отзыв:</s>"
            )
            
            samples.append({
                'prompt': prompt,
                'target': row['reviewText']
            })
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        source_inputs = self.tokenizer(
            sample['prompt'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_inputs = self.tokenizer(
            sample['target'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_inputs["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_inputs["input_ids"].squeeze(0),
            "attention_mask": source_inputs["attention_mask"].squeeze(0),
            "labels": labels
        }
