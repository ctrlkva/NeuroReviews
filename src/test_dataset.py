import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer 

class EdTechReviewDataset(Dataset):
def **init**(self, csv_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
"""
Класс датасета с поддержкой тегов в формате {тег} в конце или внутри текста. 

:param csv_path: Путь к файлу с историей отзывов
:param tokenizer: Предобученный токенизатор из Hugging Face
:param max_length: Максимальная длина последовательности токенов
"""
self.tokenizer = tokenizer
self.max_length = max_length

# Загружаем данные и сортируем по времени

self.df = pd.read_csv(csv_path)
self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
self.df = self.df.sort_values(by=['studentName', 'timestamp']).reset_index(drop=True)

self.data = self._prepare_samples()

def _extract_tags_from_text(self, text: str) -> tuple[str, list]:
"""
Ищет в тексте конструкции вида {тег}.
Возвращает очищенный текст отзыва и список найденных тегов.
Если тегов в фигурных скобках нет, генерирует базовые теги на основе анализа текста.
""" 

### Поиск всех паттернов в фигурных скобках

tags = re.findall(r'{(.*?)}', text) 

if tags: 

### Удаляем теги в фигурных скобках из текста, чтобы получить чистый целевой отзыв

clean_text = re.sub(r'{.*?}', '', text).strip() 

### Убираем лишние пробелы, если теги были разделены ими

clean_text = re.sub(r'\s+', ' ', clean_text)
return clean_text, tags 

# Если тегов нет в самом тексте, симулируем их (как в прошлой версии)

review_lower = text.lower()
mock_tags = []
if "отличн" in review_lower or "прекрасн" in review_lower:
mock_tags.append("отличные результаты")
if "старается" in review_lower:
mock_tags.append("старается на уроках")
if "путается" in review_lower:
mock_tags.append("иногда путается в материале")
if "поведени" in review_lower or "шуметь" in review_lower:
mock_tags.append("обратить внимание на поведение")

if not mock_tags:
mock_tags.append("успешно продолжает обучение")

return text.strip(), mock_tags

def _prepare_samples(self) -> list:
samples = [] 

for student, group in self.df.groupby('studentName'):
group = group.sort_values('timestamp').reset_index(drop=True)
for i in range(len(group)):
    current_row = group.iloc[i]
    raw_review = current_row['reviewText']

### Извлекаем теги и чистим текст отзыва

clean_review, tags = self._extract_tags_from_text(raw_review) 

### Форматируем теги строкой через запятую

tags_str = ", ".join(tags) 

### Формируем историю прошлых периодов

if i > 0:
# Из прошлых отзывов тоже вырезаем технические теги, если они там были
prev_raw = group.iloc[i-1]['reviewText']
history_text, _ = self._extract_tags_from_text(prev_raw)
else:
history_text = "Нет предыдущих отзывов" 

### Новый формат промпта согласно вашему требованию

### Теги теперь идут в конце контекста перед блоком истории или как финальный триггер

prompt = (
f"Студент: {current_row['studentName']}. "
f"Курс: {current_row['courseName']}. "
f"Прогресс: {current_row['total_hours']} ч. "
f"История: {history_text}. "
f"Теги: {tags_str}."
) 

### В качестве таргета отдаем чистый отзыв, но при желании

### можно настроить модель так, чтобы она сама генерировала {теги} в конце.

    # Пока оставляем чистый текст отзыва, как результат генерации по тегам.
    target = clean_review
    
    samples.append({
        'prompt': prompt,
        'target': target
    })

return samples

def **len**(self):
return len(self.data) 

def **getitem**(self, idx):
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