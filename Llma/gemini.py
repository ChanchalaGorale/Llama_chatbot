from transformers import GeminiTokenizer, GeminiForTokenClassification, Trainer, TrainingArguments
import torch

# Load the tokenizer and model
tokenizer = GeminiTokenizer.from_pretrained('huggingface/gemini')
model = GeminiForTokenClassification.from_pretrained('huggingface/gemini', num_labels=len(relation_list))

# Function to read the data
def read_re_data(file_path):
    import json
    with open(file_path, 'r') as file:
        lines = file.readlines()
    texts, entities, relations = [], [], []
    for line in lines:
        data = json.loads(line)
        texts.append(data['text'])
        entities.append(data['entities'])
        relations.append(data['relations'])
    return texts, entities, relations

texts, entities, relations = read_re_data('re_data.jsonl')

# Tokenize and align labels
def tokenize_and_align_labels(texts, entities, relations):
    tokenized_inputs = tokenizer(texts, truncation=True, padding=True)
    label_all_tokens = True

    new_labels = []
    for i, relation in enumerate(relations):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)
        for rel in relation:
            head_start, head_end = entities[i][rel['head']]['start'], entities[i][rel['head']]['end']
            tail_start, tail_end = entities[i][rel['tail']]['start'], entities[i][rel['tail']]['end']
            for word_idx in word_ids:
                if word_idx is None:
                    continue
                if head_start <= word_idx < head_end and tail_start <= word_idx < tail_end:
                    label_ids[word_idx] = relation_list.index(rel['label'])
        new_labels.append(label_ids)
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

relation_list = ['acquired', 'hired']
tokenized_inputs = tokenize_and_align_labels(texts, entities, relations)

# Convert to torch Dataset
class REDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_inputs):
        self.input_ids = tokenized_inputs['input_ids']
        self.attention_mask = tokenized_inputs['attention_mask']
        self.labels = tokenized_inputs['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = REDataset(tokenized_inputs)

# Fine-tuning the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
