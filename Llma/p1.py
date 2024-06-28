from transformers import LlamaTokenizer, LlamaForTokenClassification, Trainer, TrainingArguments
import torch

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('huggingface/llama')
model = LlamaForTokenClassification.from_pretrained('huggingface/llama', num_labels=len(label_list))

# Function to read the data
def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sentences, labels = [], []
    sentence, label = [], []
    for line in lines:
        if line.strip() == '':
            sentences.append(sentence)
            labels.append(label)
            sentence, label = [], []
        else:
            token, tag = line.split()
            sentence.append(token)
            label.append(tag)
    return sentences, labels

sentences, labels = read_data('ner_data.txt')

# Tokenize and align labels
def tokenize_and_align_labels(sentences, labels):
    tokenized_inputs = tokenizer(sentences, truncation=True, is_split_into_words=True, padding=True)
    label_all_tokens = True

    new_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_list.index(label[word_idx]))
            else:
                label_ids.append(label_list.index(label[word_idx]) if label_all_tokens else -100)
            previous_word_idx = word_idx
        new_labels.append(label_ids)
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

label_list = ['O', 'B-ORG', 'B-LOC', 'B-MONEY', 'I-MONEY']
tokenized_inputs = tokenize_and_align_labels(sentences, labels)

# Convert to torch Dataset
class NERDataset(torch.utils.data.Dataset):
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

train_dataset = NERDataset(tokenized_inputs)

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
