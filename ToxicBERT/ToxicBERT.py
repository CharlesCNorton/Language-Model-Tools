import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from colorama import Fore, Style
import colorama
import tkinter as tk
from tkinter import filedialog
import json
import random
colorama.init()

def safe_load_dataset(dataset_name):
    try:
        print(f"{Fore.BLUE}Loading dataset {dataset_name}...{Style.RESET_ALL}")
        dataset = load_dataset(dataset_name)
        print(f"{Fore.GREEN}Dataset {dataset_name} successfully loaded.{Style.RESET_ALL}")
        return dataset
    except Exception as e:
        print(f"{Fore.RED}Failed to load dataset {dataset_name}. Error: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please check if the dataset name '{dataset_name}' is correct and available. You may also want to check your network connection if it's an online dataset.{Style.RESET_ALL}")
        return None

def prepare_data(dataset):
    try:
        print(f"{Fore.BLUE}Preparing data...{Style.RESET_ALL}")
        train_data = pd.DataFrame(dataset['train'])
        test_data = pd.DataFrame(dataset['test'])
        print(f"{Fore.GREEN}Data preparation successful.{Style.RESET_ALL}")
        return train_data, test_data
    except Exception as e:
        print(f"{Fore.RED}Error during data preparation: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Check if the dataset is properly formatted. Ensure columns 'train' and 'test' exist and contain the necessary data.{Style.RESET_ALL}")
        return None, None

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        try:
            print(f"{Fore.BLUE}Tokenizing and encoding data...{Style.RESET_ALL}")
            texts = [str(text) for text in texts.tolist() if pd.notna(text)]
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
            self.labels = torch.tensor(labels.values)
            print(f"{Fore.GREEN}Data tokenization and encoding complete.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error during data tokenization and encoding: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Make sure that all texts are strings and not missing. If you have non-English texts, ensure the tokenizer supports them.{Style.RESET_ALL}")
            raise e

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.sigmoid() > 0.5
    accuracy = accuracy_score(labels.cpu(), preds.cpu())
    precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='macro')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(train_dataset, val_dataset, epochs, batch_size, warmup_steps, weight_decay):
    print(f"{Fore.BLUE}Initializing model...{Style.RESET_ALL}")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
    print(f"{Fore.GREEN}Model initialized.{Style.RESET_ALL}")

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_steps=1,
        eval_steps=50,
        save_steps=50,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return model, trainer

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_name = "civil_comments"
    dataset = safe_load_dataset(dataset_name)
    if dataset:
        train_data, test_data = prepare_data(dataset)
        train_dataset = ToxicityDataset(train_data['text'], train_data[['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']], tokenizer)
        val_dataset = ToxicityDataset(test_data['text'], test_data[['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']], tokenizer)

        model, trainer = train_model(train_dataset, val_dataset, epochs=3, batch_size=16, warmup_steps=100, weight_decay=0.01)

        model.save_pretrained('./toxic_model')
        tokenizer.save_pretrained('./toxic_model')

if __name__ == "__main__":
    main()
