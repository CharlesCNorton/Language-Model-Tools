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

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        try:
            print(f"{Fore.BLUE}Tokenizing and encoding data...{Style.RESET_ALL}")
            texts = [str(text) for text in texts.tolist() if pd.notna(text)]
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
            self.labels = labels.tolist()
            print(f"{Fore.GREEN}Data tokenization and encoding complete.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error during data tokenization and encoding: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Make sure that all texts are strings and not missing. If you have non-English texts, ensure the tokenizer supports them.{Style.RESET_ALL}")
            raise e

    def __getitem__(self, idx):
        try:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        except Exception as e:
            print(f"{Fore.RED}Error retrieving item at index {idx}: {e}{Style.RESET_ALL}")
            raise e

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    try:
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    except Exception as e:
        print(f"{Fore.RED}Error computing metrics: {e}{Style.RESET_ALL}")
        return None

def train_model(train_dataset, val_dataset, epochs, batch_size, warmup_steps, weight_decay):
    try:
        print(f"{Fore.BLUE}Initializing model...{Style.RESET_ALL}")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        print(f"{Fore.GREEN}Model initialized.{Style.RESET_ALL}")

        print(f"{Fore.BLUE}Setting up training arguments...{Style.RESET_ALL}")
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            eval_steps=50,
            save_steps=50,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True
        )
        print(f"{Fore.GREEN}Training arguments set.{Style.RESET_ALL}")

        print(f"{Fore.BLUE}Setting up trainer...{Style.RESET_ALL}")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        print(f"{Fore.GREEN}Trainer set up.{Style.RESET_ALL}")

        print(f"{Fore.BLUE}Starting model training...{Style.RESET_ALL}")
        trainer.train()
        print(f"{Fore.GREEN}Model training complete.{Style.RESET_ALL}")

        print(f"{Fore.BLUE}Evaluating model...{Style.RESET_ALL}")
        eval_result = trainer.evaluate()
        print(f"{Fore.GREEN}Model evaluation complete. Evaluation result: {eval_result}{Style.RESET_ALL}")
        return model, trainer, eval_result
    except Exception as e:
        print(f"{Fore.RED}Model training failed: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Check the model parameters and dataset integrity. Ensure the batch size and learning rate are appropriately set for your hardware capabilities.{Style.RESET_ALL}")
        return None, None, None

def auto_train(train_dataset, val_dataset, num_iterations):
    try:
        best_model = None
        best_trainer = None
        best_eval_result = None
        best_f1 = 0
        best_params = None
        all_results = []

        for i in range(num_iterations):
            print(f"{Fore.BLUE}Auto-training iteration {i+1}/{num_iterations}...{Style.RESET_ALL}")

            epochs = random.randint(2, 10)
            batch_size = random.randint(8, 64)
            warmup_steps = random.randint(0, 1000)
            weight_decay = random.uniform(0.01, 0.1)
            params = {
                'epochs': epochs,
                'batch_size': batch_size,
                'warmup_steps': warmup_steps,
                'weight_decay': weight_decay
            }

            print(f"{Fore.BLUE}Training with hyperparameters: {params}{Style.RESET_ALL}")
            model, trainer, eval_result = train_model(train_dataset, val_dataset, **params)
            if eval_result:
                all_results.append((params, eval_result))
                if eval_result['eval_f1'] > best_f1:
                    best_model = model
                    best_trainer = trainer
                    best_eval_result = eval_result
                    best_f1 = eval_result['eval_f1']
                    best_params = params
            print(f"{Fore.GREEN}Auto-training iteration {i+1}/{num_iterations} complete.{Style.RESET_ALL}")

        if best_model:
            print(f"{Fore.GREEN}Best model found with F1 score: {best_f1}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Best hyperparameters: {best_params}{Style.RESET_ALL}")

            print(f"{Fore.BLUE}Running additional iterations with the same hyperparameters as the best model...{Style.RESET_ALL}")
            additional_results = []
            for params, _ in all_results:
                print(f"{Fore.BLUE}Additional iteration with hyperparameters: {params}...{Style.RESET_ALL}")
                model, trainer, eval_result = train_model(train_dataset, val_dataset, **params)
                if eval_result:
                    additional_results.append((params, eval_result))
                    if eval_result['eval_f1'] > best_f1:
                        best_model = model
                        best_trainer = trainer
                        best_eval_result = eval_result
                        best_f1 = eval_result['eval_f1']
                print(f"{Fore.GREEN}Additional iteration complete.{Style.RESET_ALL}")

            print(f"{Fore.GREEN}Final best model found with F1 score: {best_f1}{Style.RESET_ALL}")
            all_results.extend(additional_results)
            return best_model, best_trainer, best_eval_result, best_params, all_results
        else:
            print(f"{Fore.RED}No suitable model found during auto-training.{Style.RESET_ALL}")
            return None, None, None, None, all_results
    except Exception as e:
        print(f"{Fore.RED}Auto-training failed: {e}{Style.RESET_ALL}")
        return None, None, None, None, []

def save_model(model, params, all_results):
    try:
        print(f"{Fore.BLUE}Opening file dialog to select save location...{Style.RESET_ALL}")
        root = tk.Tk()
        root.withdraw()
        model_path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("PyTorch Model", "*.pt")])
        if model_path:
            print(f"{Fore.BLUE}Saving model to {model_path}...{Style.RESET_ALL}")
            model.save_pretrained(model_path)
            print(f"{Fore.GREEN}Model saved successfully at {model_path}{Style.RESET_ALL}")

            info_path = model_path.replace(".pt", "_info.json")
            info = {
                'best_params': params,
                'all_results': [
                    {
                        'params': result[0],
                        'eval_result': result[1]
                    } for result in all_results
                ]
            }
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=4)
            print(f"{Fore.GREEN}Model information saved at {info_path}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Model saving cancelled by the user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed to save model: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Ensure you have write permissions to the selected directory and sufficient disk space.{Style.RESET_ALL}")


def load_model():
    try:
        print(f"{Fore.BLUE}Opening file dialog to select model file...{Style.RESET_ALL}")
        root = tk.Tk()
        root.withdraw()
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if model_path:
            print(f"{Fore.BLUE}Loading model from {model_path}...{Style.RESET_ALL}")
            model = BertForSequenceClassification.from_pretrained(model_path)
            print(f"{Fore.GREEN}Model loaded successfully from {model_path}{Style.RESET_ALL}")
            return model
        else:
            print(f"{Fore.YELLOW}Model loading cancelled by the user.{Style.RESET_ALL}")
            return None
    except Exception as e:
        print(f"{Fore.RED}Failed to load model: {e}{Style.RESET_ALL}")
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{Fore.BLUE}Using device: {device}{Style.RESET_ALL}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = None
    train_dataset = val_dataset = None
    model = None
    params = None
    all_results = []

    while True:
        print("\nMenu:")
        print("1. Download and prepare the dataset")
        print("2. Train new model with custom hyperparameters")
        print("3. Auto-train model with different hyperparameters")
        print("4. Predict sentiment of new text")
        print("5. Save current best model")
        print("6. Load model")
        print("7. Exit")
        choice = input("Enter your choice: ")

        try:
            if choice == '1':
                if dataset is None:
                    dataset = safe_load_dataset("imdb")
                    if dataset:
                        train_data, val_data = prepare_data(dataset)
                        train_dataset = SentimentDataset(train_data['text'], train_data['label'], tokenizer)
                        val_dataset = SentimentDataset(val_data['text'], val_data['label'], tokenizer)
                else:
                    print(f"{Fore.YELLOW}Dataset already downloaded and prepared.{Style.RESET_ALL}")
            elif choice == '2':
                if train_dataset and val_dataset:
                    epochs = int(input("Enter the number of epochs: "))
                    batch_size = int(input("Enter batch size: "))
                    warmup_steps = int(input("Enter warmup steps: "))
                    weight_decay = float(input("Enter weight decay: "))
                    model, trainer, eval_result = train_model(train_dataset, val_dataset, epochs, batch_size, warmup_steps, weight_decay)
                    params = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'warmup_steps': warmup_steps,
                        'weight_decay': weight_decay
                    }
                    all_results = [(params, eval_result)]
                else:
                    print(f"{Fore.RED}No dataset available. Please download and prepare the dataset first.{Style.RESET_ALL}")
            elif choice == '3':
                if train_dataset and val_dataset:
                    num_iterations = int(input("Enter the number of auto-training iterations: "))
                    model, trainer, eval_result, params, all_results = auto_train(train_dataset, val_dataset, num_iterations)
                else:
                    print(f"{Fore.RED}No dataset available. Please download and prepare the dataset first.{Style.RESET_ALL}")
            elif choice == '4':
                if model:
                    text = input("Enter text to analyze: ")
                    print(f"{Fore.BLUE}Tokenizing input text...{Style.RESET_ALL}")
                    inputs = tokenizer(text, return_tensors='pt').to(device)
                    print(f"{Fore.BLUE}Running sentiment analysis...{Style.RESET_ALL}")
                    outputs = model(**inputs)
                    sentiment = 'Positive' if outputs.logits.argmax().item() == 1 else 'Negative'
                    print(f"{Fore.BLUE}The sentiment of the text is: {sentiment}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}No model loaded or trained. Please train or load a model first.{Style.RESET_ALL}")
            elif choice == '5':
                if model:
                    save_model(model, params, all_results)
                else:
                    print(f"{Fore.RED}No model to save. Train or load a model first.{Style.RESET_ALL}")
            elif choice == '6':
                model = load_model()
                if model:
                    model.to(device)
            elif choice == '7':
                print(f"{Fore.CYAN}Exiting the application...{Style.RESET_ALL}")
                break
            else:
                print(f"{Fore.YELLOW}Invalid choice. Please enter a number from 1 to 7.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
