import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, get_scheduler
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tkinter import Tk, filedialog, simpledialog, messagebox
import random

MODEL_NAME = 'bert-base-uncased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
CATEGORY_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
HYPERPARAMETER_OPTIONS = {
    'BATCH_SIZE': [16, 32, 48],
    'LEARNING_RATE': [2e-5, 3e-5, 5e-5],
    'EPOCHS': [1, 2, 3]
}

try:
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
except OSError as e:
    print(f"Error creating directories: {e}")
    exit(1)

class NERDataset(Dataset):
    def __init__(self, tokenizer, texts, tags):
        self.tokenizer = tokenizer
        self.texts = texts
        self.tags = tags
        print(f"Dataset initialized with {len(self.texts)} texts.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            tokens = self.texts[idx]
            tokenized_input = self.tokenizer(tokens, is_split_into_words=True,
                                             add_special_tokens=True, max_length=MAX_LEN,
                                             padding='max_length', truncation=True,
                                             return_attention_mask=True, return_tensors='pt')

            word_ids = tokenized_input.word_ids()
            labels = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    labels.append(-100)
                elif word_idx != previous_word_idx:
                    labels.append(self.tags[idx][word_idx])
                else:
                    labels.append(self.tags[idx][word_idx] if word_idx is not None else -100)
                previous_word_idx = word_idx

            return {
                'input_ids': tokenized_input['input_ids'].squeeze(0),
                'attention_mask': tokenized_input['attention_mask'].squeeze(0),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        except IndexError as e:
            print(f"Index error: {e}")
            return None
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            return None

def create_data_loaders(tokenizer, texts, tags, batch_size):
    try:
        print("Splitting data into training and validation sets...")
        train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=0.1, random_state=42)
        print(f"Data split into {len(train_texts)} training and {len(val_texts)} validation texts.")

        train_dataset = NERDataset(tokenizer, train_texts, train_tags)
        val_dataset = NERDataset(tokenizer, val_texts, val_tags)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print("Data loaders created and ready for training and validation.")
        return train_loader, val_loader
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return None, None

def train_model(model, train_loader, optimizer, scheduler, epochs):
    try:
        model.train()
        print("Starting training...")

        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}...")
            for batch_index, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                print(f'Epoch {epoch+1}, Batch {batch_index+1}, Loss: {loss.item()}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            print(f"Epoch {epoch+1} completed.")

        print("Training process completed.")
    except Exception as e:
        print(f"Error during training: {e}")

def evaluate_model(model, data_loader, dataset_type="validation"):
    try:
        model.eval()
        all_preds = []
        all_labels = []
        print(f"Starting {dataset_type} evaluation...")

        with torch.no_grad():
            for batch_index, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

                labels = labels.cpu().numpy()
                preds = preds.cpu().numpy()
                active_accuracy = labels != -100
                active_labels = labels[active_accuracy]
                active_preds = preds[active_accuracy]

                all_labels.extend(active_labels)
                all_preds.extend(active_preds)

                if batch_index % 10 == 0:
                    print(f"Processed {batch_index+1}/{len(data_loader)} batches for {dataset_type} evaluation.")

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        report = classification_report(all_labels, all_preds, target_names=CATEGORY_NAMES, zero_division=0)

        print(f"{dataset_type.capitalize()} Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        print(f"Detailed {dataset_type} classification report:")
        print(report)

        return precision, recall, f1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0, 0, 0

def search_hyperparameters(texts, tags):
    try:
        print("Starting hyperparameter search...")
        best_f1 = 0
        best_hyperparams = {}
        total_trials = 10

        for trial in range(total_trials):
            batch_size = random.choice(HYPERPARAMETER_OPTIONS['BATCH_SIZE'])
            learning_rate = random.choice(HYPERPARAMETER_OPTIONS['LEARNING_RATE'])
            epochs = random.choice(HYPERPARAMETER_OPTIONS['EPOCHS'])

            print(f"Trial {trial+1}/{total_trials}: Batch Size={batch_size}, Learning Rate={learning_rate}, Epochs={epochs}")

            tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
            model = BertForTokenClassification.from_pretrained(
                MODEL_NAME, num_labels=len(CATEGORY_NAMES)).to(DEVICE)

            train_loader, val_loader = create_data_loaders(tokenizer, texts, tags, batch_size)
            if train_loader is None or val_loader is None:
                print("Skipping trial due to data loader creation failure.")
                continue

            optimizer = AdamW(model.parameters(), lr=learning_rate)
            total_steps = len(train_loader) * epochs
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            train_model(model, train_loader, optimizer, scheduler, epochs)
            _, _, f1 = evaluate_model(model, val_loader)

            if f1 > best_f1:
                best_f1 = f1
                best_hyperparams = {
                    'BATCH_SIZE': batch_size,
                    'LEARNING_RATE': learning_rate,
                    'EPOCHS': epochs
                }
            print(f"Trial {trial+1} completed with F1 Score: {f1:.4f}")

        print(f"Best hyperparameters found: {best_hyperparams} with F1 Score: {best_f1:.4f}")
        return best_hyperparams
    except Exception as e:
        print(f"Error during hyperparameter search: {e}")
        return {}

def predict_entities(model, tokenizer, text):
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].flatten())
        word_ids = inputs.word_ids()
        predicted_entities = [CATEGORY_NAMES[pred] for pred in predictions.flatten().cpu().numpy()]

        aligned_entities = []
        previous_word_idx = None
        for word_idx, token, entity in zip(word_ids, tokens, predicted_entities):
            if word_idx is not None and word_idx != previous_word_idx:
                aligned_entities.append((token, entity))
            previous_word_idx = word_idx

        return aligned_entities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return []

def main():
    try:
        gui_root = Tk()
        gui_root.withdraw()

        print("Loading dataset...")
        dataset = load_dataset("conll2003", split='train')
        texts = [data['tokens'] for data in dataset]
        tags = [data['ner_tags'] for data in dataset]
        print("Dataset loaded and processed.")

        best_hyperparams = search_hyperparameters(texts, tags)
        if not best_hyperparams:
            print("Hyperparameter search failed.")
            return

        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

        print("Initializing model with best hyperparameters...")
        best_model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(CATEGORY_NAMES)).to(DEVICE)

        total_runs = 10
        best_f1 = 0

        for run in range(total_runs):
            print(f"Run {run+1}/{total_runs} with hyperparameters {best_hyperparams}...")
            train_loader, val_loader = create_data_loaders(tokenizer, texts, tags, best_hyperparams['BATCH_SIZE'])
            if train_loader is None or val_loader is None:
                print(f"Skipping run {run+1} due to data loader creation failure.")
                continue

            optimizer = AdamW(best_model.parameters(), lr=best_hyperparams['LEARNING_RATE'])
            total_steps = len(train_loader) * best_hyperparams['EPOCHS']
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            train_model(best_model, train_loader, optimizer, scheduler, best_hyperparams['EPOCHS'])
            _, _, f1 = evaluate_model(best_model, val_loader)

            if f1 > best_f1:
                best_f1 = f1
                try:
                    torch.save(best_model.state_dict(), './models/best_ner_model.pth')
                except Exception as e:
                    print(f"Error saving model: {e}")
            print(f"Run {run+1} completed with F1 Score: {f1:.4f}")

        print(f"Best model saved with F1 Score: {best_f1:.4f}")

        try:
            model_save_path = filedialog.askdirectory(title="Select Model Save Directory")
            if model_save_path:
                best_model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                print(f"Model and tokenizer saved to {model_save_path}")
            else:
                messagebox.showerror("Error", "Model save path not selected. Exiting.")
                return
        except Exception as e:
            print(f"Error during model save dialog: {e}")
            return

        while True:
            try:
                text_input = simpledialog.askstring("Input", "Enter text for NER prediction (or type 'exit' to quit):")
                if not text_input or text_input.lower() == 'exit':
                    print("User exited the prediction loop.")
                    break

                predictions = predict_entities(best_model, tokenizer, text_input)
                result = "\n".join(f"{token}: {entity}" for token, entity in predictions if entity != 'O')

                messagebox.showinfo("NER Predictions", f"Predictions:\n{result}")
                print(f"Predictions for input '{text_input}':\n{result}")
            except Exception as e:
                print(f"Error during prediction input loop: {e}")
                break

        gui_root.mainloop()
    except Exception as e:
        print(f"Error in main program: {e}")

if __name__ == "__main__":
    print("Starting main program...")
    main()
    print("Program completed.")
