import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering, get_scheduler
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tkinter import Tk, filedialog, simpledialog, messagebox
import random
import logging
from tqdm import tqdm

MODEL_NAME = 'bert-large-cased'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 384
HYPERPARAMETER_OPTIONS = {
    'BATCH_SIZE': [8, 16, 32],
    'LEARNING_RATE': [2e-5, 4e-5, 6e-5],
    'EPOCHS': [1, 2, 3]
}
LOGGING_LEVEL = logging.INFO

logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

class QADataset(Dataset):
    def __init__(self, tokenizer, contexts, questions, answers):
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        logger.info(f"Dataset initialized with {len(self.contexts)} examples.")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]

        inputs = self.tokenizer(
            question, context,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        start_positions = inputs.char_to_token(0, answer['answer_start'])
        end_positions = inputs.char_to_token(0, answer['answer_start'] + len(answer['text']) - 1)

        if start_positions is None:
            start_positions = MAX_LEN - 1
        if end_positions is None:
            end_positions = MAX_LEN - 1

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'start_positions': torch.tensor(start_positions, dtype=torch.long),
            'end_positions': torch.tensor(end_positions, dtype=torch.long)
        }

def create_data_loaders(tokenizer, contexts, questions, answers, batch_size):
    logger.info("Splitting data into training and validation sets...")
    train_contexts, val_contexts, train_questions, val_questions, train_answers, val_answers = train_test_split(
        contexts, questions, answers, test_size=0.1, random_state=42
    )
    logger.info(f"Data split into {len(train_contexts)} training and {len(val_contexts)} validation examples.")

    train_dataset = QADataset(tokenizer, train_contexts, train_questions, train_answers)
    val_dataset = QADataset(tokenizer, val_contexts, val_questions, val_answers)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    logger.info("Data loaders created and ready for training and validation.")
    return train_loader, val_loader

def train_model(model, train_loader, optimizer, scheduler, epochs):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    logger.info("Starting training...")

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}...")
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_index, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_positions = batch['start_positions'].to(DEVICE)
            end_positions = batch['end_positions'].to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
                epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}, Learning Rate: {current_lr:.2e}")

    logger.info("Training process completed.")

def evaluate_model(model, data_loader, dataset_type="validation"):
    model.eval()
    all_start_preds = []
    all_end_preds = []
    all_start_labels = []
    all_end_labels = []
    logger.info(f"Starting {dataset_type} evaluation...")

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"{dataset_type.capitalize()} Evaluation")
        for batch_index, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_positions = batch['start_positions'].to(DEVICE)
            end_positions = batch['end_positions'].to(DEVICE)

            try:
                outputs = model(input_ids, attention_mask=attention_mask)
                start_preds = torch.argmax(outputs.start_logits, dim=-1)
                end_preds = torch.argmax(outputs.end_logits, dim=-1)

                all_start_labels.extend(start_positions.cpu().numpy())
                all_end_labels.extend(end_positions.cpu().numpy())
                all_start_preds.extend(start_preds.cpu().numpy())
                all_end_preds.extend(end_preds.cpu().numpy())

                if batch_index % 10 == 0:
                    logger.info(f"Processed {batch_index+1}/{len(data_loader)} batches for {dataset_type} evaluation.")
            except RuntimeError as e:
                logger.error(f"Error during evaluation at batch {batch_index+1}: {e}")
                continue

    start_precision, start_recall, start_f1, _ = precision_recall_fscore_support(
        all_start_labels, all_start_preds, average='macro', zero_division=0)
    end_precision, end_recall, end_f1, _ = precision_recall_fscore_support(
        all_end_labels, all_end_preds, average='macro', zero_division=0)

    logger.info(f"{dataset_type.capitalize()} Start Precision: {start_precision:.4f}, Recall: {start_recall:.4f}, F1 Score: {start_f1:.4f}")
    logger.info(f"{dataset_type.capitalize()} End Precision: {end_precision:.4f}, Recall: {end_recall:.4f}, F1 Score: {end_f1:.4f}")

    return (start_precision + end_precision) / 2, (start_recall + end_recall) / 2, (start_f1 + end_f1) / 2

def search_hyperparameters(contexts, questions, answers):
    logger.info("Starting hyperparameter search...")
    best_f1 = 0
    best_hyperparams = {}
    total_trials = 1

    for trial in range(total_trials):
        batch_size = random.choice(HYPERPARAMETER_OPTIONS['BATCH_SIZE'])
        learning_rate = random.choice(HYPERPARAMETER_OPTIONS['LEARNING_RATE'])
        epochs = random.choice(HYPERPARAMETER_OPTIONS['EPOCHS'])

        logger.info(f"Trial {trial+1}/{total_trials}: Batch Size={batch_size}, Learning Rate={learning_rate}, Epochs={epochs}")

        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

        train_loader, val_loader = create_data_loaders(tokenizer, contexts, questions, answers, batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
        logger.info(f"Trial {trial+1} completed with F1 Score: {f1:.4f}")

    logger.info(f"Best hyperparameters found: {best_hyperparams} with F1 Score: {best_f1:.4f}")
    return best_hyperparams

def predict_answer(model, tokenizer, context, question):
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, max_length=MAX_LEN)
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    if start_index > end_index:
        end_index = start_index

    answer_tokens = input_ids[0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

def main():
    gui_root = Tk()
    gui_root.withdraw()

    try:
        logger.info("Loading dataset...")
        dataset = load_dataset("squad", split='train')
        contexts = dataset['context']
        questions = dataset['question']
        answers = [{'text': ans['text'][0], 'answer_start': ans['answer_start'][0]} for ans in dataset['answers']]
        logger.info("Dataset loaded and processed.")

        best_hyperparams = search_hyperparameters(contexts, questions, answers)
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

        logger.info("Initializing model with best hyperparameters...")
        best_model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

        total_runs = 3
        best_f1 = 0

        for run in range(total_runs):
            logger.info(f"Run {run+1}/{total_runs} with hyperparameters {best_hyperparams}...")
            train_loader, val_loader = create_data_loaders(tokenizer, contexts, questions, answers, best_hyperparams['BATCH_SIZE'])
            optimizer = torch.optim.AdamW(best_model.parameters(), lr=best_hyperparams['LEARNING_RATE'])  # Use AdamW from PyTorch
            total_steps = len(train_loader) * best_hyperparams['EPOCHS']
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            train_model(best_model, train_loader, optimizer, scheduler, best_hyperparams['EPOCHS'])
            _, _, f1 = evaluate_model(best_model, val_loader)

            if f1 > best_f1:
                best_f1 = f1
                torch.save(best_model.state_dict(), './models/best_qa_model.pth')
            logger.info(f"Run {run+1} completed with F1 Score: {f1:.4f}")

        logger.info(f"Best model saved with F1 Score: {best_f1:.4f}")

        model_save_path = filedialog.askdirectory(title="Select Model Save Directory")
        if model_save_path:
            best_model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"Model and tokenizer saved to {model_save_path}")
        else:
            messagebox.showerror("Error", "Model save path not selected. Exiting.")
            return

        while True:
            context = simpledialog.askstring("Input", "Enter context for QA prediction (or type 'exit' to quit):")
            if not context or context.lower() == 'exit':
                logger.info("User exited the prediction loop.")
                break

            question = simpledialog.askstring("Input", "Enter question for QA prediction (or type 'exit' to quit):")
            if not question or question.lower() == 'exit':
                logger.info("User exited the prediction loop.")
                break

            try:
                answer = predict_answer(best_model, tokenizer, context, question)
                messagebox.showinfo("QA Predictions", f"Answer:\n{answer}")
                logger.info(f"Predictions for context '{context}' and question '{question}':\n{answer}")
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                messagebox.showerror("Error", f"Error during prediction: {e}")

        gui_root.mainloop()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    logger.info("Starting main program...")
    main()
    logger.info("Program completed.")

if __name__ == "__main__":
    logger.info("Starting main program...")
    main()
    logger.info("Program completed.")
