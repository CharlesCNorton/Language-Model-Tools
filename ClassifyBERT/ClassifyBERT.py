import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import os
from tqdm.auto import tqdm

class BertSummarizer:
    def __init__(self, model_name="bert-large-cased", device=None):
        self.model_name = model_name
        self.device = device if device else self.setup_device()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(self.device)
        print("Model and tokenizer are set up.")

    def setup_device(self):
        if torch.cuda.is_available():
            print("Using CUDA device.")
            return torch.device("cuda")
        else:
            print("CUDA is not available. Using CPU.")
            return torch.device("cpu")

    def load_and_prepare_data(self):
        try:
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            print("Dataset loaded successfully.")

            reduced_dataset = DatasetDict({
                "train": dataset["train"].shuffle(seed=42).select(range(int(dataset["train"].num_rows * 0.1))),
                "validation": dataset["validation"].shuffle(seed=42).select(range(int(dataset["validation"].num_rows * 0.1)))
            })
            return reduced_dataset.map(self.tokenize_function, batched=True)
        except Exception as e:
            print(f"Failed to load or process dataset: {e}")
            raise

    def tokenize_function(self, examples):
        return self.tokenizer(examples['article'], truncation=True, padding="max_length", max_length=512)

    def train_model(self, dataset):
        try:
            training_args = TrainingArguments(
                output_dir="./bert_finetuned_summarization",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_dir='./logs',
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
            )

            trainer.train()
            print("Fine-tuning completed successfully.")
            return trainer
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_model(self):
        self.model.save_pretrained("./saved_model")
        self.tokenizer.save_pretrained("./saved_model")
        print("Model and tokenizer have been saved.")

    def load_model(self):
        try:
            self.model = BertForSequenceClassification.from_pretrained("./saved_model").to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained("./saved_model")
            print("Model and tokenizer loaded from disk.")
        except Exception as e:
            print(f"Error loading model from saved files: {e}")
            raise

    def generate_summary(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            return self.tokenizer.decode(outputs.logits.argmax(-1), skip_special_tokens=True)
        except Exception as e:
            print(f"Error generating summary: {e}")
            raise

def main():
    summarizer = BertSummarizer()
    try:
        dataset = summarizer.load_and_prepare_data()
    except Exception as e:
        print("Exiting due to data loading error.")
        return

    with tqdm(total=3, desc="Epoch Progress") as pbar:
        try:
            trainer = summarizer.train_model(dataset)
            pbar.update(3)
        except Exception as e:
            print("Exiting due to training error.")
            return

    summarizer.save_model()

    print("Enter some text to summarize or 'exit' to quit:")
    while True:
        input_text = input("> ")
        if input_text.lower() == "exit":
            break
        try:
            summary = summarizer.generate_summary(input_text)
            print(f"Generated Summary: {summary}")
        except Exception as e:
            print(f"Error during summarization: {e}")

if __name__ == "__main__":
    main()
