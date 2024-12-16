import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, AdamW, pipeline
from sklearn.metrics import accuracy_score, f1_score
import random

class BertFineTuner:
    def __init__(self, model_name='bert-large-uncased', load_pretrained_model=False, model_dir=None):
        print("Initializing the model and tokenizer from the model name provided...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        if load_pretrained_model and model_dir:
            print(f"Loading model from {model_dir}")
            self.model = BertForMaskedLM.from_pretrained(model_dir).to(self.device)
        else:
            self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        print("Initialization complete.")

    def load_and_prepare_data(self):
        print("Attempting to load and preprocess the dataset...")
        try:
            dataset = load_dataset("yahma/alpaca-cleaned")
            print("Dataset loaded successfully.")
            dataset = dataset.map(self.preprocess_function, batched=True)
            train_test_split = dataset['train'].train_test_split(test_size=0.1)
            val_test_split = train_test_split['test'].train_test_split(test_size=0.5)
            self.train_dataset = train_test_split['train']
            self.val_dataset = val_test_split['train']
            self.test_dataset = val_test_split['test']
            print("Data preprocessing and splitting complete.")
        except Exception as e:
            print(f"An unexpected error occurred while loading or preprocessing the dataset: {e}")

    def preprocess_function(self, examples):
        print("Preprocessing data...")
        prompts = ["[CLS] " + inst + (" [SEP] " + inp if inp else "[SEP]") for inst, inp in zip(examples['instruction'], examples.get('input', [''] * len(examples['instruction'])))]
        inputs = self.tokenizer(prompts, max_length=512, truncation=True, padding="max_length")
        outputs = self.tokenizer(examples['output'], max_length=512, truncation=True, padding="max_length")
        inputs['labels'] = outputs['input_ids']
        print("Batch of data preprocessed.")
        return inputs

    def train_model(self, config):
        if not self.train_dataset or not self.val_dataset:
            print("Training and evaluation datasets are not available. Please load data first.")
            return

        print(f"Setting up training configuration with {config}...")
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=config['learning_rate'],
            per_device_train_batch_size=config['batch_size'],
            num_train_epochs=config['epochs'],
            weight_decay=config['weight_decay'],
            load_best_model_at_end=True,
            logging_dir='./logs',
            logging_steps=50,
        )

        optimizer = AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        print("Optimizer initialized with the training configuration.")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, None)
        )

        print("Starting the training process...")
        trainer.train()
        print("Training completed successfully.")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        f1 = f1_score(labels.flatten(), predictions.flatten(), average='weighted')
        metrics = {"accuracy": accuracy, "f1": f1}
        return metrics

    def evaluate(self, eval_dataset):
        if not eval_dataset:
            print("Evaluation dataset is not available.")
            return None
        print("Initiating evaluation of the model...")
        trainer = Trainer(
            model=self.model,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        result = trainer.evaluate()
        print(f"Evaluation result: {result}")
        return result

    def inference(self, text):
        print("Performing inference on provided text...")
        nlp = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, device=0 if self.device == 'cuda' else -1)
        result = nlp(text)
        print(f"Inference result: {result}")
        return result

    def save_model(self, output_dir='./model'):
        print(f"Saving model to the directory: {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model successfully saved.")

    def hyperparameter_search(self, num_trials):
        print("Commencing hyperparameter search...")
        best_loss = float('inf')
        best_config = None
        for _ in range(num_trials):
            config = {
                'learning_rate': 5e-5 * random.choice([0.2, 0.6, 1, 1.5, 2]),
                'batch_size': random.choice([4, 8, 16]),
                'epochs': random.choice([2, 3, 4]),
                'weight_decay': random.choice([0.01, 0.05, 0.1]),
            }
            print(f"Evaluating configuration: {config}")
            self.train_model(config)
            eval_results = self.evaluate(self.val_dataset)
            if eval_results and 'eval_loss' in eval_results and eval_results['eval_loss'] < best_loss:
                best_loss = eval_results['eval_loss']
                best_config = config
                print(f"Found new best configuration with eval loss: {best_loss}")
        print(f"Best configuration determined: {best_config}")
        return best_config

    def run(self):
        while True:
            print("\nInteractive Menu:")
            print("1: Load and Prepare Data")
            print("2: Train Model")
            print("3: Advanced Model Training with Hyperparameter Search")
            print("4: Evaluate Model")
            print("5: Perform Inference")
            print("6: Save Model")
            print("7: Exit Program")
            choice = input("Enter your choice (1-7): ")
            if choice == '1':
                self.load_and_prepare_data()
            elif choice == '2':
                config = {
                    'learning_rate': float(input("Enter learning rate (e.g., 5e-5): ")),
                    'batch_size': int(input("Enter batch size (e.g., 8): ")),
                    'epochs': int(input("Enter number of epochs (e.g., 3): ")),
                    'weight_decay': float(input("Enter weight decay (e.g., 0.01): "))
                }
                self.train_model(config)
            elif choice == '3':
                num_trials = int(input("Enter number of trials for hyperparameter search: "))
                self.hyperparameter_search(num_trials)
            elif choice == '4':
                self.evaluate(self.val_dataset)
            elif choice == '5':
                text = input("Enter text for inference: ")
                self.inference(text)
            elif choice == '6':
                output_dir = input("Enter output directory to save the model: ")
                self.save_model(output_dir)
            elif choice == '7':
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 7.")

if __name__ == '__main__':
    fine_tuner = BertFineTuner()
    fine_tuner.run()
