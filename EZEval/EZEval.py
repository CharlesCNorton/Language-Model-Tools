import logging
import os
import tkinter as tk
from tkinter import filedialog
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer
from colorama import init, Fore, Style, Back
from tabulate import tabulate

init()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FileSelector:
    @staticmethod
    def select_directory(title="Select a Directory"):
        print(Fore.YELLOW + Style.BRIGHT + f"Prompting user to select a directory with title: {title}" + Style.RESETALL)
        folder_selected = filedialog.askdirectory(title=title)
        if not folder_selected:
            raise ValueError("No directory selected.")
        print(Fore.GREEN + Style.BRIGHT + f"Directory selected: {folder_selected}" + Style.RESETALL)
        return folder_selected

    @staticmethod
    def select_file(title="Select a File"):
        print(Fore.YELLOW + Style.BRIGHT + f"Prompting user to select a file with title: {title}" + Style.RESETALL)
        file_selected = filedialog.askopenfilename(title=title)
        if not file_selected:
            raise ValueError("No file selected.")
        print(Fore.GREEN + Style.BRIGHT + f"File selected: {file_selected}" + Style.RESETALL)
        return file_selected

class ModelLoader:
    @staticmethod
    def load_tokenizer(model_directory):
        try:
            print(Fore.YELLOW + Style.BRIGHT + f"Loading tokenizer from directory: {model_directory}" + Style.RESETALL)
            tokenizer = AutoTokenizer.from_pretrained(model_directory)
            if tokenizer.pad_token is None:
                print(Fore.YELLOW + Style.BRIGHT + "Pad token not found, adding pad token." + Style.RESETALL)
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.padding_side = "right"
            print(Fore.GREEN + Style.BRIGHT + "Tokenizer loaded successfully." + Style.RESETALL)
            return tokenizer
        except Exception as e:
            logging.error(Fore.RED + Style.BRIGHT + f"Error loading tokenizer: {e}" + Style.RESETALL)
            raise ValueError("Failed to load tokenizer from the specified directory.")

    @staticmethod
    def load_model(model_directory):
        try:
            print(Fore.YELLOW + Style.BRIGHT + f"Loading model from directory: {model_directory}" + Style.RESETALL)
            model = AutoModelForCausalLM.from_pretrained(model_directory).to("cuda")
            print(Fore.GREEN + Style.BRIGHT + "Model loaded successfully." + Style.RESETALL)
            return model
        except Exception as e:
            logging.error(Fore.RED + Style.BRIGHT + f"Error loading model: {e}" + Style.RESETALL)
            raise ValueError("Failed to load model from the specified directory.")

class DatasetLoader:
    @staticmethod
    def load_evaluation_dataset(evaluation_file):
        try:
            print(Fore.YELLOW + Style.BRIGHT + f"Loading evaluation file: {evaluation_file}" + Style.RESETALL)
            evaluation_dataset = load_dataset("text", data_files=evaluation_file)
            print(Fore.GREEN + Style.BRIGHT + "Evaluation file loaded successfully." + Style.RESETALL)
            return evaluation_dataset["train"]
        except FileNotFoundError:
            logging.error(Fore.RED + Style.BRIGHT + f"Evaluation file not found: {evaluation_file}" + Style.RESETALL)
            raise ValueError("Evaluation file not found.")
        except Exception as e:
            logging.error(Fore.RED + Style.BRIGHT + f"Error loading evaluation file: {e}" + Style.RESETALL)
            raise ValueError("Failed to load evaluation file.")

class ModelEvaluator:
    def __init__(self, model_directory, evaluation_dataset):
        self.model_directory = model_directory
        self.evaluation_dataset = evaluation_dataset
        self.tokenizer = None

    def evaluate_model(self):
        try:
            print(Fore.YELLOW + Style.BRIGHT + f"Starting model evaluation for directory: {self.model_directory}" + Style.RESETALL)
            model = ModelLoader.load_model(self.model_directory)
            self.tokenizer = ModelLoader.load_tokenizer(self.model_directory)

            print(Fore.YELLOW + Style.BRIGHT + "Tokenizing evaluation dataset..." + Style.RESETALL)
            tokenized_evaluation_dataset = self.evaluation_dataset.map(
                self.tokenize_function, batched=True
            )
            print(Fore.GREEN + Style.BRIGHT + "Evaluation dataset tokenized successfully." + Style.RESETALL)
            tokenized_evaluation_dataset = tokenized_evaluation_dataset.with_format("torch")

            print(Fore.YELLOW + Style.BRIGHT + "Setting up training arguments for evaluation..." + Style.RESETALL)
            training_args = TrainingArguments(
                output_dir="./eval_results",
                do_train=False,
                do_eval=True,
                per_device_eval_batch_size=1,
                eval_accumulation_steps=1,
            )
            print(Fore.YELLOW + Style.BRIGHT + "Initializing SFTTrainer for evaluation..." + Style.RESETALL)
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                eval_dataset=tokenized_evaluation_dataset,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
                tokenizer=self.tokenizer,
                dataset_text_field="text",
                max_seq_length=4096,
            )
            print(Fore.YELLOW + Style.BRIGHT + "Evaluating the model..." + Style.RESETALL)
            eval_results = trainer.evaluate()
            print(Fore.GREEN + Style.BRIGHT + "Model evaluation completed." + Style.RESETALL)

            del model
            del self.tokenizer
            torch.cuda.empty_cache()

            return eval_results
        except Exception as e:
            logging.error(Fore.RED + Style.BRIGHT + f"Error evaluating the model: {e}" + Style.RESETALL)
            raise ValueError("Failed to evaluate the model.")

    def tokenize_function(self, examples):
        try:
            print(Fore.YELLOW + Style.BRIGHT + "Tokenizing examples..." + Style.RESETALL)
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        except Exception as e:
            logging.error(Fore.RED + Style.BRIGHT + f"Error tokenizing examples: {e}" + Style.RESETALL)
            raise ValueError("Failed to tokenize examples.")

def main_menu():
    evaluation_dataset = None
    model_directories = []
    while True:
        print(Fore.CYAN + Back.BLACK + Style.BRIGHT + "\nModel Evaluation Menu:" + Style.RESETALL)
        print(Fore.CYAN + "1. Select Evaluation File" + Style.RESETALL)
        print(Fore.CYAN + "2. Add Model Directory for Evaluation" + Style.RESETALL)
        if len(model_directories) > 1:
            print(Fore.CYAN + "3. Run Batch Evaluation" + Style.RESETALL)
        else:
            print(Fore.CYAN + "3. Run Evaluation" + Style.RESETALL)
        print(Fore.CYAN + "4. Exit" + Style.RESETALL)
        choice = input(Fore.YELLOW + Style.BRIGHT + "Select an option: " + Style.RESETALL)
        try:
            if choice == "1":
                print(Fore.YELLOW + Style.BRIGHT + "User selected option 1: Select Evaluation File" + Style.RESETALL)
                evaluation_file = FileSelector.select_file("Select Evaluation File")
                evaluation_dataset = DatasetLoader.load_evaluation_dataset(evaluation_file)
            elif choice == "2":
                print(Fore.YELLOW + Style.BRIGHT + "User selected option 2: Add Model Directory for Evaluation" + Style.RESETALL)
                model_directory = FileSelector.select_directory("Select Model Directory")
                model_directories.append(model_directory)
                print(Fore.GREEN + Style.BRIGHT + f"Model directory '{model_directory}' added for evaluation." + Style.RESETALL)
            elif choice == "3":
                print(Fore.YELLOW + Style.BRIGHT + "User selected option 3: Run Evaluation" + Style.RESETALL)
                if evaluation_dataset is None:
                    print(Fore.RED + Style.BRIGHT + "Please select an evaluation file first." + Style.RESETALL)
                elif not model_directories:
                    print(Fore.RED + Style.BRIGHT + "Please add at least one model directory for evaluation." + Style.RESETALL)
                else:
                    results = []
                    for model_directory in model_directories:
                        try:
                            print(Fore.YELLOW + Style.BRIGHT + f"Evaluating model from directory: {model_directory}" + Style.RESETALL)
                            evaluator = ModelEvaluator(model_directory, evaluation_dataset)
                            eval_results = evaluator.evaluate_model()
                            results.append([model_directory] + [f"{value:.4f}" for value in eval_results.values()])
                        except Exception as e:
                            print(Fore.RED + Style.BRIGHT + f"Failed to evaluate model from directory: {model_directory}. Error: {e}" + Style.RESETALL)
                            continue

                    if results:
                        headers = ["Model"] + list(eval_results.keys())
                        min_eval_loss = min(float(result[1]) for result in results)
                        table_data = []
                        for result in results:
                            if float(result[1]) == min_eval_loss:
                                result[1] = Fore.GREEN + result[1] + Style.RESETALL
                            table_data.append(result)

                        table = tabulate(table_data, headers, tablefmt="grid")
                        print(Fore.BLUE + Back.WHITE + Style.BRIGHT + "\nEvaluation Results:" + Style.RESETALL)
                        print(table)
            elif choice == "4":
                print(Fore.GREEN + Style.BRIGHT + "Exiting the program. Goodbye!" + Style.RESETALL)
                break
            else:
                print(Fore.RED + Style.BRIGHT + "Invalid choice. Please try again." + Style.RESETALL)
        except ValueError as e:
            print(Fore.RED + Style.BRIGHT + str(e) + Style.RESETALL)
        except Exception as e:
            logging.error(Fore.RED + Style.BRIGHT + f"An unexpected error occurred: {e}" + Style.RESETALL)
            print(Fore.RED + Style.BRIGHT + "An unexpected error occurred. Please check the logs for more details." + Style.RESETALL)

if __name__ == "__main__":
    print(Fore.MAGENTA + Back.WHITE + Style.BRIGHT + "Welcome to the Model Evaluation Program!" + Style.RESETALL)
    main_menu()
