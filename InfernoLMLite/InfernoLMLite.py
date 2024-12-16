import os
import logging
import warnings
import torch
import tkinter as tk
from tkinter import filedialog
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import init, Fore, Style

init(autoreset=True)

os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class InfernoLM:
    def __init__(self, device="cuda", precision="float16", model_path=None):
        self.device = device
        self.precision = precision
        self.model_path = model_path
        if model_path:
            self.tokenizer, self.model = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        print(Fore.YELLOW + "Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device=self.device, dtype=torch.float16 if self.precision == "float16" else torch.float32)
        return tokenizer, model

    def chat_with_assistant(self):
        system_prompt = "You are Mistral, a helpful and contextually appropriate A.I. servant developed by Mistral A.I."
        context_history = f"<s>[INST] <<SYS>> {system_prompt} <</SYS>>\nAssistant: I am your A.I. servant created by Mistral A.I. and will do what is asked. What shall I do?\n"
        print(Fore.CYAN + "Assistant: I am your A.I. servant created by Mistral A.I. and will do what is asked. What shall I do?")

        eos_token = '</s>'

        while True:
            user_input = input(Fore.GREEN + "You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break

            context_history += f"[user_message: {user_input}]{eos_token}\n"

            inputs = self.tokenizer.encode_plus(context_history, return_tensors='pt', padding=True, truncation=True, max_length=4096)
            inputs = inputs.to(self.device)

            eos_token_id = self.tokenizer.convert_tokens_to_ids(eos_token)

            try:
                generation_config = {
                    'input_ids': inputs['input_ids'],
                    'max_length': len(inputs['input_ids'][0]) + 500,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'eos_token_id': eos_token_id,
                    'pad_token_id': self.tokenizer.eos_token_id,
                    'do_sample': True,
                    'num_beams': 10,
                    'early_stopping': True
                }
                outputs = self.model.generate(**generation_config)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                assistant_response = self.extract_response(generated_text, user_input)

                assistant_response_formatted = assistant_response.replace("Assistant: Assistant:", "Assistant:")
                print(Fore.CYAN + assistant_response_formatted.strip())

                context_history += f"<s>Assistant: {assistant_response_formatted}{eos_token}\n"

            except Exception as e:
                print(Fore.RED + f"An error occurred during generation: {e}")

    def extract_response(self, generated_text, user_input):
        split_text = generated_text.split(f"[user_message: {user_input}]")
        if len(split_text) > 1:
            response_part = split_text[-1].split("</s>", 1)[0].strip()
            return response_part
        else:
            return "I couldn't generate a proper response."

def select_path():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select the directory containing model and tokenizer files")
    return folder_selected

def display_menu():
    print(Fore.MAGENTA + "\nInfernoLM: The Language Model Inferencer\n")
    print(Fore.YELLOW + "1. Select Model Path")
    print(Fore.YELLOW + "2. Chat with Assistant")
    print(Fore.YELLOW + "3. Exit")

def main():
    print(Fore.MAGENTA + "Welcome to InfernoLM: The Language Model Inferencer!")

    inferencer = InfernoLM(device="cuda", precision="float16")

    while True:
        display_menu()
        choice = input(Fore.GREEN + "Enter your choice: ")
        if choice == "1":
            model_path = select_path()
            if model_path:
                inferencer.model_path = model_path
                inferencer.tokenizer, inferencer.model = inferencer._load_model_and_tokenizer()
            else:
                print(Fore.RED + "No model path selected.")
        elif choice == "2":
            if inferencer.model_path:
                print(Fore.YELLOW + "\nEntering Chatbot Mode. Type 'quit' or 'exit' to leave.")
                inferencer.chat_with_assistant()
            else:
                print(Fore.RED + "Please select a model path first.")
        elif choice == "3":
            print(Fore.MAGENTA + "\nFarewell!")
            break

if __name__ == "__main__":
    main()
