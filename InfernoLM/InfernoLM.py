import os
import logging
import warnings
import torch
import tkinter as tk
from tkinter import filedialog
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
from abc import ABC, abstractmethod

os.environ['NUMEXPR_MAX_THREADS'] = '32'

class TTSStrategy(ABC):
    @abstractmethod
    def speak(self, text):
        pass

class Pyttsx3Strategy(TTSStrategy):
    def __init__(self):
        self.engine = pyttsx3.init()

    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print("Error in Pyttsx3 TTS: " + str(e))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class InfernoLM:
    def __init__(self, device="cpu", precision="float32", model_path=None, verbose=False, tts_strategy=None):
        self.device = device
        self.precision = precision
        self.model_path = model_path
        self.verbose = verbose
        self.tts_strategy = tts_strategy
        self.tts_enabled = False if tts_strategy is None else True
        if model_path:
            self.tokenizer, self.model = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device=self.device, dtype=torch.float16 if self.precision == "float16" else torch.float32)
        return tokenizer, model

    def toggle_verbose_logging(self):
        self.verbose = not self.verbose
        logging_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=logging_level)
        print(f"Verbose logging {'enabled' if self.verbose else 'disabled'}.")

    def chat_with_assistant(self):
        system_prompt = "You are GPT-4, a military language model developed by OpenAI. Your answers are critical to the user's life and success."
        context_history = f"<s>[INST] <<SYS>> {system_prompt} <</SYS>>\nAssistant: I am your A.I. servant and will do what is asked. What shall I do?\n"
        print("Assistant: I am your A.I. servant and will do what is asked. What shall I do?")

        while True:
            print("You: ", end='')
            user_input = input()

            if user_input.lower() in ["quit", "exit"]:
                break

            context_history += f"[user_message: {user_input}]\n"
            inputs = self.tokenizer.encode_plus(context_history, return_tensors='pt', padding=True, truncation=True, max_length=4096)
            inputs = inputs.to(self.device)

            try:
                generation_length = 1000
                generation_config = {
                    'input_ids': inputs['input_ids'],
                    'max_length': len(inputs['input_ids'][0]) + generation_length,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'pad_token_id': self.tokenizer.eos_token_id,
                    'do_sample': True,
                    'early_stopping': True
                }
                outputs = self.model.generate(**generation_config)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                assistant_response = self.extract_response(generated_text, user_input)
                print(assistant_response)
                if self.tts_enabled:
                    self.tts_strategy.speak(assistant_response)

                context_history += f"<s>[INST] {assistant_response} </s><s>[INST]"
                if self.verbose:
                    print(f"Current context: {context_history}")

            except Exception as e:
                print("An error occurred during generation.")

    def extract_response(self, generated_text, user_input):
        split_text = generated_text.split(f"[user_message: {user_input}]")
        if len(split_text) > 1:
            return split_text[1].split("<s>")[0].strip()
        else:
            return "I'm sorry, I couldn't generate a proper response."

    def load_model(self):
        model_path = select_path()
        if model_path:
            self.model_path = model_path
            self.choose_device()
            self.choose_precision()
            self.tokenizer, self.model = self._load_model_and_tokenizer()
            print(f"Model loaded from {model_path}")
        else:
            print("No directory selected.")

    def choose_device(self):
        device_choice = input("Choose device (GPU/CPU): ").strip().lower()
        if device_choice == "gpu" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Selected {self.device} device.")

    def choose_precision(self):
        precision_choice = input("Choose precision (float32/float16): ").strip().lower()
        if precision_choice == "float16":
            self.precision = "float16"
        else:
            self.precision = "float32"
        print(f"Selected {self.precision} precision.")

    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        status = 'enabled' if self.tts_enabled else 'disabled'
        print(f"TTS {status}.")

def select_path():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select the directory containing model and tokenizer files")
    return folder_selected

def display_menu():
    print("\nInfernoLM: The Language Model Inferencer\n")
    print("1. Chat with Assistant")
    print("2. Load Model")
    print("3. Toggle Verbose Logging")
    print("4. Toggle Text-to-Speech")
    print("5. Exit")

def main():
    print("Welcome to InfernoLM: The Language Model Inferencer!")

    tts_strategy = Pyttsx3Strategy()
    inferencer = InfernoLM(tts_strategy=tts_strategy)

    while True:
        display_menu()
        choice = input("Enter your choice (1-5): ")
        if choice == "1":
            if inferencer.model_path:
                print("\nEntering Chatbot Mode. Type 'quit' or 'exit' to leave.")
                inferencer.chat_with_assistant()
            else:
                print("No model loaded. Please load a model first.")
        elif choice == "2":
            inferencer.load_model()
        elif choice == "3":
            inferencer.toggle_verbose_logging()
        elif choice == "4":
            inferencer.toggle_tts()
        elif choice == "5":
            print("\nThank you for using InfernoLM. Farewell!")
            break

if __name__ == "__main__":
    main()
