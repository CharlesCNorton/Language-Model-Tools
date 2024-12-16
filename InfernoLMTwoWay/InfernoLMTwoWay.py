import os
import logging
import warnings
import torch
import tkinter as tk
from tkinter import filedialog
from transformers import AutoTokenizer, AutoModelForCausalLM
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import tempfile
import keyboard
import time

os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
fs = 44100
recording = None
is_recording = False

class InfernoLM:
    def __init__(self, device="cuda", precision="float16", model_path="D:\\text-generation-webui\\models\\LLama-3-8B"):
        self.device = device
        self.precision = precision
        self.model_path = "D:\\text-generation-webui\\models\\LLama-3-8B"
        if model_path:
            self.tokenizer, self.model = self._load_model_and_tokenizer()
        self.whisper_model = whisper.load_model("medium.en", device=self.device)
        self.is_recording = False
        self.recording = None

    def _load_model_and_tokenizer(self):
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device=self.device, dtype=torch.float16 if self.precision == "float16" else torch.float32)
        return tokenizer, model

    def setup_recording(self):
        keyboard.add_hotkey('space', lambda: self.toggle_recording())
        while not self.is_recording:
            time.sleep(0.1)
        while self.is_recording:
            time.sleep(0.1)
        keyboard.clear_all_hotkeys()

    def toggle_recording(self):
        if not self.is_recording:
            self.recording = sd.rec(int(10 * fs), samplerate=fs, channels=2, dtype='int16', blocking=False)
            self.is_recording = True
        else:
            sd.stop()
            self.is_recording = False

    def save_recording(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix='recorded_', dir=os.getcwd()) as tmpfile:
            write(tmpfile.name, fs, self.recording)
            return tmpfile.name

    def transcribe_with_whisper(self, file_path):
        result = self.whisper_model.transcribe(file_path)
        return result["text"]

    def chat_with_assistant(self, user_input, context_history):
        system_prompt = ""
        if not context_history:
            context_history = f""

        eos_token = '</s>'
        context_history += f"[user_message: {user_input}]{eos_token}\n"
        inputs = self.tokenizer.encode_plus(context_history, return_tensors='pt', padding=True, truncation=True, max_length=4096)
        inputs = inputs.to(self.device)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(eos_token)

        try:
            generation_config = {
                'input_ids': inputs['input_ids'],
                'max_length': len(inputs['input_ids'][0]) + 500,
                'temperature': 0.8,
                'top_p': 0.9,
                'top_k': 50,
                'eos_token_id': eos_token_id,
                'pad_token_id': self.tokenizer.eos_token_id,
                'do_sample': True,
                'num_beams': 15,
                'early_stopping': True
            }
            outputs = self.model.generate(**generation_config)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = self.extract_response(generated_text, user_input, eos_token)
            print(f"Assistant: {assistant_response}")
            context_history += f"Assistant: {assistant_response}{eos_token}\n"
        except Exception as e:
            print(f"An error occurred during generation: {e}")

        return context_history

    def extract_response(self, generated_text, user_input, eos_token):
        split_text = generated_text.split(f"[user_message: {user_input}]")
        if len(split_text) > 1:
            response_part = split_text[-1].split("[user_message:", 1)[0].strip()
            return response_part
        else:
            return "I couldn't generate a proper response."

def select_path():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select the directory containing model and tokenizer files")
    return folder_selected

def display_menu():
    print("\nInfernoLM: The Language Model Inferencer\n")
    print("1. Chat with Assistant (Text Input)")
    print("2. Chat with Assistant (Voice Input)")
    print("3. Exit")

def main():
    print("Welcome to InfernoLM: The Language Model Inferencer!")
    inferencer = InfernoLM(device="cuda", precision="float16", model_path="D:\\text-generation-webui\\models\\Mistral-7b-Instruct-v0.2")
    context_history = ""

    while True:
        display_menu()
        choice = input("Enter your choice: ")
        if choice == "1":
            print("\nEntering Chatbot Mode (Text Input). Type 'quit' or 'exit' to leave.")
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ["quit", "exit"]:
                    break
                context_history = inferencer.chat_with_assistant(user_input, context_history)
        elif choice == "2":
            print("\nEntering Chatbot Mode (Voice Input). Press 'q' to leave.")
            while True:
                print("Press the space key to record your message and then press it again when you are finished.")
                inferencer.setup_recording()
                if inferencer.recording is not None:
                    file_path = inferencer.save_recording()
                    user_input = inferencer.transcribe_with_whisper(file_path)
                    print(f"Transcribed: {user_input}")
                    if user_input.lower() == "q":
                        break
                    context_history = inferencer.chat_with_assistant(user_input, context_history)
        elif choice == "3":
            print("\nFarewell!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
