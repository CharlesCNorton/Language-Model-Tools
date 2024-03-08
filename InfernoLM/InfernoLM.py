import logging
import os
import warnings
import tkinter as tk
from tkinter import filedialog
import torch
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import init, Fore, Style
import pyttsx3
import keyboard
import time
init(autoreset=True)
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
fs = 44100
is_recording = False
whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
class InfernoLM:
    def __init__(self, device="cuda", precision="float16", model_path=None, tts_enabled=False):
       self.device = device
       self.precision = precision
       self.model_path = model_path
       self.tts_enabled = tts_enabled
       self.model_loaded = False
       if self.model_path:
           self.load_model()
       if self.tts_enabled:
           try:
               self.tts_engine = pyttsx3.init()
               print("TTS engine initialized.")
           except Exception as e:
               print(f"Error initializing TTS engine: {str(e)}")
               self.tts_engine = None
       else:
           self.tts_engine = None
    def load_model(self):
        if not self.model_path:
            print(Fore.RED + "Model path not set. Please select a model path first.")
            return
        print(Fore.YELLOW + "Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(device=self.device, dtype=torch.float16 if self.precision == "float16" else torch.float32)
        self.model_loaded = True
    def unload_model(self):
        if self.model_loaded:
            print(Fore.YELLOW + "Unloading model...")
            del self.model
            del self.tokenizer
            self.model_loaded = False
        else:
            print(Fore.RED + "No model is currently loaded.")
    def speak(self, text):
        if self.tts_enabled and self.tts_engine is not None:
           try:
               self.tts_engine.say(text)
               self.tts_engine.runAndWait()
               print("TTS finished speaking.")
           except Exception as e:
               print(f"Error during TTS speaking: {str(e)}")
        else:
           print("TTS is not enabled or initialized.")

    def chat_with_assistant(self, use_audio=False):
        if not self.model_loaded:
            print(Fore.RED + "Model not loaded. Please load a model first.")
            return
        system_prompt = "You are a helpful A.I. assistant."
        context_history = f"<s>[INST] <<SYS>> {system_prompt} <</SYS>>\nAssistant: I am your A.I. servant. How may I help you?\n"
        print(Fore.CYAN + "Assistant: How may I help you?")
        eos_token = '</s>'
        if use_audio:
            self.setup_audio_input(context_history, eos_token)
        else:
            self.setup_text_input(context_history, eos_token)

    def setup_text_input(self, context_history, eos_token):
        while True:
            user_input = input(Fore.GREEN + "You: ").strip().lower()
            if user_input in ["quit", "exit", "bye"]:
                return
            self.process_user_input(user_input, context_history, eos_token)
    def setup_audio_input(self, context_history, eos_token):
        print(Fore.CYAN + "You can now speak to the assistant. Press 'space' to start and then 'space' again stop recording.")
        while True:
            print(Fore.GREEN + "Press 'space' to start recording.")
            keyboard.wait('space')
            print(Fore.YELLOW + "Recording... press 'space' to stop.")
            recording = sd.rec(int(10 * fs), samplerate=fs, channels=2, dtype='int16', blocking=True)
            print(Fore.GREEN + "Processing audio...")
            user_input = self.transcribe_audio(recording)
            print(Fore.GREEN + "You (Transcribed): " + user_input)
            if user_input.lower() in ["quit", "exit", "bye"]:
                return
            self.process_user_input(user_input, context_history, eos_token)

    def transcribe_audio(self, recording):
        text = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix='recorded_', dir=os.getcwd()) as tmpfile:
                write(tmpfile.name, fs, recording)
                tmpfile_path = tmpfile.name
            text = whisper_model.transcribe(tmpfile_path)["text"]
        finally:
            time.sleep(1)
            os.unlink(tmpfile_path)

        return text

    def process_user_input(self, user_input, context_history, eos_token):
        context_history += f"[user_message: {user_input}]{eos_token}\n"
        inputs = self.tokenizer.encode_plus(context_history, return_tensors='pt', padding=True, truncation=True, max_length=4096)
        inputs = inputs.to(self.device)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(eos_token)
        generation_config = {'input_ids': inputs['input_ids'], 'max_length': len(inputs['input_ids'][0]) + 500, 'temperature': 0.7,
                             'top_p': 0.9, 'top_k': 50, 'eos_token_id': eos_token_id, 'pad_token_id': self.tokenizer.eos_token_id,
                             'do_sample': True, 'num_beams': 10, 'early_stopping': True}
        outputs = self.model.generate(**generation_config)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = self.extract_response(generated_text, user_input)
        assistant_response_formatted = assistant_response.replace("Assistant: Assistant:", "Assistant:")
        print(Fore.CYAN + assistant_response_formatted.strip())
        self.speak(assistant_response_formatted.strip())
        context_history += f"<s>Assistant: {assistant_response_formatted}{eos_token}\n"
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
    print(Fore.WHITE + "\nInfernoLM: The Language Model Inferencer\n")
    print(Fore.YELLOW + "1. Select Model Path")
    print(Fore.YELLOW + "2. Chat with Assistant (Text Mode)")
    print(Fore.YELLOW + "3. Chat with Assistant (Voice Mode)")
    print(Fore.YELLOW + "4. Toggle Text-to-Speech")
    print(Fore.YELLOW + "5. Unload Model")
    print(Fore.YELLOW + "6. Exit")
def main():
    inferencer = InfernoLM()
    while True:
        display_menu()
        choice = input(Fore.GREEN + "Enter your choice: ")
        if choice == "1":
            model_path = select_path()
            if model_path:
                inferencer.model_path = model_path
                inferencer.load_model()
            else:
                print(Fore.RED + "No model path selected.")
        elif choice == "2":
            inferencer.chat_with_assistant(use_audio=False)
        elif choice == "3":
            inferencer.chat_with_assistant(use_audio=True)
        elif choice == "4":
            inferencer.tts_enabled = not inferencer.tts_enabled
            state = "enabled" if inferencer.tts_enabled else "disabled"
            print(Fore.YELLOW + f"Text-to-Speech has been {state}.")
            if inferencer.tts_enabled:
                try:
                    inferencer.tts_engine = pyttsx3.init()
                    print("TTS initialized.")
                except Exception as e:
                    print(f"Error initializing TTS: {str(e)}")
                    inferencer.tts_engine = None
            else:
                inferencer.tts_engine = None
        elif choice == "5":
            inferencer.unload_model()
        elif choice == "6":
            print(Fore.MAGENTA + "\nFarewell!")
            break
if __name__ == "__main__":
    main()
