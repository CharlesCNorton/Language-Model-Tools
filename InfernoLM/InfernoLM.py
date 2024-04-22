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
import re

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
    def __init__(self, device="cuda", precision="float16", model_path=None, tts_enabled=False, trust_remote_code=False, debug_mode=False, stop_strings=None):
        self.device = device
        self.precision = precision
        self.model_path = model_path
        self.tts_enabled = tts_enabled
        self.trust_remote_code = trust_remote_code
        self.debug_mode = debug_mode
        self.stop_strings = stop_strings or []
        self.model_loaded = False
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
            print(Fore.RED + "No model path selected.")
            return
        print(Fore.YELLOW + "Loading model and tokenizer...")
        try:
            if self.debug_mode:
                print(Fore.MAGENTA + "Debug: Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
            if self.debug_mode:
                print(Fore.MAGENTA + "Debug: Tokenizer loaded successfully.")
                print(Fore.MAGENTA + "Debug: Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float16 if self.precision == "float16" else torch.float32,
                low_cpu_mem_usage=True
            )
            if self.debug_mode:
                print(Fore.MAGENTA + "Debug: Model loaded successfully.")
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.debug_mode:
                print(Fore.MAGENTA + "Debug: Moving model to device...")
            self.model.to(self.device)
            if self.debug_mode:
                print(Fore.MAGENTA + "Debug: Model moved to device successfully.")
            self.model_loaded = True
            print(Fore.GREEN + "Model and tokenizer loaded successfully.")
        except Exception as e:
            print(Fore.RED + f"Error loading model and tokenizer: {str(e)}")
            self.model_loaded = False

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

    def chat_with_assistant(self, use_audio=False):
        if not self.model_loaded:
            print(Fore.RED + "Model not loaded. Please load a model first.")
            return
        system_prompt = "You are a helpful A.I. assistant!"
        context_history = f"{system_prompt}\nAssistant: Hello! I am your A.I. assistant. How may I help you?\n"
        print(Fore.CYAN + "Assistant: Hello! How may I help you?")
        eos_token = ''
        if use_audio:
            self.setup_audio_input(context_history, eos_token)
        else:
            self.setup_text_input(context_history, eos_token)

    def setup_text_input(self, context_history, eos_token):
        while True:
            user_input = input(Fore.GREEN + "You: ").strip().lower()
            if user_input in ["quit", "exit", "bye"]:
                return
            assistant_response, context_history = self.process_user_input(user_input, context_history, eos_token)
            print(Fore.CYAN + f"Assistant: {assistant_response}")

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
            assistant_response, context_history = self.process_user_input(user_input, context_history, eos_token)
            print(Fore.CYAN + f"Assistant: {assistant_response}")

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
        generation_config = {'input_ids': inputs['input_ids'], 'max_length': len(inputs['input_ids'][0]) + 500, 'temperature': 0.6,
                             'top_p': 0.9, 'top_k': 50, 'eos_token_id': eos_token_id, 'pad_token_id': self.tokenizer.eos_token_id,
                             'do_sample': True, 'num_beams': 10, 'early_stopping': True}
        try:
            if self.debug_mode:
                print(Fore.MAGENTA + "Debug: Generating response...")
            outputs = self.model.generate(**generation_config)
            if self.debug_mode:
                print(Fore.MAGENTA + f"Debug: Generated output: {outputs}")
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if self.debug_mode:
                print(Fore.MAGENTA + f"Debug: Generated text: {generated_text}")
            assistant_response = self.extract_response(generated_text, user_input)
            if self.debug_mode:
                print(Fore.MAGENTA + f"Debug: Extracted response: {assistant_response}")
            self.speak(assistant_response.strip())
            context_history += f"<s>Assistant: {assistant_response}{eos_token}\n"
            return assistant_response.strip(), context_history
        except Exception as e:
            print(Fore.RED + f"Error generating response: {str(e)}")
            return "I couldn't generate a proper response.", context_history

    def extract_response(self, generated_text, user_input):
        split_text = generated_text.split(f"[user_message: {user_input}]")
        if len(split_text) > 1:
            response_part = split_text[-1].strip()
            response_part = re.sub(r'<[^>]*>', '', response_part)
            response_part = re.sub(r'\n', ' ', response_part)
            response_part = re.sub(r'\s+', ' ', response_part)
            if response_part.startswith("</s>"):
                response_part = response_part[4:].strip()
            if response_part.startswith("<s>"):
                response_part = response_part[3:].strip()
            user_message_tags = re.findall(r'\[user_message:.*?]', response_part)
            if user_message_tags:
                response_part = response_part.split(user_message_tags[0])[0].strip()
            else:
                stop_string_found = False
                for stop_string in self.stop_strings:
                    if stop_string in response_part:
                        response_part = response_part.split(stop_string)[0].strip()
                        stop_string_found = True
                        break
                if not stop_string_found:
                    response_part = response_part.strip()
            while response_part.startswith("Assistant: "):
                response_part = response_part[11:].strip()
            return response_part
        else:
            return "I couldn't generate a proper response."

    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        print(Fore.YELLOW + f"Debug mode has been {'enabled' if self.debug_mode else 'disabled'}.")

def select_path():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select the directory containing model and tokenizer files")
    return folder_selected

def display_menu(inferencer):
    print(Fore.WHITE + "\nInfernoLM: The Language Model Inferencer\n")
    print(Fore.YELLOW + "1. Select Model Path")
    print(Fore.YELLOW + "2. Chat with Assistant (Text Mode)")
    print(Fore.YELLOW + "3. Chat with Assistant (Voice Mode)")
    print(Fore.YELLOW + f"4. Toggle Text-to-Speech (Currently: {Fore.GREEN if inferencer.tts_enabled else Fore.RED}{inferencer.tts_enabled})")
    print(Fore.YELLOW + "5. Unload Model")
    print(Fore.YELLOW + f"6. Toggle Trust Remote Code (Currently: {Fore.GREEN if inferencer.trust_remote_code else Fore.RED}{inferencer.trust_remote_code})")
    print(Fore.YELLOW + f"7. Toggle Debug Mode (Currently: {Fore.GREEN if inferencer.debug_mode else Fore.RED}{inferencer.debug_mode})")
    print(Fore.YELLOW + "8. Set Stop Strings")
    print(Fore.YELLOW + "9. Exit")

def set_stop_strings(inferencer):
    print(Fore.YELLOW + "Enter stop strings (comma-separated):")
    stop_strings = input(Fore.GREEN + "> ").strip().split(",")
    inferencer.stop_strings = [s.strip() for s in stop_strings if s.strip()]
    print(Fore.YELLOW + f"Stop strings set to: {inferencer.stop_strings}")

def main():
    trust_remote_code = False
    debug_mode = False
    stop_strings = []
    inferencer = InfernoLM(trust_remote_code=trust_remote_code, debug_mode=debug_mode, stop_strings=stop_strings)

    while True:
        display_menu(inferencer)
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
            inferencer.trust_remote_code = not inferencer.trust_remote_code
        elif choice == "7":
            inferencer.toggle_debug_mode()
        elif choice == "8":
            set_stop_strings(inferencer)
        elif choice == "9":
            print(Fore.MAGENTA + "\nFarewell!")
            break

if __name__ == "__main__":
    main()
