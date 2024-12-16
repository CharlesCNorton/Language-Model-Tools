import logging
import os
import warnings
import sys
import threading
import time
import re
import tkinter as tk
from tkinter import filedialog
import torch
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
from colorama import init, Fore, Style, Back
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import pyttsx3
import keyboard
import whisper

init(autoreset=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='inferno_lm.log',
    filemode='a'
)

os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings("ignore")

fs = 44100

try:
    whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Whisper model: {str(e)}")
    whisper_model = None


class StopOnUserPrompt(StoppingCriteria):
    """
    Custom stopping criteria to halt generation when the user prompt is detected.
    """

    def __init__(self, user_prompt_id):
        super().__init__()
        self.user_prompt_id = user_prompt_id

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[-1] < len(self.user_prompt_id):
            return False

        if list(input_ids[0][-len(self.user_prompt_id):].cpu().numpy()) == self.user_prompt_id:
            return True
        return False


class InfernoLM:
    """
    InfernoLM is a language model inferencer that allows users to interact with
    an AI assistant via text or voice. It supports various settings and can
    save/load conversation history.
    """

    def __init__(self, device="cuda", precision="float16", model_path=None,
                 tts_enabled=False, trust_remote_code=False, debug_mode=False, stop_strings=None):
        self.device = device
        self.precision = precision
        self.model_path = model_path
        self.tts_enabled = tts_enabled
        self.trust_remote_code = trust_remote_code
        self.debug_mode = debug_mode
        self.stop_strings = stop_strings or []
        self.model_loaded = False
        self.is_llama = False
        self.context_history = ""
        self.session_active = False
        self.user_name = "User"
        self.assistant_name = "Assistant"
        self.tts_engine = None

        if self.tts_enabled:
            self.initialize_tts()

        self.logger = logging.getLogger(__name__)

    def initialize_tts(self):
        """Initializes the text-to-speech engine."""
        try:
            self.tts_engine = pyttsx3.init()
            self.logger.info("TTS engine initialized.")
        except Exception as e:
            self.logger.error(f"Error initializing TTS engine: {str(e)}")
            self.tts_engine = None
            self.tts_enabled = False

    def load_model(self):
        """Loads the language model and tokenizer."""
        if not self.model_path:
            self.logger.error("No model path selected.")
            print(Fore.RED + "No model path selected. Please select a valid model path.")
            return

        print(Fore.YELLOW + "Loading model and tokenizer...")

        try:
            if self.debug_mode:
                self.logger.debug("Loading tokenizer...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )

            if not self.tokenizer.pad_token:
                self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
                if self.debug_mode:
                    self.logger.debug("Added new pad token '<PAD>'.")
            else:
                if self.debug_mode:
                    self.logger.debug("Pad token already exists.")

            if self.debug_mode:
                self.logger.debug("Tokenizer loaded successfully.")
                self.logger.debug("Loading model...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float16 if self.precision == "float16" else torch.float32,
                low_cpu_mem_usage=True
            )

            if self.debug_mode:
                self.logger.debug("Model loaded successfully.")

            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.tokenizer.pad_token = self.tokenizer.pad_token

            if self.is_llama:
                if "llama" not in self.model.config.model_type.lower():
                    self.logger.warning("The loaded model may not be a LLaMA model.")

            if self.debug_mode:
                self.logger.debug("Moving model to device...")
            self.model.to(self.device)
            if self.debug_mode:
                self.logger.debug("Model moved to device successfully.")

            if not self.tokenizer.pad_token:
                self.model.resize_token_embeddings(len(self.tokenizer))
                if self.debug_mode:
                    self.logger.debug("Resized token embeddings to accommodate new pad token.")

            self.model_loaded = True
            print(Fore.GREEN + "Model and tokenizer loaded successfully.")

        except Exception as e:
            self.logger.error(f"Error loading model and tokenizer: {str(e)}")
            print(Fore.RED + f"Error loading model: {str(e)}")

            print(Fore.YELLOW + "Would you like to try again? (y/n)")
            retry_choice = input(Fore.GREEN + "> ").strip().lower()

            if retry_choice == "y":

                self.load_model()
            else:
                print(Fore.CYAN + "Returning to the main menu...")
                self.model_loaded = False

    def unload_model(self):
        """Unloads the model and frees up resources."""
        if self.model_loaded:
            print(Fore.YELLOW + "Unloading model...")
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model_loaded = False
            print(Fore.GREEN + "Model unloaded successfully.")
        else:
            self.logger.warning("No model is currently loaded.")

    def speak(self, text):
        """Speaks the given text using TTS if enabled."""
        if self.tts_enabled and self.tts_engine is not None:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                if self.debug_mode:
                    self.logger.debug("TTS finished speaking.")
            except Exception as e:
                self.logger.error(f"Error during TTS speaking: {str(e)}")

    def chat_with_assistant(self, use_audio=False):
        """Initiates a chat session with the assistant."""
        if not self.model_loaded:
            self.logger.error("Model not loaded. Please load a model first.")
            return
        self.session_active = True
        system_prompt = f"You are {self.assistant_name}, a helpful A.I. assistant!"
        self.context_history = f"{system_prompt}\n{self.assistant_name}: Hello! How may I assist you today?\n"
        print(Fore.CYAN + f"{self.assistant_name}: Hello! How may I assist you today?")
        if use_audio:
            self.setup_audio_input()
        else:
            self.setup_text_input()

    def setup_text_input(self):
        """Handles text input from the user."""
        while self.session_active:
            user_input = input(Fore.GREEN + f"{self.user_name}: ").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                print(Fore.MAGENTA + "Exiting chat mode.")
                self.session_active = False
                return
            assistant_response = self.process_user_input(user_input)
            print(Fore.CYAN + f"{self.assistant_name}: {assistant_response}")

    def setup_audio_input(self):
        """Handles audio input from the user."""
        print(Fore.CYAN + "You can now speak to the assistant. Press 'space' to start and then 'space' again to stop recording.")
        while self.session_active:
            print(Fore.GREEN + "Press 'space' to start recording.")
            keyboard.wait('space')
            print(Fore.YELLOW + "Recording... press 'space' to stop.")
            recording = []
            rec_thread = threading.Thread(target=self.record_audio, args=(recording,))
            rec_thread.start()
            keyboard.wait('space')
            # Stop recording
            sd.stop()
            rec_thread.join()
            print(Fore.GREEN + "Processing audio...")
            user_input = self.transcribe_audio(recording[0])
            print(Fore.GREEN + f"{self.user_name} (Transcribed): {user_input}")
            if user_input.lower() in ["quit", "exit", "bye"]:
                print(Fore.MAGENTA + "Exiting chat mode.")
                self.session_active = False
                return
            assistant_response = self.process_user_input(user_input)
            print(Fore.CYAN + f"{self.assistant_name}: {assistant_response}")

    def record_audio(self, recording_list):
        """Records audio from the user."""
        try:
            duration = 10
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            recording_list.append(recording)
            sd.wait()
        except Exception as e:
            self.logger.error(f"Error during audio recording: {str(e)}")
            recording_list.append(None)

    def transcribe_audio(self, recording):
        """Transcribes the recorded audio using Whisper."""
        text = ""
        tmpfile_path = None
        try:
            if recording is None:
                return "I couldn't record your audio."
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix='recorded_', dir=os.getcwd()) as tmpfile:
                write(tmpfile.name, fs, recording)
                tmpfile_path = tmpfile.name
            transcription = whisper_model.transcribe(tmpfile_path)
            text = transcription["text"]
        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}")
            text = "I couldn't transcribe your audio."
        finally:
            time.sleep(1)
            if tmpfile_path and os.path.exists(tmpfile_path):
                os.unlink(tmpfile_path)
        return text.strip()

    def process_user_input(self, user_input):
        """Processes the user's input and generates a response."""
        self.context_history += f"{self.user_name}: {user_input}\n"
        MAX_HISTORY = 10
        lines = self.context_history.strip().split('\n')
        if len(lines) > MAX_HISTORY * 2 + 1:
            lines = lines[-(MAX_HISTORY * 2 + 1):]
            self.context_history = '\n'.join(lines) + '\n'

        inputs = self.tokenizer.encode_plus(
            self.context_history,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=4096
        )
        inputs = inputs.to(self.device)

        eos_token_id = self.tokenizer.eos_token_id

        stop_user_prompt = f"{self.user_name}: "
        stop_user_prompt_ids = self.tokenizer.encode(stop_user_prompt, add_special_tokens=False)
        stopping_criteria = StoppingCriteriaList([StopOnUserPrompt(stop_user_prompt_ids)])

        generation_config = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'max_length': len(inputs['input_ids'][0]) + 1000,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'eos_token_id': eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'do_sample': True,
            'num_beams': 5,
            'early_stopping': True,
            'stopping_criteria': stopping_criteria
        }

        try:
            if self.debug_mode:
                self.logger.debug("Generating response...")
                self.logger.debug(f"Generation config: {generation_config}")
            outputs = self.model.generate(**generation_config)
            if self.debug_mode:
                self.logger.debug(f"Generated output: {outputs}")
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if self.debug_mode:
                self.logger.debug(f"Generated text: {generated_text}")
            assistant_response = self.extract_response(generated_text)
            if self.debug_mode:
                self.logger.debug(f"Extracted response: {assistant_response}")
            self.speak(assistant_response.strip())
            self.context_history += f"{self.assistant_name}: {assistant_response}\n"
            return assistant_response.strip()
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I couldn't generate a response."

    def extract_response(self, generated_text):
        """Extracts the assistant's response from the generated text."""
        assistant_prefix = f"{self.assistant_name}: "
        user_prefix = f"{self.user_name}: "

        last_assistant_idx = generated_text.rfind(assistant_prefix)
        if last_assistant_idx == -1:
            return "I'm sorry, I couldn't generate a response."

        response_part = generated_text[last_assistant_idx + len(assistant_prefix):].strip()

        next_user_idx = response_part.find(user_prefix)
        if next_user_idx != -1:
            response_part = response_part[:next_user_idx].strip()

        response_part = re.sub(r'<[^>]*>', '', response_part)
        response_part = re.sub(r'\n', ' ', response_part)
        response_part = re.sub(r'\s+', ' ', response_part)

        if response_part.startswith("</s>"):
            response_part = response_part[4:].strip()
        if response_part.startswith("<s>"):
            response_part = response_part[3:].strip()

        if response_part.startswith(f"{self.assistant_name}: "):
            response_part = response_part[len(f"{self.assistant_name}: "):].strip()

        for stop_string in self.stop_strings:
            if stop_string in response_part:
                response_part = response_part.split(stop_string)[0].strip()
                break

        return response_part if response_part else "I'm sorry, I couldn't generate a response."

    def toggle_debug_mode(self):
        """Toggles the debug mode on or off."""
        self.debug_mode = not self.debug_mode
        print(Fore.YELLOW + f"Debug mode has been {'enabled' if self.debug_mode else 'disabled'}.")

    def toggle_llama_mode(self):
        """Toggles LLaMA mode on or off."""
        self.is_llama = not self.is_llama
        print(Fore.YELLOW + f"LLaMA Mode has been {'enabled' if self.is_llama else 'disabled'}.")

    def set_stop_strings(self, stop_strings):
        """Sets the stop strings used during generation."""
        self.stop_strings = stop_strings
        print(Fore.YELLOW + f"Stop strings set to: {self.stop_strings}")

    def toggle_tts(self):
        """Toggles text-to-speech on or off."""
        self.tts_enabled = not self.tts_enabled
        if self.tts_enabled:
            self.initialize_tts()
            print(Fore.GREEN + "Text-to-Speech has been enabled.")
        else:
            self.tts_engine = None
            print(Fore.YELLOW + "Text-to-Speech has been disabled.")

    def change_names(self):
        """Allows the user to change their name and the assistant's name."""
        print(Fore.YELLOW + "Enter your name:")
        self.user_name = input(Fore.GREEN + "> ").strip() or "User"
        print(Fore.YELLOW + "Enter assistant's name:")
        self.assistant_name = input(Fore.GREEN + "> ").strip() or "Assistant"
        print(Fore.YELLOW + f"Names updated. You are '{self.user_name}' and the assistant is '{self.assistant_name}'.")

    def save_context(self):
        """Saves the conversation history to a file."""
        if self.context_history:
            try:
                with open("conversation_history.txt", "w", encoding='utf-8') as file:
                    file.write(self.context_history)
                print(Fore.GREEN + "Conversation history saved to 'conversation_history.txt'.")
            except Exception as e:
                self.logger.error(f"Error saving conversation history: {str(e)}")
        else:
            print(Fore.RED + "No conversation history to save.")

    def load_context(self):
        """Loads the conversation history from a file."""
        if os.path.exists("conversation_history.txt"):
            try:
                with open("conversation_history.txt", "r", encoding='utf-8') as file:
                    self.context_history = file.read()
                print(Fore.GREEN + "Conversation history loaded.")
            except Exception as e:
                self.logger.error(f"Error loading conversation history: {str(e)}")
        else:
            print(Fore.RED + "No saved conversation history found.")


def select_path():
    """Opens a dialog for the user to select a model path."""
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select the directory containing model and tokenizer files")
    return folder_selected


def display_menu(inferencer):
    """Displays the main menu."""
    print(Back.BLUE + Fore.WHITE + Style.BRIGHT + "\n InfernoLM: The Language Model Inferencer \n" + Style.RESET_ALL)
    print(Fore.YELLOW + Style.BRIGHT + "\n=== Model Management ===")
    print(Fore.CYAN + "1. Select Model Path")
    print("2. Unload Model")
    print(Fore.YELLOW + Style.BRIGHT + "\n=== Settings ===")
    print(Fore.CYAN + f"3. Toggle Trust Remote Code      (Currently: {Fore.GREEN if inferencer.trust_remote_code else Fore.RED}{inferencer.trust_remote_code}{Fore.CYAN})")
    print(f"4. Toggle LLaMA Mode             (Currently: {Fore.GREEN if inferencer.is_llama else Fore.RED}{inferencer.is_llama}{Fore.CYAN})")
    print(f"5. Toggle Text-to-Speech         (Currently: {Fore.GREEN if inferencer.tts_enabled else Fore.RED}{inferencer.tts_enabled}{Fore.CYAN})")
    print(f"6. Toggle Debug Mode             (Currently: {Fore.GREEN if inferencer.debug_mode else Fore.RED}{inferencer.debug_mode}{Fore.CYAN})")
    print("7. Set Stop Strings")
    print("8. Change Names")
    print(Fore.YELLOW + Style.BRIGHT + "\n=== Interaction ===")
    print(Fore.CYAN + "9. Chat with Assistant (Text Mode)")
    print("10. Chat with Assistant (Voice Mode)")
    print("11. Save Conversation History")
    print("12. Load Conversation History")
    print(Fore.YELLOW + Style.BRIGHT + "\n=== Exit ===")
    print(Fore.CYAN + "13. Exit\n")


def set_stop_strings(inferencer):
    """Allows the user to set stop strings."""
    print(Fore.YELLOW + "Enter stop strings (comma-separated):")
    stop_strings_input = input(Fore.GREEN + "> ").strip()
    stop_strings = [s.strip() for s in stop_strings_input.split(",") if s.strip()]
    inferencer.set_stop_strings(stop_strings)


def main():
    """Main function to run the InfernoLM program."""
    trust_remote_code = False
    debug_mode = False
    stop_strings = []

    inferencer = InfernoLM(trust_remote_code=trust_remote_code, debug_mode=debug_mode, stop_strings=stop_strings)

    while True:
        display_menu(inferencer)
        choice = input(Fore.GREEN + Style.BRIGHT + "Enter your choice: ").strip()

        if choice == "1":
            model_path = select_path()
            if model_path:
                inferencer.model_path = model_path
                inferencer.load_model()
            else:
                print(Fore.RED + "No model path selected.")
        elif choice == "2":
            inferencer.unload_model()
        elif choice == "3":
            inferencer.trust_remote_code = not inferencer.trust_remote_code
            print(Fore.YELLOW + f"Trust Remote Code has been {'enabled' if inferencer.trust_remote_code else 'disabled'}.")
        elif choice == "4":
            inferencer.toggle_llama_mode()
        elif choice == "5":
            inferencer.toggle_tts()
        elif choice == "6":
            inferencer.toggle_debug_mode()
        elif choice == "7":
            set_stop_strings(inferencer)
        elif choice == "8":
            inferencer.change_names()
        elif choice == "9":
            inferencer.chat_with_assistant(use_audio=False)
        elif choice == "10":
            inferencer.chat_with_assistant(use_audio=True)
        elif choice == "11":
            inferencer.save_context()
        elif choice == "12":
            inferencer.load_context()
        elif choice == "13":
            print(Fore.MAGENTA + "\nFarewell!")
            inferencer.unload_model()
            sys.exit()
        else:
            print(Fore.RED + "Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()
