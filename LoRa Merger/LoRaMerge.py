import os
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LoraMerge:
    def __init__(self, trust_remote_code=False):
        self.trust_remote_code = trust_remote_code

    def display_menu(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        logging.info("Displaying main menu")
        print("========= LoraMerge: Merge PEFT Adapters with a Base Model =========")
        print("\nOptions:")
        print("1. Merge models")
        print("2. Acknowledgment & Citation")
        print(f"3. Toggle Trust Remote Code (Currently: {self.trust_remote_code})")
        print("4. Exit")
        choice = input("\nEnter your choice: ")
        return choice

    def select_directory(self, title="Select a directory"):
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory(title=title)
        if not folder_selected:
            messagebox.showerror("Error", f"{title} cancelled or failed. Exiting.")
            logging.error(f"Directory selection failed or cancelled for {title}")
            exit()
        logging.info(f"Directory selected: {folder_selected}")
        return folder_selected

    def merge_models(self, args):
        try:
            base_model_name_or_path = self.select_directory("Select pretrained directory for base model")
            peft_model_path = self.select_directory("Select pretrained directory for PEFT model")
            output_dir = self.select_directory("Select directory to save the model")
            device_arg = {'device_map': 'auto'} if args.device == 'auto' else {'device_map': {"": args.device}}
            logging.info(f"Loading base model from {base_model_name_or_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                return_dict=True,
                torch_dtype=torch.float16,
                trust_remote_code=self.trust_remote_code,
                **device_arg
            )
            logging.info(f"Loading PEFT model from {peft_model_path}")
            model = PeftModel.from_pretrained(base_model, peft_model_path, **device_arg)
            logging.info("Merging models")
            model = model.merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=self.trust_remote_code)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved to {output_dir}")
        except Exception as e:
            logging.error(f"Failed to merge models: {str(e)}")
            messagebox.showerror("Error", f"Failed to merge models: {str(e)}")

    def display_acknowledgment(self):
        print("\nAcknowledgment & Citation:")
        print("\nLoraMerge is based on an original script by TheBloke available as a Gist at:")
        print("https://gist.github.com/TheBloke/d31d289d3198c24e0ca68aaf37a19032")
        print("\nWe deeply appreciate the foundational work by TheBloke and aim to extend its usability with this tool.")
        logging.info("Displayed acknowledgment and citation")

    def run(self):
        args = self.get_args()
        while True:
            choice = self.display_menu()
            if choice == '1':
                self.merge_models(args)
                input("\nPress Enter to continue...")
            elif choice == '2':
                self.display_acknowledgment()
                input("\nPress Enter to continue...")
            elif choice == '3':
                self.trust_remote_code = not self.trust_remote_code
                logging.info(f"Trust remote code set to {self.trust_remote_code}")
            elif choice == '4':
                logging.info("Exiting application")
                break
            else:
                logging.warning("Invalid choice")
                input("Invalid choice. Press Enter to continue...")

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="Merge PEFT Adapters with a Base Model.")
        parser.add_argument("--device", type=str, default="auto", help="Device for model loading, e.g. 'cuda:0', 'cpu'. 'auto' to auto-select.")
        return parser.parse_args()


if __name__ == "__main__":
    LoraMerge().run()
