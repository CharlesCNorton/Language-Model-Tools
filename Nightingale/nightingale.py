import os
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageSequence
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import copy
import torch
import tkinter as tk
from tkinter import filedialog
from colorama import Fore, Style, init

init(autoreset=True)

class Nightingale:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image = None
        self.colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
                         'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']

    def init_model(self, model_path):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            print(f"{Fore.GREEN}{Style.BRIGHT}Model loaded successfully from {model_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error loading model: {e}{Style.RESET_ALL}")
            raise

    def run_example(self, task_prompt, image=None, text_input=None):
        try:
            if not self.model or not self.processor:
                raise Exception("Model or processor is not initialized.")

            prompt = task_prompt if text_input is None else task_prompt + text_input

            if image:
                image = image.convert('RGB')
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height) if image else None
            )
            return parsed_answer
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error running example: {e}{Style.RESET_ALL}")
            raise

    def plot_bbox(self, image, data):
        try:
            fig, ax = plt.subplots()
            ax.imshow(image)
            for bbox, label in zip(data['bboxes'], data['labels']):
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
            ax.axis('off')
            plt.show()
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error plotting bounding boxes: {e}{Style.RESET_ALL}")
            raise

    def draw_ocr_bboxes(self, image, prediction):
        try:
            scale = 1
            draw = ImageDraw.Draw(image)
            bboxes, labels = prediction['quad_boxes'], prediction['labels']
            for box, label in zip(bboxes, labels):
                color = random.choice(self.colormap)
                new_box = (np.array(box) * scale).tolist()
                draw.polygon(new_box, width=3, outline=color)
                draw.text((new_box[0] + 8, new_box[1] + 2), "{}".format(label), align="right", fill=color)
            image.show()
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error drawing OCR bounding boxes: {e}{Style.RESET_ALL}")
            raise

    def convert_to_od_format(self, data):
        try:
            bboxes = data.get('bboxes', [])
            labels = data.get('bboxes_labels', [])
            od_results = {'bboxes': bboxes, 'labels': labels}
            return od_results
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error converting to object detection format: {e}{Style.RESET_ALL}")
            raise

    def run_dense_region_caption(self, image):
        try:
            task_prompt = '<DENSE_REGION_CAPTION>'
            results = self.run_example(task_prompt, image=image)
            self.plot_bbox(image, results[task_prompt])
            return results
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error running dense region caption: {e}{Style.RESET_ALL}")
            raise

    def run_object_detection(self, image):
        try:
            task_prompt = '<OD>'
            results = self.run_example(task_prompt, image=image)
            self.plot_bbox(image, results[task_prompt])
            return results
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error running object detection: {e}{Style.RESET_ALL}")
            raise

    def run_phrase_grounding(self, image, phrase):
        try:
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
            results = self.run_example(task_prompt, image=image, text_input=phrase)
            self.plot_bbox(image, results[task_prompt])
            return results
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error running phrase grounding: {e}{Style.RESET_ALL}")
            raise

    def run_open_vocabulary_detection(self, image, phrase):
        try:
            task_prompt = '<OPEN_VOCABULARY_DETECTION>'
            results = self.run_example(task_prompt, image=image, text_input=phrase)
            bbox_results = self.convert_to_od_format(results[task_prompt])
            self.plot_bbox(image, bbox_results)
            return results
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error running open vocabulary detection: {e}{Style.RESET_ALL}")
            raise

    def run_ocr(self, image):
        try:
            task_prompt = '<OCR>'
            results = self.run_example(task_prompt, image=image)
            return results
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error running OCR: {e}{Style.RESET_ALL}")
            raise

    def run_ocr_with_region(self, image):
        try:
            task_prompt = '<OCR_WITH_REGION>'
            results = self.run_example(task_prompt, image=image)
            output_image = copy.deepcopy(image)
            self.draw_ocr_bboxes(output_image, results[task_prompt])
            return results
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error running OCR with region: {e}{Style.RESET_ALL}")
            raise

    def select_model_path(self):
        root = tk.Tk()
        root.withdraw()
        model_path = filedialog.askdirectory()
        if model_path:
            self.init_model(model_path)
        else:
            print(f"{Fore.YELLOW}{Style.BRIGHT}Model path selection cancelled.{Style.RESET_ALL}")

    def select_image_path(self):
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        return image_path

    def main_menu(self):
        while True:
            print(f"\n{Fore.BLUE}{Style.BRIGHT}======= {Fore.MAGENTA}Nightingale {Fore.BLUE}======={Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}1.{Style.RESET_ALL} Select Model Path {Fore.YELLOW}(Point to a Microsoft Florence model){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}2.{Style.RESET_ALL} Select Image {Fore.YELLOW}(Load an image file){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}3.{Style.RESET_ALL} Analyze Image {Fore.YELLOW}(Enter a custom task prompt for analysis){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}4.{Style.RESET_ALL} Dense Region Captioning {Fore.YELLOW}(Generate captions for dense regions in the image){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}5.{Style.RESET_ALL} Object Detection {Fore.YELLOW}(Detect objects and draw bounding boxes){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}6.{Style.RESET_ALL} Phrase Grounding {Fore.YELLOW}(Find phrases within the image context){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}7.{Style.RESET_ALL} Open Vocabulary Detection {Fore.YELLOW}(Detect objects using a provided phrase){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}8.{Style.RESET_ALL} OCR {Fore.YELLOW}(Extract text from the image){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}9.{Style.RESET_ALL} OCR with Region {Fore.YELLOW}(Extract text and highlight regions){Style.RESET_ALL}")
            print(f"{Fore.RED}{Style.BRIGHT}10.{Style.RESET_ALL} Exit")
            print(f"{Fore.BLUE}{Style.BRIGHT}========================={Style.RESET_ALL}")

            choice = input(f"{Fore.YELLOW}{Style.BRIGHT}Select an option: {Style.RESET_ALL}")

            if choice == '1':
                self.select_model_path()
            elif choice == '2':
                self.select_image()
            elif choice == '3':
                self.analyze_image()
            elif choice == '4':
                self.dense_region_captioning()
            elif choice == '5':
                self.object_detection()
            elif choice == '6':
                self.phrase_grounding()
            elif choice == '7':
                self.open_vocabulary_detection()
            elif choice == '8':
                self.ocr()
            elif choice == '9':
                self.ocr_with_region()
            elif choice == '10':
                print(f"{Fore.RED}{Style.BRIGHT}Exiting...{Style.RESET_ALL}")
                break
            else:
                print(f"{Fore.RED}{Style.BRIGHT}Invalid option. Please try again.{Style.RESET_ALL}")

    def select_image(self):
        image_path = self.select_image_path()
        if image_path:
            self.current_image = self.open_image(image_path)
            print(f"{Fore.GREEN}{Style.BRIGHT}Image loaded successfully from {image_path}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}{Style.BRIGHT}Image selection cancelled.{Style.RESET_ALL}")

    def open_image(self, image_path):
        try:
            image = Image.open(image_path)
            if image.format == 'PNG' and image.info.get('transparency', None) is not None:
                image = image.convert('RGBA')
            else:
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error opening image: {e}{Style.RESET_ALL}")
            return None

    def analyze_image(self):
        if not self.current_image:
            self.select_image()
            if not self.current_image:
                return

        while True:
            try:
                task_prompt = input(f"{Fore.CYAN}{Style.BRIGHT}Enter the task prompt (type 'menu' to return to main menu): {Style.RESET_ALL}")
                if task_prompt.lower() == 'menu':
                    break
                results = self.run_example(task_prompt, image=self.current_image)
                print(f"{Fore.GREEN}{Style.BRIGHT}Analysis Results:{Style.RESET_ALL}")
                for key, value in results.items():
                    print(f"{Fore.CYAN}{Style.BRIGHT}{key}:{Style.RESET_ALL} {value}")
            except Exception as e:
                print(f"{Fore.RED}{Style.BRIGHT}Error: {e}{Style.RESET_ALL}")

    def dense_region_captioning(self):
        if not self.current_image:
            print(f"{Fore.YELLOW}{Style.BRIGHT}No image selected. Please select an image first.{Style.RESET_ALL}")
            return

        try:
            print(f"{Fore.BLUE}{Style.BRIGHT}Generating captions for dense regions in the image...{Style.RESET_ALL}")
            results = self.run_dense_region_caption(self.current_image)
            print(f"{Fore.GREEN}{Style.BRIGHT}Analysis Results: {results}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error: {e}{Style.RESET_ALL}")

    def object_detection(self):
        if not self.current_image:
            print(f"{Fore.YELLOW}{Style.BRIGHT}No image selected. Please select an image first.{Style.RESET_ALL}")
            return

        try:
            print(f"{Fore.BLUE}{Style.BRIGHT}Detecting objects and drawing bounding boxes...{Style.RESET_ALL}")
            results = self.run_object_detection(self.current_image)
            print(f"{Fore.GREEN}{Style.BRIGHT}Analysis Results: {results}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error: {e}{Style.RESET_ALL}")

    def phrase_grounding(self):
        if not self.current_image:
            self.select_image()
            if not self.current_image:
                return

        while True:
            try:
                print(f"{Fore.BLUE}{Style.BRIGHT}Finding phrases within the image context...{Style.RESET_ALL}")
                phrase = input(f"{Fore.CYAN}{Style.BRIGHT}Enter the phrase (type 'menu' to return to main menu): {Style.RESET_ALL}")
                if phrase.lower() == 'menu':
                    break
                results = self.run_phrase_grounding(self.current_image, phrase)
                print(f"{Fore.GREEN}{Style.BRIGHT}Analysis Results: {results}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}{Style.BRIGHT}Error: {e}{Style.RESET_ALL}")

    def open_vocabulary_detection(self):
        if not self.current_image:
            self.select_image()
            if not self.current_image:
                return

        while True:
            try:
                print(f"{Fore.BLUE}{Style.BRIGHT}Detecting objects using the provided phrase...{Style.RESET_ALL}")
                phrase = input(f"{Fore.CYAN}{Style.BRIGHT}Enter the phrase (type 'menu' to return to main menu): {Style.RESET_ALL}")
                if phrase.lower() == 'menu':
                    break
                results = self.run_open_vocabulary_detection(self.current_image, phrase)
                print(f"{Fore.GREEN}{Style.BRIGHT}Analysis Results: {results}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}{Style.BRIGHT}Error: {e}{Style.RESET_ALL}")

    def ocr(self):
        if not self.current_image:
            print(f"{Fore.YELLOW}{Style.BRIGHT}No image selected. Please select an image first.{Style.RESET_ALL}")
            return

        try:
            print(f"{Fore.BLUE}{Style.BRIGHT}Extracting text from the image...{Style.RESET_ALL}")
            results = self.run_ocr(self.current_image)
            print(f"{Fore.GREEN}{Style.BRIGHT}Analysis Results: {results}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error: {e}{Style.RESET_ALL}")

    def ocr_with_region(self):
        if not self.current_image:
            print(f"{Fore.YELLOW}{Style.BRIGHT}No image selected. Please select an image first.{Style.RESET_ALL}")
            return

        try:
            print(f"{Fore.BLUE}{Style.BRIGHT}Extracting text and highlighting regions...{Style.RESET_ALL}")
            results = self.run_ocr_with_region(self.current_image)
            print(f"{Fore.GREEN}{Style.BRIGHT}Analysis Results: {results}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    nightingale = Nightingale()
    print(f"{Fore.BLUE}{Style.BRIGHT}Welcome to Nightingale, the image analysis tool powered by Florence!{Style.RESET_ALL}")
    nightingale.main_menu()
