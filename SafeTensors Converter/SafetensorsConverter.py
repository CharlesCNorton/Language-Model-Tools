import os
import re
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from safetensors.torch import load_file, save_file

selected_directory = ""

def is_valid_pytorch_file(filename: str) -> bool:
    return filename.endswith('.bin')

def check_file_size(sf_filename: str, pt_filename: str) -> bool:
    try:
        sf_size = os.stat(sf_filename).st_size
        pt_size = os.stat(pt_filename).st_size
        if abs(sf_size - pt_size) / pt_size > 0.01:
            raise ValueError(
                f"File size difference more than 1%: {sf_filename} = {sf_size}, {pt_filename} = {pt_size}"
            )
    except OSError as e:
        messagebox.showerror("File Size Check Error", f"OS error: {e}")
        return False
    except ValueError as e:
        messagebox.showerror("File Size Check Error", str(e))
        return False
    return True

def load_pytorch_file(pt_filename: str):
    try:
        return torch.load(pt_filename, map_location="cpu")
    except FileNotFoundError:
        messagebox.showerror("Load Error", f"File not found: {pt_filename}")
        return None
    except Exception as e:
        messagebox.showerror("Load Error", f"Error loading file {pt_filename}: {e}")
        return None

def save_safetensor_file(loaded, sf_filename: str):
    try:
        os.makedirs(os.path.dirname(sf_filename), exist_ok=True)
        save_file(loaded, sf_filename, metadata={"format": "pt"})
    except Exception as e:
        messagebox.showerror("Save Error", f"Error saving file {sf_filename}: {e}")
        return False
    return True

def validate_tensor_integrity(loaded, sf_filename: str):
    try:
        reloaded = load_file(sf_filename)
        for k in loaded:
            pt_tensor = loaded[k]
            sf_tensor = reloaded[k]
            if not torch.equal(pt_tensor, sf_tensor):
                raise ValueError(f"Tensor mismatch for key {k}")
    except Exception as e:
        messagebox.showerror("Validation Error", f"Error validating file {sf_filename}: {e}")
        return False
    return True

def determine_sf_filename(directory: str, filename: str) -> str:
    match = re.match(r"pytorch_model-(\d+)-of-(\d+).bin", filename)
    if match:
        part_num, total_parts = match.groups()
        return os.path.join(directory, f"model-{part_num.zfill(5)}-of-{total_parts.zfill(5)}.safetensors")
    elif filename == "pytorch_model.bin":
        return os.path.join(directory, "model.safetensors")
    return ""
def convert_file(pt_filename: str, sf_filename: str):
    loaded = load_pytorch_file(pt_filename)
    if loaded is None:
        return False
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    loaded = {k: v.contiguous() for k, v in loaded.items()}
    if not save_safetensor_file(loaded, sf_filename):
        return False
    if not check_file_size(sf_filename, pt_filename):
        return False
    if not validate_tensor_integrity(loaded, sf_filename):
        return False
    return True

def convert_all_files_in_directory(directory: str):
    if not os.path.isdir(directory):
        messagebox.showerror("Directory Error", f"Invalid directory: {directory}")
        return
    for filename in os.listdir(directory):
        if not is_valid_pytorch_file(filename):
            continue
        pt_filename = os.path.join(directory, filename)
        sf_filename = determine_sf_filename(directory, filename)
        if sf_filename:
            if not convert_file(pt_filename, sf_filename):
                messagebox.showerror("Conversion Error", f"Failed to convert {filename}")

def select_directory():
    global selected_directory
    selected_directory = filedialog.askdirectory()
    if selected_directory:
        directory_label.config(text=f"Selected Directory: {selected_directory}")

def start_conversion():
    if selected_directory:
        try:
            convert_all_files_in_directory(selected_directory)
            messagebox.showinfo("Success", "Files converted successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("No Directory Selected", "Please select a directory first.")

def main():
    global directory_label

    root = tk.Tk()
    root.title("Bin to SafeTensors Converter")

    frame = ttk.Frame(root, padding="30")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    title_label = ttk.Label(frame, text="Bin to SafeTensors Converter", font=("Arial", 16))
    title_label.grid(row=0, column=0, pady=20, sticky=tk.W)

    directory_label = ttk.Label(frame, text="Selected Directory: None", font=("Arial", 12))
    directory_label.grid(row=1, column=0, pady=10, sticky=tk.W)

    select_button = ttk.Button(frame, text="Select Directory", command=select_directory)
    select_button.grid(row=2, column=0, pady=10)

    convert_button = ttk.Button(frame, text="Convert", command=start_conversion)
    convert_button.grid(row=3, column=0, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
