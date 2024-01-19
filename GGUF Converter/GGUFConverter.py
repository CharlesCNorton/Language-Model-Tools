import os
import subprocess
from tkinter import Tk, filedialog, messagebox

LLAMA_CPP_DIR = ""

def validate_path(input_path):
    return os.path.exists(input_path)

def set_llama_cpp_dir():
    global LLAMA_CPP_DIR
    root = Tk()
    root.withdraw()
    selected_directory = filedialog.askdirectory(title="Select llama.cpp Directory")
    root.destroy()
    if selected_directory:
        LLAMA_CPP_DIR = selected_directory
        print(f"llama.cpp directory set to: {LLAMA_CPP_DIR}")
    else:
        print("Invalid or no directory selected. Exiting...")

def transpose_hf_to_gguf(root, input_dir, outfile, outtype):
    if not (validate_path(input_dir) and validate_path(os.path.dirname(outfile))):
        print("Invalid input or output file path.")
        return
    cmd = ['python', os.path.join(LLAMA_CPP_DIR, "convert.py"), input_dir, '--outfile', outfile, '--outtype', outtype]
    print(f"Starting conversion: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        error_message = f"Error during conversion: {e}"
        print(error_message)
        messagebox.showerror("Conversion Error", error_message)

def display_help_menu():
    print("\nHelp Menu: HuggingFace to GGUF Converter")
    print("1. GGUF Format: Optimized for Large Language Models.")
    print("2. This Tool: Converts HuggingFace models to GGUF, offering 8-bit, 16-bit, and 32-bit quantization.")
    print("3. Quantization Tradeoffs:")
    print("   - 8-bit: Faster but less accurate.")
    print("   - 16-bit: Balanced.")
    print("   - 32-bit: Original quality, larger size. Avoid upsizing from 16-bit.")
    print("4. Usage: Set directory, choose model and quantization, then convert.")
    print("Refer to llama.cpp and GGUF docs for details.")

def main():
    root = Tk()
    root.withdraw()
    set_llama_cpp_dir()
    if not LLAMA_CPP_DIR:
        print("Required directories not selected. Exiting...")
        return
    while True:
        print("\nHuggingFace to GGUF Converter Menu:")
        print("1. Convert Local HF Model to GGUF")
        print("2. Help")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            folder_selected = filedialog.askdirectory(parent=root, title="Select Local HuggingFace Model Directory")
            outfile = filedialog.asksaveasfilename(parent=root, title="Select Output GGUF File", defaultextension=".gguf")
            if folder_selected and outfile and validate_path(folder_selected) and validate_path(os.path.dirname(outfile)):
                print("Available quantization types: 8-bit (q8_0), 16-bit (f16), 32-bit (f32)")
                outtype = input("Choose quantization type (1/2/3): ")
                quant_types = {"1": "q8_0", "2": "f16", "3": "f32"}
                if outtype in quant_types:
                    transpose_hf_to_gguf(root, folder_selected, outfile, quant_types[outtype])
                else:
                    print("Invalid choice!")
            else:
                print("Invalid input or output file path.")
        elif choice == "2":
            display_help_menu()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice!")
    root.destroy()

if __name__ == "__main__":
    main()
