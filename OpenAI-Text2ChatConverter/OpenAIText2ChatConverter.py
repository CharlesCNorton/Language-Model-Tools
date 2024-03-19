import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import openai
import json
import logging
import re
import asyncio
import aiohttp
import threading

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('chatml_conversion_detailed.log'), logging.StreamHandler()])

stop_event = threading.Event()

def get_selected_model():
    return model_var.get()

async def async_process_text(session, text_chunk, custom_system_message, model_name, mode):
    system_messages = {
        "Q&A": "You are an expert Q&A generator who converts text chunks into high-quality Q&A pairs in the format Q: and A:.",
        "Summary": "You are an expert summarizer who provides concise and informative summaries of the provided text.",
        "Transcription": "You are an expert transcriber who transcribes the provided text with spelling and formatting corrections."
    }

    user_prompts = {
        "Q&A": f"Develop a high quality Q&A pair from the given text chunk:\n{text_chunk}",
        "Summary": f"Summarize the following text:\n{text_chunk}",
        "Transcription": f"Transcribe the following text with spelling corrections and modern longform paragraph formatting, but without annotating and without inserting non-text related strings:\n{text_chunk}"
    }

    system_message = custom_system_message if custom_system_message else system_messages.get(mode, "")
    user_prompt = user_prompts.get(mode, "")

    try:
        response = await session.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.8,
                "max_tokens": 4096
            },
            headers={
                "Authorization": f"Bearer {openai.api_key}"
            }
        )
        response_status = response.status
        response_text = await response.text()
        logging.debug(f"OpenAI API Response Status: {response_status}")
        logging.debug(f"OpenAI API Full Response: {response_text}")
        data = await response.json()
        if data.get('choices') and data['choices'][0]['message']['content'].strip() != '':
            processed_content = data['choices'][0]['message']['content'].strip()
            logging.info(f"Generated Content: {processed_content}")
            return processed_content
        else:
            logging.warning("Received empty or invalid response from OpenAI API.")
            return None
    except Exception as e:
        logging.error(f"An error occurred with OpenAI API: {e}")
        return None

def separate_qa_pairs(content):
    try:
        q_and_a_pairs = re.findall(
            r'Q:\s*(.*?)\nA:\s*(.*?)(?=Q:|\Z)', content, re.DOTALL)
        chatml_content = []

        for question, answer in q_and_a_pairs:
            question = question.strip()
            answer = answer.strip()
            if question and answer:
                chatml_content.append({"role": "user", "content": question})
                chatml_content.append({"role": "assistant", "content": answer})
                logging.debug(f"Extracted Q&A: Q: '{question}', A: '{answer}'")
            else:
                logging.warning(f"Empty or incomplete Q&A pair found: Q: '{question}', A: '{answer}'")

        return chatml_content

    except Exception as e:
        logging.error(f"Error in separate_qa_pairs: {e}")
        return []

content_lock = threading.Lock()

def format_content_for_mode(content, mode):
    if mode == "Q&A":
        return separate_qa_pairs(content)
    else:
        return [{"role": "user", "content": content}]

async def convert_to_chatml(input_file_path, output_file_path, custom_system_message, progress_var, root, model_name, mode, max_chunk_size):
    logging.info("Starting conversion to ChatML format.")

    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            logging.info(f"Reading text from file: {input_file_path}")
            text = file.read()
            logging.info("Text file read successfully.")

        try:
            max_chunk_size = int(max_chunk_size)
            if max_chunk_size <= 0:
                raise ValueError("Max chunk size must be a positive integer.")
        except ValueError as e:
            logging.error(f"Invalid max_chunk_size: {e}")
            messagebox.showerror("Error", f"Invalid max chunk size: {e}")
            return

        formatted_content = []
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        logging.info(f"Text divided into {len(chunks)} chunks.")

        progress_var.set(0)
        root.update_idletasks()

        async with aiohttp.ClientSession() as session:
            for chunk_index, chunk in enumerate(chunks):
                if stop_event.is_set():
                    logging.info("Stop requested by user. Halting conversion.")
                    break

                logging.info(f"Processing chunk {chunk_index + 1}/{len(chunks)}.")
                processed_content = await async_process_text(session, chunk, custom_system_message, model_name, mode)

                if processed_content:
                    formatted_chunk = format_content_for_mode(processed_content, mode)
                    with content_lock:
                        formatted_content.extend(formatted_chunk)
                    logging.info(f"Chunk {chunk_index + 1}: Content added. Current total: {len(formatted_content)}")
                else:
                    logging.warning(f"Chunk {chunk_index + 1}: Skipped due to API error or no content.")

                progress_var.set((chunk_index + 1) / len(chunks) * 100)
                root.update_idletasks()

        with content_lock:
            logging.info(f"Total content items to write: {len(formatted_content)}")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                if mode == "Q&A":
                    json.dump(formatted_content, output_file, indent=2, ensure_ascii=False)
                else:
                    for item in formatted_content:
                        output_file.write(item["content"] + "\n\n")
            logging.info(f"Writing data to file: {output_file_path}")
            logging.info("File conversion successful.")
            messagebox.showinfo("Success", "The file has been converted successfully.")

    except Exception as e:
        logging.exception("An unexpected error occurred during the conversion process.")
        messagebox.showerror("Error", "An error occurred during the conversion process. Check logs for details.")

def start_conversion_thread(input_file_path, output_file_path, custom_system_message, progress_var, root, model_name, mode, max_chunk_size):
    stop_event.clear()
    asyncio.run(convert_to_chatml(input_file_path, output_file_path, custom_system_message, progress_var, root, model_name, mode, max_chunk_size))

def main():
    root = tk.Tk()
    root.title("Text to ChatML Converter")
    progress_var = tk.DoubleVar()
    model_var = tk.StringVar(value="gpt-3.5-turbo")
    mode_var = tk.StringVar(value="Q&A")
    max_chunk_size_var = tk.IntVar(value=500)
    api_key_var = tk.StringVar()

    def select_input_file():
        file_path = filedialog.askopenfilename()
        input_entry.delete(0, tk.END)
        input_entry.insert(0, file_path)

    def select_output_file():
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        output_entry.delete(0, tk.END)
        output_entry.insert(0, file_path)

    def start_conversion():
        input_file_path = input_entry.get()
        output_file_path = output_entry.get()
        custom_system_message = system_message_entry.get()
        selected_model = model_var.get()
        selected_mode = mode_var.get()
        max_chunk_size = max_chunk_size_var.get()
        api_key = api_key_var.get()

        if not api_key:
            messagebox.showwarning("Warning", "Please enter your OpenAI API key.")
            return

        openai.api_key = api_key

        if input_file_path and output_file_path:
            threading.Thread(target=start_conversion_thread, args=(input_file_path, output_file_path, custom_system_message, progress_var, root, selected_model, selected_mode, max_chunk_size), daemon=True).start()
        else:
            messagebox.showwarning("Warning", "Please select both input and output file paths.")

    def stop_conversion():
        stop_event.set()

    api_key_label = tk.Label(root, text="OpenAI API Key:")
    api_key_label.grid(row=0, column=0, sticky="e")
    api_key_entry = tk.Entry(root, textvariable=api_key_var, width=50)
    api_key_entry.grid(row=0, column=1)

    input_label = tk.Label(root, text="Select input text file:")
    input_label.grid(row=1, column=0, sticky="e")
    input_entry = tk.Entry(root, width=50)
    input_entry.grid(row=1, column=1)
    input_button = tk.Button(root, text="Browse...", command=select_input_file)
    input_button.grid(row=1, column=2)

    output_label = tk.Label(root, text="Select output JSON file:")
    output_label.grid(row=2, column=0, sticky="e")
    output_entry = tk.Entry(root, width=50)
    output_entry.grid(row=2, column=1)
    output_button = tk.Button(root, text="Browse...", command=select_output_file)
    output_button.grid(row=2, column=2)

    system_message_label = tk.Label(root, text="Custom System Message (Optional):")
    system_message_label.grid(row=3, column=0, sticky="e")
    system_message_entry = tk.Entry(root, width=50)
    system_message_entry.grid(row=3, column=1)
    system_message_entry.insert(0, "You are an expert assistant who provides detailed, accurate, and clear answers.")

    model_label = tk.Label(root, text="Select Model:")
    model_label.grid(row=4, column=0, sticky="e")
    model_dropdown = ttk.Combobox(root, textvariable=model_var, values=["gpt-3.5-turbo", "gpt-4"])
    model_dropdown.grid(row=4, column=1)

    mode_label = tk.Label(root, text="Select Processing Mode:")
    mode_label.grid(row=5, column=0, sticky="e")
    mode_dropdown = ttk.Combobox(root, textvariable=mode_var, values=["Q&A", "Summary", "Transcription"])
    mode_dropdown.grid(row=5, column=1)

    max_chunk_size_label = tk.Label(root, text="Max Chunk Size:")
    max_chunk_size_label.grid(row=6, column=0, sticky="e")
    max_chunk_size_entry = tk.Entry(root, textvariable=max_chunk_size_var, width=10)
    max_chunk_size_entry.grid(row=6, column=1)

    convert_button = tk.Button(root, text="Convert to ChatML", command=start_conversion)
    convert_button.grid(row=7, column=0, columnspan=3)

    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew")

    stop_button = tk.Button(root, text="STOP", command=stop_conversion)
    stop_button.grid(row=9, column=0, columnspan=3)

    root.mainloop()

if __name__ == "__main__":
    main()
