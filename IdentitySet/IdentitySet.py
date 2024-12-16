import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk, Menu
import json
import re

class QAManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Q&A Editor")
        self.data = {}
        self.repair_mode = tk.BooleanVar(value=False)
        self.categories = ["Identity Questions", "Capability Queries", "Preference Queries", "Knowledge Questions", "Contextual Interactions",
                           "Emotional Responses", "Historical Information", "Future and Hypotheticals", "Technical Support", "Feedback and Learning"]
        self.length_categories = ['Short (1-3 words)', 'Medium (4-10 words)', 'Long (11-20 words)', 'Extended (21+ words)']
        self.setup_gui()
        self.setup_menu()
        self.initialize_data_structure()

    def initialize_data_structure(self):
        for category in self.categories:
            self.data[category] = {length: [] for length in self.length_categories}

    def setup_gui(self):
        self.tree = ttk.Treeview(self.root, columns=('Category', 'Length', 'Question', 'Answer'), show='headings')
        self.tree.heading('Category', text='Category')
        self.tree.heading('Length', text='Length')
        self.tree.heading('Question', text='Question')
        self.tree.heading('Answer', text='Answer')
        self.tree.grid(row=1, column=0, columnspan=4, sticky='nsew')

        # Dropdowns for category and length selection
        self.category_var = tk.StringVar(self.root)
        self.length_var = tk.StringVar(self.root)
        self.category_dropdown = ttk.Combobox(self.root, textvariable=self.category_var, values=self.categories, state="readonly")
        self.length_dropdown = ttk.Combobox(self.root, textvariable=self.length_var, values=self.length_categories, state="readonly")
        self.category_dropdown.grid(row=2, column=0, sticky='ew')
        self.length_dropdown.grid(row=2, column=1, sticky='ew')
        self.category_var.set(self.categories[0])  # Set default value
        self.length_var.set(self.length_categories[0])  # Set default value

        ttk.Button(self.root, text="Add Q&A", command=self.add_qa).grid(row=2, column=2)
        ttk.Button(self.root, text="Edit Selected", command=self.edit_selected).grid(row=2, column=3)
        ttk.Button(self.root, text="Delete Selected", command=self.delete_selected).grid(row=2, column=4)
        ttk.Button(self.root, text="Save to File", command=self.save_to_file).grid(row=3, column=4)

        self.status = tk.Label(self.root, text="Ready", anchor='w')
        self.status.grid(row=4, column=0, columnspan=4, sticky='ew')

    def add_qa(self):
        category = self.category_var.get()
        length = self.length_var.get()
        question = simpledialog.askstring("Question", "Enter the question:")
        answer = simpledialog.askstring("Answer", "Enter the answer:")
        if category and length and question and answer:
            word_count = len(question.split()) + len(answer.split())
            max_count = 4 if 'Short' in length or 'Medium' in length else 2
            if ((length.startswith('Short') and word_count <= 3) or
                (length.startswith('Medium') and 4 <= word_count <= 10) or
                (length.startswith('Long') and 11 <= word_count <= 20) or
                (length.startswith('Extended') and word_count >= 21)) and len(self.data[category][length]) < max_count * 2:
                self.data[category][length].append({'role': 'user', 'content': question})
                self.data[category][length].append({'role': 'assistant', 'content': answer})
                self.tree.insert('', 'end', values=(category, length, question, answer))
                self.status.config(text="Q&A added.")
            else:
                if len(self.data[category][length]) >= max_count * 2:
                    messagebox.showinfo("Limit Reached", "The maximum number of Q&A pairs for this category and length has been reached.")
                else:
                    messagebox.showinfo("Invalid Length", "The word count for the selected length category does not match.")
        else:
            self.status.config(text="No input provided.")

    def edit_selected(self):
        selected_item = self.tree.selection()
        if selected_item:
            values = self.tree.item(selected_item, 'values')
            category = values[0]
            length = values[1]
            new_question = simpledialog.askstring("Edit Question", "Edit the question:", initialvalue=values[2])
            new_answer = simpledialog.askstring("Edit Answer", "Edit the answer:", initialvalue=values[3])
            if new_question and new_answer:
                # Update the data structure
                qa_list = self.data[category][length]
                for index, qa in enumerate(qa_list):
                    if qa['content'] == values[2] and qa['role'] == 'user':
                        qa_list[index]['content'] = new_question
                    elif qa['content'] == values[3] and qa['role'] == 'assistant':
                        qa_list[index]['content'] = new_answer
                self.tree.item(selected_item, values=(category, length, new_question, new_answer))
                self.status.config(text="Q&A updated.")
        else:
            messagebox.showerror("Error", "No item selected.")

    def delete_selected(self):
        selected_item = self.tree.selection()
        if selected_item:
            values = self.tree.item(selected_item, 'values')
            category = values[0]
            length = values[1]
            # Remove the selected Q&A pair
            qa_list = self.data[category][length]
            new_list = [qa for qa in qa_list if qa['content'] != values[2] and qa['content'] != values[3]]
            self.data[category][length] = new_list
            self.tree.delete(selected_item)
            self.status.config(text="Q&A deleted.")
        else:
            messagebox.showerror("Error", "No item selected.")

    def save_to_file(self):
        file_path = filedialog.asksaveasfilename(filetypes=[("JSON Files", "*.json")], defaultextension=".json")
        if file_path:
            # Flatten the data into a list of alternating user/assistant pairs
            flat_list = []
            for category in self.data:
                for length_category in self.data[category]:
                    flat_list.extend(self.data[category][length_category])  # Collect all Q&A pairs

            # Save the flattened list to a JSON file
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(flat_list, file, indent=2)
                self.status.config(text="Data saved to file.")
            except IOError as e:
                messagebox.showerror("Error", f"Unable to save data: {e}")
        else:
            self.status.config(text="Save cancelled.")

    def setup_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_command(label="Save Data", command=self.save_to_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.data = json.load(file)
                    self.status.config(text="Data loaded successfully.")
            except json.JSONDecodeError as e:
                messagebox.showerror("Error", f"Failed to parse JSON file. Error message: {str(e)}")
                self.status.config(text="Failed to load data. JSON format error.")
            except FileNotFoundError:
                messagebox.showerror("Error", "File not found.")
                self.status.config(text="Failed to load data. File not found.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                self.status.config(text="Failed to load data. Unexpected error.")
        else:
            self.status.config(text="Load cancelled.")

if __name__ == "__main__":
    root = tk.Tk()
    app = QAManager(root)
    root.mainloop()
