import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk, Menu
import json
import re

class QAManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Q&A Editor")
        self.data = []
        self.repair_mode = tk.BooleanVar(value=False)
        self.setup_gui()
        self.setup_menu()

    def setup_gui(self):
        self.tree = ttk.Treeview(self.root, columns=('Question', 'Answer'), show='headings')
        self.tree.heading('Question', text='Question')
        self.tree.heading('Answer', text='Answer')
        self.tree.grid(row=1, column=0, columnspan=4, sticky='nsew')

        ttk.Button(self.root, text="Add Q&A", command=self.add_qa).grid(row=2, column=0)
        ttk.Button(self.root, text="Edit Selected", command=self.edit_selected).grid(row=2, column=1)
        ttk.Button(self.root, text="Delete Selected", command=self.delete_selected).grid(row=2, column=2)
        ttk.Button(self.root, text="Save to File", command=self.save_to_file).grid(row=2, column=3)

        self.status = tk.Label(self.root, text="Ready", anchor='w')
        self.status.grid(row=3, column=0, columnspan=4, sticky='ew')

    def setup_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_command(label="Save Data", command=self.save_to_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Add Q&A", command=self.add_qa)
        edit_menu.add_command(label="Edit Selected Q&A", command=self.edit_selected)
        edit_menu.add_command(label="Delete Selected Q&A", command=self.delete_selected)

        options_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=options_menu)
        options_menu.add_checkbutton(label="Enable Repair Mode", onvalue=1, offvalue=0, variable=self.repair_mode)

        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.data = json.load(file)
                self.populate_tree()
                self.status.config(text="Data loaded successfully.")
            except json.JSONDecodeError as e:
                if self.repair_mode.get():
                    self.status.config(text="Attempting to repair JSON file...")
                    self.attempt_repair(file_path)
                else:
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

    def attempt_repair(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
            repaired_content = self.repair_json(file_content)
            self.data = json.loads(repaired_content)
            self.populate_tree()
            self.status.config(text="Data loaded successfully with repairs.")
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Repair attempt failed. JSON error: {str(e)}")
            self.status.config(text="Repair attempt failed. JSON error.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during repair: {str(e)}")
            self.status.config(text="Failed to repair data. Unexpected error.")

    def repair_json(self, content):
        content = re.sub(r'(?<=[}\]])(?=[{[])', ',', content)
        if not content.endswith('}'):
            content += '}'
        if not content.startswith('['):
            content = '[' + content
        if not content.endswith(']'):
            content += ']'
        return content

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i in range(0, len(self.data), 2):
            self.tree.insert('', 'end', values=(self.data[i]['content'], self.data[i+1]['content']))

    def add_qa(self):
        question = simpledialog.askstring("Question", "Enter the question:")
        answer = simpledialog.askstring("Answer", "Enter the answer:")
        if question and answer:
            qa_pair = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
            self.data.extend(qa_pair)
            self.tree.insert('', 'end', values=(question, answer))
            self.status.config(text="Q&A added.")
        else:
            self.status.config(text="No input provided.")

    def edit_selected(self):
        selected_item = self.tree.selection()
        if selected_item:
            index = self.tree.index(selected_item) * 2
            question, answer = self.tree.item(selected_item, 'values')
            new_question = simpledialog.askstring("Edit Question", "Edit the question:", initialvalue=question)
            new_answer = simpledialog.askstring("Edit Answer", "Edit the answer:", initialvalue=answer)
            if new_question and new_answer:
                self.data[index]['content'] = new_question
                self.data[index + 1]['content'] = new_answer
                self.tree.item(selected_item, values=(new_question, new_answer))
                self.status.config(text="Q&A updated.")
        else:
            messagebox.showerror("Error", "No item selected.")

    def delete_selected(self):
        selected_item = self.tree.selection()
        if selected_item:
            index = self.tree.index(selected_item) * 2
            del self.data[index:index + 2]
            self.tree.delete(selected_item)
            self.status.config(text="Q&A deleted.")
        else:
            messagebox.showerror("Error", "No item selected.")

    def save_to_file(self):
        file_path = filedialog.asksaveasfilename(filetypes=[("JSON Files", "*.json")], defaultextension=".json")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(self.data, file, indent=2)
                self.status.config(text="Data saved to file.")
            except IOError as e:
                messagebox.showerror("Error", f"Unable to save data: {e}")
        else:
            self.status.config(text="Save cancelled.")

    def show_about(self):
        messagebox.showinfo("About", "Q&A Editor\nDesigned for easy editing and creation of Q&A pairs.")

if __name__ == "__main__":
    root = tk.Tk()
    app = QAManager(root)
    root.mainloop()
