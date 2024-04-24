
# QAeditor: Role-Paired Q&A Management for AI Training

## Overview

QAeditor is a powerful and intuitive tool designed to streamline the creation, management, and output of question and answer (Q&A) pairs directly into a structured JSON format. This format pairs roles with responses, making it ideal for training conversational AI models, such as chatbots, or for preprocessing in machine learning workflows. By providing an easy-to-use graphical interface, QAeditor enables experts and novices alike to contribute to the development of AI training datasets without requiring programming expertise.

## Features

- **Interactive GUI**: A user-friendly graphical interface that allows for the intuitive management of Q&A pairs.
- **Role-Paired JSON Output**: Directly outputs Q&A sets in a JSON format that associates each question and answer with a specific role, facilitating immediate use in AI training models.
- **Data Integrity Tools**: Features automatic JSON error detection and repair, ensuring that data files are always ready for use without requiring manual fixes.
- **Flexible Data Management**: Supports adding, editing, and deleting Q&A pairs to refine training datasets as needed.
- **Persistent Data Storage**: Offers options to save and load Q&A datasets from disk, allowing for long-term project management.
- **Repair Mode**: A toggleable mode that automatically attempts to correct common JSON formatting errors, enhancing data reliability.

## Installation

### Prerequisites

Before installing QAeditor, ensure you have the following:

- Python 3.6 or higher
- Tkinter library (usually included with Python)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/phanerozoic/qaeditor.git
   ```
2. **Navigate to the QAeditor directory:**
   ```bash
   cd qaeditor
   ```

3. **Run the application:**
   ```bash
   python3 qaeditor.py
   ```

## Usage

Upon launching QAeditor, you will be greeted with a simple and intuitive interface. Below are the steps to manage your Q&A data:

### Adding Q&A Pairs

1. Click on `Add Q&A` in the toolbar or right-click in the tree view area and select `Add Q&A`.
2. Enter the question and answer in the dialog boxes that appear.

### Editing and Deleting Q&A Pairs

- To **edit** a Q&A pair, select the pair in the tree view, then click `Edit Selected` or choose the same from the right-click menu.
- To **delete** a Q&A pair, select the pair and click `Delete Selected` or use the right-click menu.

### Saving and Loading Data

- Use the `File` menu to save your current dataset to a JSON file or load an existing dataset into the application.

## About the Developer

QAeditor was developed by Phanerozoic, a dedicated developer passionate about making AI and machine learning accessible to a broader audience through innovative tools and applications.
