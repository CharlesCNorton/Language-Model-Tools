GGUF Converter
==============

GGUF Converter is a Python tool for converting HuggingFace models to the single-file GGUF format, optimized for large language models with llama.cpp.

Features
--------
- Convert HuggingFace models to GGUF.
- Quantization options: 8-bit, 16-bit, 32-bit.
- GUI for easy interaction.
- Robust error handling and input validation.

Installation
------------
1. Ensure Python is installed.
2. Clone/download this repository.
3. Install dependencies: `pip install -r requirements.txt` (if needed).

Usage
-----
1. Run `python gguf_converter.py`.
2. Set the llama.cpp directory and choose the HuggingFace model.
3. Select quantization level and output location.
4. The model is converted and saved in GGUF format.

Quantization Options
--------------------
- 8-bit (q8_0): Faster, less accurate.
- 16-bit (f16): Balanced performance and accuracy.
- 32-bit (f32): Original quality, higher accuracy.

System Requirements
-------------------
- Python 3.x
- GUI access (for Tkinter dialogs).
- Adequate storage for models.

Support
-------
For issues or questions, open an issue or pull request in the repository.

License
-------
Distributed under MIT License. See `LICENSE` file.
