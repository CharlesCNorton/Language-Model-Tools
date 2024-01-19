Code Genius
Code Genius is a Python utility designed to utilize artificial intelligence for the creation, analysis, and enhancement of Python code. It employs the OpenAI API, with an emphasis on the advanced GPT-4 model. This tool is operated via the command line and offers a sequence of prompts to guide users in developing new Python scripts or optimizing existing ones.

Prerequisites
- Python 3.x
- OpenAI API key
- OpenAI organization name

Usage Instructions
1. Ensure the script is configured to use the GPT-4 model within the OpenAI API.
2. Set your OpenAI API key and organization name in the script's openai.api_key and openai.organization variables.
3. Execute the program: python CodeGenius.py
4. Follow the prompts to choose between creating a new script or analyzing an existing one.
5. Provide the necessary input based on your selection.
6. Code Genius will iteratively generate, analyze, and enhance the code, leveraging GPT-4's advanced capabilities.
7. After each iteration, choose to continue, save the current code, or end the process.
8. Upon completion, restart the process or exit the program.

Example Usage
Python Code Generator and Analyzer

This utility employs AI, specifically GPT-4, to generate new Python code or to analyze and improve existing code. Follow the prompts to proceed.

- Create a new program or analyze an existing one? (new/existing): new
- What code do you want me to write?: print("Hello, World!")

Processing initial request...
Initial code:
print("Hello, World!")

Iteration...
Assistant's response using GPT-4:
print("Hello, Code Genius!")

- Continue to the next step? (y/n): y

Iteration...
Assistant's response using GPT-4:
print("Hello, Code Genius!")
print("Experience the power of AI-driven coding!")

- Continue to the next step? (y/n): n
- Enter filename to save the code: hello.py

- Restart the process? (y/n): n
- Goodbye!

License
This project is licensed under the MIT license.
