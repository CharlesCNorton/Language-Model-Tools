import tkinter as tk
from tkinter import filedialog
import os
import requests
import random
import json
import openai
import anthropic
from colorama import init, Fore, Style
from google.api_core import exceptions
import google.generativeai as genai

init()

API_KEYS = {
    "gemini": "",
    "claude": "",
    "gpt4": ""
}

def update_api_key(api_name, api_key):
    API_KEYS[api_name] = api_key
    print(Fore.GREEN + f"{api_name.capitalize()} API key updated successfully." + Style.RESET_ALL)

def set_google_credentials_via_gui():
    root = tk.Tk()
    root.withdraw()
    credentials_path = filedialog.askopenfilename(title="Select Google JSON Credentials")
    if credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        print(Fore.GREEN + "Google Application Credentials updated successfully." + Style.RESET_ALL)

def call_gemini_api(prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except exceptions.GoogleAPIError as e:
        print(Fore.RED + f"Error calling Gemini API: {e}" + Style.RESET_ALL)
        return None
    except Exception as e:
        print(Fore.RED + f"Unexpected error calling Gemini API: {e}" + Style.RESET_ALL)
        return None

def call_claude_api(prompt):
    try:
        anthropic_client = anthropic.Anthropic(api_key=API_KEYS["claude"])
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except anthropic.AnthropicError as ae:
        print(Fore.RED + f"Error calling Claude API: {ae}" + Style.RESET_ALL)
        return None
    except Exception as e:
        print(Fore.RED + f"Unexpected error calling Claude API: {e}" + Style.RESET_ALL)
        return None

def call_gpt4_api(prompt):
    try:
        openai.api_key = API_KEYS["gpt4"]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        print(Fore.RED + f"Error calling GPT-4 API: {e}" + Style.RESET_ALL)
        return None

def judge_responses(responses):
    try:
        shuffled_responses = random.sample(responses, len(responses))
        judgment_prompt = "Please rank the following responses from best to worst, considering factors like relevance, coherence, and helpfulness. Provide only the rank order (e.g., '2, 1, 3'), without explaining your reasoning:\n\n"
        for i, response in enumerate(shuffled_responses, start=1):
            judgment_prompt += f"Response {i}: {response}\n\n"
        rank_order = call_claude_api(judgment_prompt)
        if rank_order is None:
            return None
        rank_order = [int(x.strip()) for x in rank_order.split(",")]
        best_response_index = shuffled_responses.index(responses[rank_order[0] - 1])
        return best_response_index
    except (ValueError, IndexError) as e:
        print(Fore.RED + f"Error judging responses: {e}" + Style.RESET_ALL)
        return None

def ask_triumvirate(question):
    responses = [
        call_gemini_api(question),
        call_claude_api(question),
        call_gpt4_api(question)
    ]
    responses = [response for response in responses if response]
    if not responses:
        return "No responses available."
    best_response_index = judge_responses(responses)
    if best_response_index is None:
        return "Error judging responses."
    return best_response_index, responses[best_response_index]

def main_menu():
    while True:
        print(Fore.CYAN + "\nTriumvirate AI System" + Style.RESET_ALL)
        print("1. Ask a question")
        print("2. Update Gemini API key" + (Fore.GREEN + " [ACTIVE]" + Style.RESET_ALL if API_KEYS["gemini"] else ""))
        print("3. Update Claude API key" + (Fore.GREEN+ " [ACTIVE]" + Style.RESET_ALL if API_KEYS["claude"] else ""))
        print("4. Update GPT-4 API key" + (Fore.GREEN + " [ACTIVE]" + Style.RESET_ALL if API_KEYS["gpt4"] else ""))
        print("5. Update Google Application Credentials (GUI)")
        print("6. Exit")
        try:
            choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)
            if choice == "1":
                question = input(Fore.YELLOW + "What is your question? " + Style.RESET_ALL)
                print(Fore.MAGENTA + "Thinking...\n" + Style.RESET_ALL)
                result = ask_triumvirate(question)
                if isinstance(result, tuple):
                    best_response_index, answer = result
                    print(Fore.GREEN + f"Best response (from AI #{best_response_index + 1}): {answer}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + result + Style.RESET_ALL)
            elif choice == "2":
                api_key = input(Fore.YELLOW + "Enter the Gemini API key: " + Style.RESET_ALL)
                update_api_key("gemini", api_key)
            elif choice == "3":
                api_key = input(Fore.YELLOW + "Enter the Claude API key: " + Style.RESET_ALL)
                update_api_key("claude", api_key)
            elif choice == "4":
                api_key = input(Fore.YELLOW + "Enter the GPT-4 API key: " + Style.RESET_ALL)
                update_api_key("gpt4", api_key)
            elif choice == "5":
                set_google_credentials_via_gui()
            elif choice == "6":
                print(Fore.RED + "Exiting..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Invalid choice. Please enter a number between 1 and 6." + Style.RESET_ALL)
        except KeyboardInterrupt:
            print(Fore.RED + "\nKeyboard interrupt detected. Exiting..." + Style.RESET_ALL)
            break

if __name__ == "__main__":
    main_menu()
