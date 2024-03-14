from colorama import Fore, init
import anthropic
import sys

init(autoreset=True)

API_KEY = ""
MODEL = "claude-3-opus-20240229"  # Default model
MAX_TOKENS = 1024


def display_menu():
    print(Fore.CYAN + "Welcome to ClaudeChat, the Anthropic Chatbot Interface!")
    print(Fore.YELLOW + "1. Enter API Key")
    print("2. Enter chat")
    print("3. Change Model")
    print("4. Modify max tokens")
    print("5. Exit")
    choice = input(Fore.GREEN + "Enter your choice (1-5): ")
    return choice


def enter_api_key():
    global API_KEY
    API_KEY = input(Fore.MAGENTA + "Enter your Anthropic API Key: ")
    if API_KEY:
        print(Fore.GREEN + "API Key set successfully!")
    else:
        print(Fore.RED + "No API Key was entered. Please try again.")


def enter_chat():
    global API_KEY, MODEL, MAX_TOKENS
    if not API_KEY:
        print(Fore.RED + "Error: Please enter your API key first.")
        return

    print(Fore.GREEN + "Entering chat mode with model " + MODEL + ". Type 'quit' to exit.")
    try:
        anthropic_client = anthropic.Anthropic(api_key=API_KEY)
    except Exception as e:
        print(Fore.RED + f"Failed to initialize chat client: {e}")
        return

    chat_history = []

    while True:
        user_input = input(Fore.YELLOW + "You: ")
        if user_input.lower() == 'quit':
            break

        chat_history.append({"role": "user", "content": user_input})

        try:
            response = anthropic_client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=chat_history
            )
            if response.content:
                chat_history.append({"role": "assistant", "content": response.content[0].text})
                print(Fore.GREEN + "Chatbot: " + response.content[0].text)
            else:
                print(Fore.RED + "The chatbot did not provide a response.")
        except Exception as e:
            print(Fore.RED + "Failed to send message or process response: " + str(e))


def change_model():
    global MODEL
    print("Select the model you wish to use:")
    print("1. Claude 3 Haiku")
    print("2. Claude 3 Sonnet")
    print("3. Claude 3 Opus")
    model_choice = input("Enter your choice (1-3): ")

    if model_choice == "1":
        MODEL = "claude-3-haiku-20240229"
    elif model_choice == "2":
        MODEL = "claude-3-sonnet-20240229"
    elif model_choice == "3":
        MODEL = "claude-3-opus-20240229"
    else:
        print(Fore.RED + "Invalid choice. Model not changed.")
        return

    print(Fore.GREEN + f"Model changed to {MODEL}.")


def modify_max_tokens():
    global MAX_TOKENS
    try:
        MAX_TOKENS = int(input(Fore.CYAN + "Enter new max tokens (integer, current: " + str(MAX_TOKENS) + "): "))
        print(Fore.GREEN + "Max tokens updated successfully.")
    except ValueError:
        print(Fore.RED + "Invalid input for max tokens. Please enter an integer.")


def main():
    while True:
        user_choice = display_menu()
        if user_choice == "1":
            enter_api_key()
        elif user_choice == "2":
            enter_chat()
        elif user_choice == "3":
            change_model()
        elif user_choice == "4":
            modify_max_tokens()
        elif user_choice == "5":
            print(Fore.CYAN + "Exiting ClaudeChat. Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.CYAN + "\nOperation cancelled by user. Exiting ClaudeChat.")
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred: {e}")
        sys.exit(1)
