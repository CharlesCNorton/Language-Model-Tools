from colorama import Fore, init
import anthropic
import sys

init(autoreset=True)

class ConfigurationManager:
    def __init__(self):
        self.api_key = ""
        self.model = "claude-3-opus-20240229"
        self.max_tokens = 1024

    def update_api_key(self, api_key):
        self.api_key = api_key

    def update_model(self, model):
        self.model = model

    def update_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

class ChatClient:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def enter_chat(self):
        if not self.config_manager.api_key:
            print(Fore.RED + "Error: Please enter your API key first.")
            return

        print(Fore.GREEN + f"Entering chat mode with model {self.config_manager.model}. Type 'quit' to exit.")
        try:
            anthropic_client = anthropic.Anthropic(api_key=self.config_manager.api_key)
        except Exception as e:
            print(Fore.RED + f"Failed to initialize chat client: {e}")
            return

        chat_history = []

        while True:
            try:
                user_input = input(Fore.YELLOW + "You: ").strip()
                if user_input.lower() == 'quit':
                    break
                chat_history.append({"role": "user", "content": user_input})
                response = anthropic_client.messages.create(
                    model=self.config_manager.model,
                    max_tokens=self.config_manager.max_tokens,
                    messages=chat_history
                )
                if response.content:
                    chat_history.append({"role": "assistant", "content": response.content[0].text})
                    print(Fore.GREEN + "Chatbot: " + response.content[0].text)
                else:
                    print(Fore.RED + "The chatbot did not provide a response.")
            except Exception as e:
                print(Fore.RED + f"Error during chat: {e}")

class MenuSystem:
    def __init__(self, config_manager, chat_client):
        self.config_manager = config_manager
        self.chat_client = chat_client

    def display_menu(self):
        print(Fore.CYAN + "Welcome to ClaudeChat, the Anthropic Chatbot Interface!")
        print(Fore.YELLOW + "1. Enter API Key")
        print("2. Enter chat")
        print("3. Change Model")
        print("4. Modify max tokens")
        print("5. Exit")
        choice = input(Fore.GREEN + "Enter your choice (1-5): ").strip()
        return choice

    def change_model(self):
        print("Select the model you wish to use:")
        print("1. Claude 3 Haiku")
        print("2. Claude 3 Sonnet")
        print("3. Claude 3 Opus")
        model_choice = input("Enter your choice (1-3): ").strip()

        model_map = {
            "1": "claude-3-haiku-20240229",
            "2": "claude-3-sonnet-20240229",
            "3": "claude-3-opus-20240229"
        }

        if model_choice in model_map:
            self.config_manager.update_model(model_map[model_choice])
            print(Fore.GREEN + f"Model changed to {self.config_manager.model}.")
        else:
            print(Fore.RED + "Invalid choice. Model not changed.")

    def run(self):
        try:
            while True:
                choice = self.display_menu()
                if choice == "1":
                    new_api_key = input(Fore.MAGENTA + "Enter your Anthropic API Key: ").strip()
                    self.config_manager.update_api_key(new_api_key)
                    if new_api_key:
                        print(Fore.GREEN + "API Key set successfully!")
                    else:
                        print(Fore.RED + "No API Key was entered. Please try again.")
                elif choice == "2":
                    self.chat_client.enter_chat()
                elif choice == "3":
                    self.change_model()
                elif choice == "4":
                    try:
                        new_max_tokens = int(input(Fore.CYAN + "Enter new max tokens: ").strip())
                        if new_max_tokens > 0:
                            self.config_manager.update_max_tokens(new_max_tokens)
                        else:
                            print(Fore.RED + "Max tokens must be a positive integer.")
                    except ValueError:
                        print(Fore.RED + "Invalid input. Max tokens must be a positive integer.")
                elif choice == "5":
                    print(Fore.CYAN + "Exiting ClaudeChat. Goodbye!")
                    break
                else:
                    print(Fore.RED + "Invalid choice. Please enter a number between 1 and 5.")
        except KeyboardInterrupt:
            print(Fore.CYAN + "\nOperation cancelled by user. Exiting ClaudeChat.")
        except Exception as e:
            print(Fore.RED + f"An unexpected error occurred: {e}")
            sys.exit(1)

if __name__ == "__main__":
    config_manager = ConfigurationManager()
    chat_client = ChatClient(config_manager)
    menu_system = MenuSystem(config_manager, chat_client)
    menu_system.run()
