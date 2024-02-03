import asyncio
import logging
import sys
import telnetlib3
from colorama import Fore, init
import openai
import time

# Initialize OpenAI API key and Colorama for colored terminal output.
openai.api_key = "ENTER_API_KEY"  # Placeholder for OpenAI API key.
init(autoreset=True)
logging.basicConfig(level=logging.INFO)

# Configuration placeholders for connecting to the MUD server.
HOST = "SET HOST"  # Placeholder for the MUD server host.
PORT = "SET PORT"  # Placeholder for the MUD server port.

# System message to guide the behavior of the AI in generating responses.
SYSTEM_MESSAGE = """
You are directly interacting with a MUD (multi-user dungeon) and not a human user who responds via natural language. Here are your directives:

1. Always provide clear, actionable game commands suitable for direct input into the MUD. Avoid conversational language unless specifically requested.

2. For tasks like entering passwords or making menu selections, generate direct and applicable responses.

3. Respond with the exact number or keyword for menu choices. Ensure responses are precise.

5. Provide alternatives or corrective actions in case of errors or invalid command suggestions.

6. Utilize the history of commands and outcomes to refine suggestions for the game environment.

7. Use the 'say' command for internal monologue if unsure about the next step.

8. Output should be plaintext with no formatting or markup.

9. Generate and use complex passwords for any required authentication.

"""

# Global variables to hold the history of contexts and messages.
context_history = []
message_buffer = []

async def query_gpt(prompt):
    """Query GPT model for actions based on the current context."""
    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4-0125-preview",
            messages=[{"role": "system", "content": SYSTEM_MESSAGE},
                      {"role": "user", "content": prompt}]
        )
        action = response.choices[0].message['content'].strip()
        return action
    except Exception as e:
        logging.error(f"{Fore.RED}Error querying GPT: {e}")
        return ""

async def read_server_messages(reader):
    """Read messages from the MUD server and log them."""
    global message_buffer
    while True:
        message = await reader.read(32000)
        if message:
            decoded_message = message.decode('utf-8').strip() if isinstance(message, bytes) else message.strip()
            logging.info(f"{Fore.GREEN}Server message: '{decoded_message}'")
            message_buffer.append((time.time(), decoded_message))

async def process_message_buffer(message_queue, buffer_delay=10):
    """Process the buffer of messages from the server after a delay."""
    global message_buffer
    last_process_time = time.time()
    while True:
        current_time = time.time()
        time_since_last_process = current_time - last_process_time
        if time_since_last_process >= buffer_delay:
            messages_to_process = [(timestamp, msg) for (timestamp, msg) in message_buffer if current_time - timestamp >= buffer_delay]
            if messages_to_process:
                last_process_time = current_time
                context_update = " ".join([msg for _, msg in messages_to_process])
                await message_queue.put(context_update)
                message_buffer = [(timestamp, msg) for (timestamp, msg) in message_buffer if current_time - timestamp < buffer_delay]
        await asyncio.sleep(1)

async def send_commands(writer, message_queue):
    """Send commands to the MUD server based on AI's suggestions."""
    while True:
        context_update = await message_queue.get()
        action = await query_gpt(context_update)
        if action:
            try:
                writer.write(action + "\r\n")
                await writer.drain()
                logging.info(f"{Fore.BLUE}Command sent: {action}")
            except Exception as e:
                logging.error(f"{Fore.RED}Error sending command: {e}")

async def listen_for_user_input():
    """Listen for user input and add it to the message buffer with priority."""
    global message_buffer
    while True:
        user_input = await asyncio.to_thread(input, "Type your message to the bot: ")
        if user_input:
            priority_message = f"<PRIORITY DIRECTIVE> {user_input}"
            current_time = time.time()
            message_buffer.append((current_time, priority_message))
            logging.info(f"{Fore.YELLOW}Priority user message added to buffer: '{priority_message}'")

async def start_client(host, port):
    """Start the telnet client to connect to the MUD server."""
    reader, writer = await telnetlib3.open_connection(host, port, connect_minwait=1.0)
    message_queue = asyncio.Queue()
    tasks = [
        read_server_messages(reader),
        process_message_buffer(message_queue),
        send_commands(writer, message_queue),
        listen_for_user_input(),
    ]
    await asyncio.gather(*tasks)

def main_menu():
    """Display the main menu and handle user actions."""
    global HOST, PORT, SYSTEM_MESSAGE
    while True:
        print(f"\n{Fore.CYAN}Main Menu")
        print("1. Start Client")
        print("2. Change Host")
        print("3. Change Port")
        print("4. Change System Message")
        print("5. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            try:
                asyncio.run(start_client(HOST, PORT))
            except KeyboardInterrupt:
                logging.info(f"{Fore.MAGENTA}Client shutdown by user.")
            except Exception as e:
                logging.error(f"{Fore.RED}Unexpected error: {e}")
        elif choice == '2':
            HOST = input("Enter new host: ")
        elif choice == '3':
            PORT = int(input("Enter new port: "))
        elif choice == '4':
            SYSTEM_MESSAGE = input("Enter new system message:\n")
        elif choice == '5':
            print("Exiting...")
            sys.exit()
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
