import asyncio
import logging
import sys
import telnetlib3
from colorama import Fore, init, Style
import openai
import time
import logging

logging.getLogger('telnetlib3.stream_writer').setLevel(logging.ERROR)
openai.api_key = "ENTER_API_KEY"
openai_model = "ENTER_MODEL"
init(autoreset=True)
logging.basicConfig(level=logging.INFO)

HOST = "ENTER_HOST"
PORT = "ENTER_PORT"

SYSTEM_MESSAGE = """
You are directly interacting with a MUD (multi-user dungeon) and not a human user who responds via natural language. Here are your directives:

1. Always provide clear, actionable game commands suitable for direct input into the MUD. Avoid conversational language unless specifically requested.

2. For tasks like entering passwords or making menu selections, generate direct and applicable responses.

3. Respond with the exact number or keyword for menu choices. Ensure responses are precise.

4. Utilize the history of commands and outcomes to refine suggestions for the game environment.

5. Use the 'say' command to conduct a reflective monologue if unsure about the next step.

6. Output should be plaintext with no formatting or markup or newlines.

7. Do not engage conversationally with the MUD as if it was a user. It accepts commands and not natural language responses.

"""

context_history = []
message_buffer = []
direct_input_mode = False

async def query_gpt(prompt):
    global context_history, openai_model
    try:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + context_history + [{"role": "user", "content": prompt}]
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=openai_model,
            messages=messages
        )
        action = response.choices[0].message['content'].strip()
        context_history.append({"role": "user", "content": prompt})
        context_history.append({"role": "assistant", "content": action})
        return action
    except Exception as e:
        logging.error(f"{Fore.RED}Error querying GPT: {e}")
        return ""

async def chat_with_bot():
    print(f"\n{Fore.CYAN}Chat Mode: Type 'exit' to return to the main menu.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        action = await query_gpt(user_input)
        print(f"Bot: {action}")

async def read_server_messages(reader):
    global message_buffer
    while True:
        message = await reader.read(32000)
        if message:
            decoded_message = message.decode('utf-8').strip() if isinstance(message, bytes) else message.strip()
            logging.info(f"{Fore.GREEN}Server message: '{decoded_message}'")
            message_buffer.append((time.time(), decoded_message))

async def process_message_buffer(message_queue, buffer_delay=10):
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
    global direct_input_mode
    while True:
        if direct_input_mode:
            await asyncio.sleep(1)
            continue
        context_update = await message_queue.get()
        if not direct_input_mode:
            action = await query_gpt(context_update)
            if action:
                try:
                    writer.write(action + "\r\n")
                    await writer.drain()
                    logging.info(f"{Fore.BLUE}Command sent: {action}")
                except Exception as e:
                    logging.error(f"{Fore.RED}Error sending command: {e}")
        while not message_queue.empty():
            message_queue.get_nowait()
            message_queue.task_done()

async def listen_for_user_input(writer, message_queue):
    global message_buffer, direct_input_mode
    while True:
        user_input = await asyncio.to_thread(input, "Type your message or '!togglemode' to switch modes: ")
        if user_input == "!togglemode":
            direct_input_mode = not direct_input_mode
            mode = "Direct Input" if direct_input_mode else "Bot Driven"
            print(f"Mode switched to: {mode}")
            if not direct_input_mode:
                message_queue.put_nowait("Consolidated or latest state update")
            continue
        if direct_input_mode:
            try:
                writer.write(user_input + "\r\n")
                await writer.drain()
                logging.info(f"{Fore.BLUE}Direct command sent: {user_input}")
            except Exception as e:
                logging.error(f"{Fore.RED}Error sending direct command: {e}")
        else:
            priority_message = f"<PRIORITY DIRECTIVE> {user_input}"
            current_time = time.time()
            message_buffer.append((current_time, priority_message))
            logging.info(f"{Fore.YELLOW}Priority user message added to buffer: '{priority_message}'")

async def start_client(host, port, start_in_direct_mode=False):
    global direct_input_mode
    direct_input_mode = start_in_direct_mode
    reader, writer = await telnetlib3.open_connection(host, port, connect_minwait=1.0)
    message_queue = asyncio.Queue()
    tasks = [
        read_server_messages(reader),
        process_message_buffer(message_queue),
        send_commands(writer, message_queue),
        listen_for_user_input(writer, message_queue),
    ]
    await asyncio.gather(*tasks)

def set_system_message():
    global SYSTEM_MESSAGE
    print("Current System Message:")
    print(SYSTEM_MESSAGE)
    SYSTEM_MESSAGE = input("Enter new system message: ")
    print("System message updated.")

def main_menu():
    global HOST, PORT, openai_model
    while True:
        print(f"\n{Fore.CYAN}Main Menu")
        print(f"{Fore.CYAN}Chat and Client:")
        print(f"  {Fore.YELLOW}1. Chat with Bot")
        print(f"  {Fore.YELLOW}2. Start Client in Bot Mode")
        print(f"  {Fore.YELLOW}3. Start Client in Direct Mode")
        print(f"{Fore.CYAN}Configuration:")
        print(f"  {Fore.YELLOW}4. Change Host (Current: {Fore.GREEN}{HOST})")
        print(f"  {Fore.YELLOW}5. Change Port (Current: {Fore.GREEN}{PORT})")
        print(f"  {Fore.YELLOW}6. Set OpenAI API Key")
        print(f"  {Fore.YELLOW}7. Set OpenAI Model (Current: {Fore.GREEN}{openai_model})")
        print(f"  {Fore.YELLOW}8. Set System Message")
        print(f"{Fore.RED}9. Exit")

        choice = input(f"{Fore.CYAN}Enter your choice: ")
        if choice == '1':
            asyncio.run(chat_with_bot())
        elif choice == '2':
            asyncio.run(start_client(HOST, PORT, start_in_direct_mode=False))
        elif choice == '3':
            asyncio.run(start_client(HOST, PORT, start_in_direct_mode=True))
        elif choice == '4':
            HOST = input(f"{Fore.CYAN}Enter new host: ")
            print(f"{Fore.GREEN}Host updated to: {HOST}")
        elif choice == '5':
            PORT = input(f"{Fore.CYAN}Enter new port: ")
            print(f"{Fore.GREEN}Port updated to: {PORT}")
        elif choice == '6':
            openai.api_key = input(f"{Fore.CYAN}Enter OpenAI API Key: ")
            print(f"{Fore.GREEN}API Key updated.")
        elif choice == '7':
            openai_model = input(f"{Fore.CYAN}Enter OpenAI Model: ")
            print(f"{Fore.GREEN}Model set to {openai_model}")
        elif choice == '8':
            set_system_message()
        elif choice == '9':
            print(f"{Fore.GREEN}Exiting...")
            sys.exit()
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
