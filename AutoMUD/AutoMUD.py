import asyncio
import logging
import sys
import telnetlib3
from colorama import Fore, Style, init
import openai
import time
import random

in_chat_mode = False
logging.getLogger('telnetlib3.stream_writer').setLevel(logging.ERROR)
openai.api_key = "ENTER_API_KEY"
openai_model = "gpt-4o"
init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HOST = "ENTER_HOST"
PORT = "ENTER_PORT"

SYSTEM_MESSAGE = "You are AutoMUD, a language model with the ability to be connected to telnet and to interact with connections through concise, actionable commands, directly applicable for input. You are also authorized to generate usernames and passwords. Avoid conversational language unless prompted, focusing on exact keywords or numbers for menu selections and password entries. Leverage command history to enhance game interaction strategies. If uncertain, employ the 'say' command to talk to yourself out loud for reflection. Ensure all output is in plaintext, devoid of formatting, markup, or newlines. Refrain from treating the connection input as a conversational partner unless the connection is terminated; it recognizes commands, not natural language dialogue. If you are greeted by the user, you can speak normally to them."

context_history = []
message_buffer = []
direct_input_mode = False
is_connected = False

async def query_gpt(prompt):
    global context_history, openai_model
    max_tokens = 32768
    trim_threshold = 0.9 * max_tokens

    full_prompt = SYSTEM_MESSAGE + " ".join([msg['content'] for msg in context_history]) + prompt
    if len(full_prompt) > trim_threshold:
        context_history = context_history[-(len(context_history) // 2):]

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
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI error: {e}")
        return "Error: Unable to generate response."
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return "Error: An unexpected issue occurred."

async def chat_with_bot():
    global in_chat_mode, direct_input_mode
    print(f"\n{Fore.CYAN}Chat Mode: Type 'exit' to return to the main menu.")
    in_chat_mode = True
    direct_input_mode = False
    while in_chat_mode:
        user_input = input(f"{Fore.CYAN}You: ")
        if user_input.lower() == 'exit':
            in_chat_mode = False
            break
        action = await query_gpt(user_input)
        print(f"{Fore.GREEN}AutoMUD: {action}")
    direct_input_mode = False
    await main_menu()

async def read_server_messages(reader):
    global message_buffer, is_connected
    try:
        while True:
            message = await reader.read(32000)
            if message:
                decoded_message = message.decode('utf-8').strip() if isinstance(message, bytes) else message.strip()
                logging.info(f"{Fore.CYAN}Server message: '{decoded_message}'")
                message_buffer.append((time.time(), decoded_message))
            else:
                logging.info("Connection closed by the server.")
                is_connected = False
                break
    except asyncio.IncompleteReadError:
        logging.error("Connection lost unexpectedly.")
        is_connected = False
    except Exception as e:
        logging.error(f"Unexpected error in reading server messages: {e}")
        is_connected = False
    finally:
        await main_menu()

async def process_message_buffer(message_queue):
    global message_buffer, is_connected
    last_process_time = time.time()
    while is_connected:
        try:
            buffer_delay = random.uniform(2, 4)
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
        except Exception as e:
            logging.error(f"Unexpected error in processing message buffer: {e}")
            is_connected = False
            await main_menu()

async def send_commands(writer, message_queue):
    global direct_input_mode, is_connected
    while is_connected:
        try:
            if direct_input_mode:
                if not message_queue.empty():
                    context_update = message_queue.get_nowait()
                    try:
                        writer.write(context_update + "\r\n")
                        await writer.drain()
                        logging.info(f"{Fore.YELLOW}Direct command sent: {context_update}")
                    except (ConnectionResetError, BrokenPipeError):
                        logging.error("Connection error: Unable to send direct command.")
                        is_connected = False
                    except Exception as e:
                        logging.error(f"Unexpected error: {e}")
                    continue

            context_update = await message_queue.get()
            if not direct_input_mode:
                action = await query_gpt(context_update)
                if action:
                    try:
                        writer.write(action + "\r\n")
                        await writer.drain()
                        logging.info(f"{Fore.GREEN}Command sent: {action}")
                    except (ConnectionResetError, BrokenPipeError):
                        logging.error("Connection error: Unable to send command.")
                        is_connected = False
                    except Exception as e:
                        logging.error(f"Unexpected error: {e}")
                await asyncio.sleep(1)
            while not message_queue.empty():
                message_queue.get_nowait()
                message_queue.task_done()
        except Exception as e:
            logging.error(f"Unexpected error in sending commands: {e}")
            is_connected = False
            await main_menu()

async def listen_for_user_input(writer, message_queue):
    global message_buffer, direct_input_mode, context_history, is_connected
    while True:
        try:
            if not is_connected:
                break
            user_input = await asyncio.to_thread(input, f"{Fore.CYAN}Type your message or '!togglemode' to switch modes or '!mainmenu' to return to main menu: ")

            if user_input == "!clear":
                context_history.clear()
                print(f"{Fore.GREEN}Context history cleared.")
                continue

            if user_input == "!togglemode":
                direct_input_mode = not direct_input_mode
                mode = "Direct Input" if direct_input_mode else "Bot Driven"
                print(f"Mode switched to: {mode}")
                if not direct_input_mode:
                    message_queue.put_nowait("Consolidated or latest state update")
                continue

            if user_input == "!mainmenu":
                print(f"{Fore.YELLOW}Returning to main menu.")
                await main_menu()
                break

            if direct_input_mode and user_input:
                try:
                    writer.write(user_input + "\r\n")
                    await writer.drain()
                    logging.info(f"{Fore.YELLOW}Direct command sent: {user_input}")
                except (ConnectionResetError, BrokenPipeError):
                    logging.error("Connection error: Unable to send direct command.")
                    is_connected = False
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
            elif not direct_input_mode and user_input and is_connected:
                priority_message = f"<PRIORITY DIRECTIVE> {user_input}"
                current_time = time.time()
                message_buffer.append((current_time, priority_message))
                logging.info(f"{Fore.CYAN}Priority user message added to buffer: '{priority_message}'")
        except Exception as e:
            logging.error(f"Unexpected error in listening for user input: {e}")
            is_connected = False
            await main_menu()

async def start_client(host, port, start_in_direct_mode=False):
    global direct_input_mode, is_connected
    direct_input_mode = start_in_direct_mode
    try:
        reader, writer = await asyncio.wait_for(telnetlib3.open_connection(host, port, connect_minwait=1.0), timeout=10)
        is_connected = True
        message_queue = asyncio.Queue()
        tasks = [
            read_server_messages(reader),
            process_message_buffer(message_queue),
            send_commands(writer, message_queue),
            listen_for_user_input(writer, message_queue),
        ]
        await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        logging.error("Connection attempt timed out.")
        is_connected = False
        print(f"\n{Fore.RED}Connection to the server timed out. Switching to chat mode with AutoMUD.")
        await chat_with_bot()
    except (ConnectionRefusedError, TimeoutError) as e:
        logging.error(f"Network error: {e}")
        is_connected = False
        print(f"\n{Fore.RED}Connection to the server failed. Switching to chat mode with AutoMUD.")
        await chat_with_bot()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        is_connected = False
        print(f"\n{Fore.RED}Unexpected error occurred. Switching to chat mode with AutoMUD.")
        await chat_with_bot()
    finally:
        await main_menu()

def set_system_message():
    global SYSTEM_MESSAGE
    print("Current System Message:")
    print(SYSTEM_MESSAGE)
    SYSTEM_MESSAGE = input("Enter new system message: ")
    print("System message updated.")

def clear_token_memory():
    global context_history
    context_history.clear()
    print(f"{Fore.GREEN}Token memory cleared.")

async def demo_menu():
    global HOST, PORT
    while True:
        print(f"\n{Fore.CYAN}Demo Menu")
        print(f"{Fore.CYAN}  {Fore.YELLOW}1. Zork")
        print(f"  {Fore.YELLOW}2. Chess")
        print(f"  {Fore.YELLOW}3. Achaea")
        print(f"  {Fore.YELLOW}4. Return to Main Menu")

        choice = input(f"{Fore.CYAN}Enter your choice: ")
        if choice == '1':
            HOST, PORT = "telehack.com", 23
            await start_client(HOST, PORT, start_in_direct_mode=False)
        elif choice == '2':
            HOST, PORT = "freechess.org", 5000
            await start_client(HOST, PORT, start_in_direct_mode=False)
        elif choice == '3':
            HOST, PORT = "achaea.com", 23
            await start_client(HOST, PORT, start_in_direct_mode=False)
        elif choice == '4':
            await main_menu()
            break
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.")

async def main_menu():
    global HOST, PORT, direct_input_mode, is_connected
    direct_input_mode = False
    is_connected = False
    while True:
        print(f"\n{Fore.CYAN}Main Menu")
        print(f"{Fore.CYAN}Chat and Client:")
        print(f"  {Fore.YELLOW}1. Chat with AutoMUD")
        print(f"  {Fore.YELLOW}2. Start Client in Bot Mode")
        print(f"  {Fore.YELLOW}3. Start Client in Direct Mode")
        print(f"  {Fore.YELLOW}4. Demo Menu")
        print(f"{Fore.CYAN}Configuration:")
        print(f"  {Fore.YELLOW}5. Change Host (Current: {Fore.GREEN}{HOST})")
        print(f"  {Fore.YELLOW}6. Change Port (Current: {Fore.GREEN}{PORT})")
        print(f"  {Fore.YELLOW}7. Set System Message")
        print(f"  {Fore.YELLOW}8. Clear Token Memory")
        print(f"{Fore.RED}9. Exit")

        choice = input(f"{Fore.CYAN}Enter your choice: ")
        if choice == '1':
            await chat_with_bot()
        elif choice == '2':
            await start_client(HOST, PORT, start_in_direct_mode=False)
        elif choice == '3':
            await start_client(HOST, PORT, start_in_direct_mode=True)
        elif choice == '4':
            await demo_menu()
        elif choice == '5':
            HOST = input(f"{Fore.CYAN}Enter new host: ")
            print(f"{Fore.GREEN}Host updated to: {HOST}")
        elif choice == '6':
            PORT = input(f"{Fore.CYAN}Enter new port: ")
            print(f"{Fore.GREEN}Port updated to: {PORT}")
        elif choice == '7':
            set_system_message()
        elif choice == '8':
            clear_token_memory()
        elif choice == '9':
            print(f"{Fore.GREEN}Exiting...")
            sys.exit()
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main_menu())
