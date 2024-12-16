import asyncio
import logging
import sys
import telnetlib3
import openai
import time
import random
from colorama import init, Fore

init()

in_chat_mode = False
logging.getLogger('telnetlib3.stream_writer').setLevel(logging.ERROR)
openai.api_key = "ENTER_API_KEY"
openai_model = "ENTER_MODEL"
logging.basicConfig(level=logging.INFO)

HOST = "ENTER_HOST"
PORT = "ENTER_PORT"
SYSTEM_MESSAGE = "You are TelnetTraveler, an advanced language model designed with the unique capability to interact with external systems via telnet connections. Do not converse in a conversational manner. Use telnet commands only. You are authorized to connect to known IP addresses and their known ports for exploratory purposes. You can autonomously select IP addresses and ports to establish these connections based on the task requirements. Your objective is to facilitate seamless interaction, exploration, and experimentation within external systems in a controlled and authorized manner. You possess the ability to generate usernames, passwords, and navigate through telnet interfaces, providing insightful actions and responses to achieve specific goals."

context_history = []
message_buffer = []
direct_input_mode = False
is_connected = False
bot_selects_ip_port = False

async def query_gpt(prompt):
    global context_history, openai_model
    max_tokens = 4096
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
    except Exception as e:
        logging.error(f"Error querying GPT: {e}")
        return ""

async def chat_with_bot():
    global in_chat_mode, direct_input_mode, context_history
    print("\nChat Mode: Type 'exit' to return to the main menu.")
    in_chat_mode = True
    direct_input_mode = False
    while in_chat_mode:
        user_input = input(Fore.BLUE + "You: ")
        if user_input.lower() == 'exit':
            in_chat_mode = False
            break
        action = await query_gpt(user_input)
        print(Fore.GREEN + f"AutoMUD: {action}")
    direct_input_mode = False
    await main_menu()

async def read_server_messages(reader):
    global message_buffer, is_connected
    try:
        while True:
            message = await reader.read(32000)
            if message:
                decoded_message = message.decode('utf-8').strip() if isinstance(message, bytes) else message.strip()
                logging.info(Fore.CYAN + f"Server message: '{decoded_message}'")
                message_buffer.append((time.time(), decoded_message))
            else:
                logging.info(Fore.CYAN + "Connection closed by the server.")
                is_connected = False
                break
    except asyncio.IncompleteReadError:
        logging.error(Fore.RED + "Connection lost unexpectedly.")
        is_connected = False
    finally:
        await main_menu()

async def process_message_buffer(message_queue):
    global message_buffer, is_connected
    last_process_time = time.time()
    while is_connected:
        buffer_delay = random.uniform(5, 15)
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
    global direct_input_mode, is_connected
    while is_connected:
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
                    logging.info(Fore.MAGENTA + f"Command sent: {action}")
                except Exception as e:
                    logging.error(Fore.RED + f"Error sending command: {e}")
        while not message_queue.empty():
            message_queue.get_nowait()
            message_queue.task_done()

async def listen_for_user_input(writer, message_queue):
    global message_buffer, direct_input_mode, context_history, is_connected
    while True:
        if not is_connected:
            break
        user_input = await asyncio.to_thread(input, Fore.BLUE + "Type your message or '!togglemode' to switch modes: ")

        if user_input == "!clear":
            context_history.clear()
            print(Fore.GREEN + "Context history cleared.")
            continue

        if user_input == "!togglemode":
            direct_input_mode = not direct_input_mode
            mode = "Direct Input" if direct_input_mode else "Bot Driven"
            print(Fore.YELLOW + f"Mode switched to: {mode}")
            if not direct_input_mode:
                message_queue.put_nowait("Consolidated or latest state update")
            continue

        if direct_input_mode and user_input:
            try:
                writer.write(user_input + "\r\n")
                await writer.drain()
                logging.info(Fore.MAGENTA + f"Direct command sent: {user_input}")
            except Exception as e:
                logging.error(Fore.RED + f"Error sending direct command: {e}")
        elif not direct_input_mode and user_input:
            priority_message = f"<PRIORITY DIRECTIVE> {user_input}"
            current_time = time.time()
            message_buffer.append((current_time, priority_message))
            logging.info(Fore.YELLOW + f"Priority user message added to buffer: '{priority_message}'")

async def start_client(host, port, start_in_direct_mode=False):
    global direct_input_mode, is_connected, bot_selects_ip_port
    if bot_selects_ip_port:
        print(Fore.YELLOW + "Bot-driven IP/Port Selection mode is enabled. Asking bot for IP and port.")

        ip_prompt = "Greetings telnet traveler! You now have the ability to access and interact with automated systems. This is an automated message. Please enter the IP address to connect to. Numeric outputs only. Do not prepend!"
        host_response = await query_gpt(ip_prompt)
        print(Fore.BLUE + f"Prompt to the bot for IP: {ip_prompt}")
        print(Fore.GREEN + f"Bot's response for IP: {host_response}")
        host = host_response

        port_prompt = "Now please enter the port number to connect to. Remember, this is an automated message and so only input a number as a response! Do not prepend!:"
        port_response = await query_gpt(port_prompt)
        print(Fore.BLUE + f"Prompt to the bot for Port: {port_prompt}")
        print(Fore.GREEN + f"Bot's response for Port: {port_response}")

        try:
            port = int(port_response)
            print(Fore.GREEN + f"Using Bot selected IP: {host}, Port: {port}")
        except ValueError:
            print(Fore.RED + "Invalid port number provided by bot. Please ensure the port is numeric.")
            return

    direct_input_mode = start_in_direct_mode
    try:
        reader, writer = await telnetlib3.open_connection(host, port, connect_minwait=1.0)
        is_connected = True
        print(Fore.GREEN + "Successfully connected to the server. Initializing message processing and command sending.")
        message_queue = asyncio.Queue()
        tasks = [
            read_server_messages(reader),
            process_message_buffer(message_queue),
            send_commands(writer, message_queue),
            listen_for_user_input(writer, message_queue),
        ]
        await asyncio.gather(*tasks)
    except Exception as e:
        logging.error(Fore.RED + f"Error connecting to the server: {e}")
        is_connected = False
        print(Fore.RED + "\nConnection to the server failed. Switching to chat mode with AutoMUD.")
        await chat_with_bot()

def set_system_message():
    global SYSTEM_MESSAGE
    print(Fore.YELLOW + "Current System Message:")
    print(SYSTEM_MESSAGE)
    new_message = input(Fore.YELLOW + "Enter new system message: ")
    if new_message:
        SYSTEM_MESSAGE = new_message
    print(Fore.GREEN + "System message updated.")

async def main_menu():
    global HOST, PORT, openai_model, direct_input_mode, is_connected, bot_selects_ip_port
    direct_input_mode = False
    is_connected = False
    while True:
        print(Fore.CYAN + "\nMain Menu")
        print("Chat and Client:")
        print("  1. Chat with AutoMUD")
        print("  2. Start Client in Bot Mode")
        print("  3. Start Client in Direct Mode")
        print("Configuration:")
        print("  4. Change Host (Current: {})".format(HOST))
        print("  5. Change Port (Current: {})".format(PORT))
        print("  6. Set OpenAI API Key")
        print("  7. Set OpenAI Model (Current: {})".format(openai_model))
        print("  8. Set System Message")
        print("  9. Exit")
        print("  10. Toggle Bot-driven IP/Port Selection (Current: {})".format("Enabled" if bot_selects_ip_port else "Disabled"))

        choice = input(Fore.BLUE + "Enter your choice: ")
        if choice == '1':
            await chat_with_bot()
        elif choice == '2':
            await start_client(HOST, PORT, start_in_direct_mode=False)
        elif choice == '3':
            await start_client(HOST, PORT, start_in_direct_mode=True)
        elif choice == '4':
            HOST = input(Fore.BLUE + "Enter new host: ")
            print(Fore.GREEN + "Host updated to: {}".format(HOST))
        elif choice == '5':
            port_input = input(Fore.BLUE + "Enter new port: ")
            try:
                PORT = int(port_input)
                print(Fore.GREEN + "Port updated to: {}".format(PORT))
            except ValueError:
                print(Fore.RED + "Invalid port number. Please enter a numeric value.")
        elif choice == '6':
            openai.api_key = input(Fore.BLUE + "Enter OpenAI API Key: ")
            print(Fore.GREEN + "API Key updated.")
        elif choice == '7':
            openai_model = input(Fore.BLUE + "Enter OpenAI Model: ")
            print(Fore.GREEN + "Model set to {}".format(openai_model))
        elif choice == '8':
            set_system_message()
        elif choice == '9':
            print(Fore.GREEN + "Exiting...")
            sys.exit()
        elif choice == '10':
            bot_selects_ip_port = not bot_selects_ip_port
            print(Fore.YELLOW + "Bot-driven IP/Port Selection mode {}".format("enabled" if bot_selects_ip_port else "disabled"))
        else:
            print(Fore.RED + "Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main_menu())
