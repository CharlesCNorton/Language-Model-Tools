import asyncio
import tiktoken
from colorama import init, Fore

# Initialize Colorama to auto-reset colors after each print statement.
init(autoreset=True)

async def context_management_simulation():
    # Greet users and explain the purpose of the simulation.
    print(Fore.CYAN + "Welcome to the Context Management and Token Counting Simulation!")
    print(Fore.CYAN + "This simulation showcases dynamic context management using TikToken with OpenAI models.\n")

    # Prompt user to start the simulation or exit.
    action = input(Fore.LIGHTBLUE_EX + "Press Enter to start or type 'exit' to quit: ").lower()
    if action == 'exit':
        # User chose to exit the simulation.
        print(Fore.CYAN + "Exiting simulation.")
        return

    # Define a list of sentences to simulate user inputs.
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A wizard's job is to vex chumps quickly in fog.",
        "Pack my box with five dozen liquor jugs.",
        "How quickly daft jumping zebras vex.",
        "Jinxed wizards pluck ivy from the big quilt."
    ]

    # Initialize variables for context history and token management.
    context_history = []
    max_tokens = 4096  # Maximum tokens to manage context within.
    trim_threshold = 0.9 * max_tokens  # Threshold to trigger context trimming.
    encoding = tiktoken.encoding_for_model("gpt-4")  # Get encoding for specified model.
    trim_events = 0  # Counter for context trim events.
    initial_tokens = 0  # Store initial token count for statistics.

    # Main loop to simulate context management and token counting.
    for i in range(5000):  # Loop through a large number of iterations to simulate extensive use.
        # Append each sentence to the context history, simulating user input.
        context_history.append({"role": "user", "content": sentences[i % len(sentences)]})

        if i == 0:
            # Calculate initial tokens in context for statistics.
            initial_tokens = sum(len(encoding.encode(message["content"])) for message in context_history)

        if i % 100 == 0:
            # Periodically report the current state of the context and token count.
            current_tokens = sum(len(encoding.encode(message["content"])) for message in context_history)
            color = Fore.LIGHTBLUE_EX if i % 200 == 0 else Fore.CYAN
            print(color + f"Iteration {i}: Context size {len(context_history)}, Total tokens: {current_tokens}")

        if sum(len(encoding.encode(message["content"])) for message in context_history) > trim_threshold:
            # Trim context when the total token count exceeds the threshold.
            context_history = context_history[len(context_history) // 2:]
            print(Fore.LIGHTYELLOW_EX + f"Context trimmed at iteration {i}. New size: {len(context_history)}")
            trim_events += 1  # Increment trim event counter.

    # Calculate final token count for statistics.
    final_tokens = sum(len(encoding.encode(message["content"])) for message in context_history)
    # Display final statistics and conclude the simulation with a unique color.
    print(Fore.LIGHTWHITE_EX + f"\nFinal context size: {len(context_history)}, Total tokens: {final_tokens}")
    print(Fore.LIGHTWHITE_EX + f"Simulation complete. Initial tokens: {initial_tokens}, Final tokens: {final_tokens}, Total iterations: {i+1}, Context trim events: {trim_events}")
    print(Fore.LIGHTWHITE_EX + "This showcases the effectiveness of dynamic context management and token counting with TikToken.")

# Ensure the script runs in an asyncio event loop if executed as the main program.
if __name__ == "__main__":
    asyncio.run(context_management_simulation())
