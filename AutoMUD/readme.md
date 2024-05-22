
# AutoMUD README

## Description

AutoMUD is an AI agent that connects GPT-4 to Telnet-based environments, enabling seamless integration of AI-driven and manual interaction across various online platforms. From Multi-User Dungeons (MUDs) to online chess games and other Telnet-related applications, AutoMUD exemplifies how AI can autonomously engage with and play online games. It offers two modes: Direct Input Mode, where the player inputs commands directly, and Bot Mode, where GPT-4 autonomously controls the gameplay but can also take advice or requests on behavior.

## Features

- **Direct Input Mode**: Allows for direct player control with the option for GPT-4 to observe and learn.
- **Bot Mode**: GPT-4 autonomously engages in interaction and gameplay, with the flexibility to take advice or behavior requests from the player (e.g., taunting during a chess game).
- **Flexible Interaction**: Switch between manual and autonomous modes to accommodate different play styles and scenarios.
- **Real-Time Adaptation**: GPT-4 dynamically responds to the environment, whether in direct control or learning from player inputs.
- **Versatile Connectivity**: Connects to various Telnet-based platforms, including MUDs, online chess games, and more.

## Requirements

- Python 3.8 or higher
- Required libraries: asyncio, telnetlib3, colorama, openai
- An active OpenAI API key

## Setup

1. Clone the repository and install dependencies.
   ```sh
   git clone <repository_url>
   cd <repository_directory>
   pip install -r requirements.txt
   ```
2. Configure API key, HOST, and PORT settings in `autoMUD.py`.

## Usage

1. Start AutoMUD.
2. Choose your mode:
   - **Direct Input Mode**: Interact directly with the Telnet-based application.
   - **Bot Mode**: Allow GPT-4 to control the interaction, with the option to provide advice or requests.
3. Interact with your chosen Telnet-based application through GPT-4's advanced capabilities or direct input.

## Configuration

- Easily configurable settings for HOST, PORT, and API key, allowing for personalized experiences across different Telnet-based platforms.

## Contributing

- Contributions are welcome. Fork the project, commit your changes, and submit a pull request for review.

## License

- AutoMUD is available under the MIT License. See the LICENSE file for more details.

## Support

- For support, please open an issue on the GitHub project page.

## Acknowledgments

- Special thanks to OpenAI for the GPT-4 technology that powers AutoMUD.

## Philosophical Implications

The advent of AutoMUD represents a significant milestone in AI interaction with virtual environments, particularly as token costs decrease, context lengths increase, and capabilities multiply. As token costs decrease, the economic barriers to prolonged and complex AI-driven gameplay are reduced, making it feasible for AI to engage in more extensive and nuanced interactions. Increasing context lengths allow the AI to retain more information about its interactions, leading to a deeper and more coherent understanding of the game environment. This enhanced memory capability means that AI can develop long-term strategies and remember past interactions, making it a more formidable and human-like participant in online games.

Furthermore, as the capabilities of AI continue to grow, the potential for sophisticated and adaptable gameplay increases. The AI can handle a broader range of scenarios, adapt to various gameplay styles, and provide a richer interactive experience. This progress not only transforms the landscape of online gaming but also paves the way for more complex and meaningful AI-human collaborations in virtual environments.

By leveraging these advancements, AutoMUD showcases the potential of AI to autonomously interact with and master various online platforms, offering a glimpse into a future where AI agents can serve as both partners and challengers in virtual worlds.
