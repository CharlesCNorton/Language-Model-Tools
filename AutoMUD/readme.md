AutoMUD README
==============

Description
-----------
AutoMUD connects GPT-4 to Multi-User Dungeons (MUDs) through Telnet, enabling a seamless integration of AI-driven and manual gameplay. It offers two modes: Direct Input Mode, where the player inputs commands directly to the MUD, and Bot Mode, where GPT-4 autonomously controls the gameplay from the start or can be toggled to take over at any point, without the need for prior context learning.

Features
--------
- **Direct Input Mode**: Allows for direct player control over the MUD with the option for GPT-4 to observe.
- **Bot Mode**: GPT-4 can immediately engage in autonomous gameplay or be toggled on to take control, independent of prior context.
- **Flexible Interaction**: Switch between manual and autonomous modes to accommodate different play styles and scenarios.
- **Real-Time Adaptation**: GPT-4 dynamically responds to the game environment, whether in direct control or learning from player inputs.

Requirements
------------
- Python 3.8 or higher
- Required libraries: asyncio, telnetlib3, colorama
- An active OpenAI API key

Setup
-----
1. Clone the repository and install dependencies.
2. Configure API key, HOST, and PORT settings in `autoMUD.py`.

Usage
-----
- Start AutoMUD, choose your mode, and interact with your chosen MUD through GPT-4's advanced capabilities or direct input.

Configuration
-------------
- Easily configurable settings for HOST, PORT, and API key, allowing for personalized gameplay experiences.

Contributing
------------
- Contributions are welcome. Fork the project, commit your changes, and submit a pull request for review.

License
-------
- AutoMUD is available under the MIT License. See the LICENSE file for more details.

Support
-------
- For support, please open an issue on the GitHub project page.

Acknowledgments
---------------
- Special thanks to OpenAI for the GPT-4 technology that powers AutoMUD.