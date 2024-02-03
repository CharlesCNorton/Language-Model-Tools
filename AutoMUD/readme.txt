AutoMUD README
==============

Description
-----------
AutoMUD is an innovative interface that transforms interactions within Multi-User Dungeons (MUDs) through the advanced natural language processing capabilities of OpenAI's GPT-4. This state-of-the-art AI model empowers AutoMUD to seamlessly convert intuitive player commands into precise, game-ready actions, making the complex and richly detailed worlds of MUDs more accessible to both newcomers and seasoned players alike. By leveraging GPT-4, AutoMUD not only simplifies the command input process but also enhances the gaming experience with adaptive learning and personalized interactions, setting a new standard for AI-assisted gaming by merging the depth of MUDs with the simplicity and flexibility of natural language.


Features
--------
- Natural Language Processing: Interprets user inputs into actionable commands for MUDs.
- Automated Gameplay: Makes decisions and performs actions based on the game context.
- Customization: Allows users to customize system messages and commands for a tailored experience.
- Real-Time Interaction: Provides live feedback from the MUD server, including logging and command processing.
- Priority Directives: Enables users to send priority commands for immediate execution.
- Extensible: Designed with modularity in mind, allowing for future enhancements and integrations.

Requirements
------------
- Python 3.8 or newer.
- asyncio for asynchronous I/O operations.
- telnetlib3 for Telnet communication.
- colorama for enhanced terminal output with color support.
- An OpenAI API key for GPT model integration.

Installation
------------
1. Clone the AutoMUD repository:
   git clone [repository URL]

2. Navigate to the AutoMUD directory:
   cd AutoMUD

3. Install required dependencies:
   pip install -r requirements.txt

Usage
-----
1. Configure your OpenAI API key in autoMUD.py by replacing ENTER_API_KEY with your key.
2. Set the HOST and PORT variables in autoMUD.py to your target MUD server.
3. Start AutoMUD:
   python autoMUD.py
4. Use the main menu to start the client, change settings, or exit.
5. Follow on-screen prompts to interact with the MUD through natural language.

Configuration
-------------
- HOST: Define the host address of your target MUD server.
- PORT: Specify the port number for the MUD server connection.
- SYSTEM_MESSAGE: Customize the instructions provided to the GPT model for generating context-appropriate commands.

Contributing
------------
We welcome contributions to AutoMUD! If you have suggestions for improvements or bug fixes, please feel free to fork the repository, make your changes, and submit a pull request. For more detailed information, refer to the CONTRIBUTING.md file in the repository.

License
-------
AutoMUD is licensed under the MIT License. This permissive license allows for reuse within both proprietary and open source software, provided that the license and copyright notice are included with any substantial portion of the software. For the full license text, please see the LICENSE file in the repository.

Support
-------
For questions, issues, or support related to AutoMUD, please open an issue on the GitHub project page. We aim to provide timely and helpful support to all users.

Acknowledgments
---------------
- OpenAI for the GPT model, enabling the core functionality of AutoMUD.
- The vibrant MUD community, whose passion and creativity have been a constant source of inspiration for this project.
