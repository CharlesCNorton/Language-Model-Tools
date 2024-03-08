# InfernoLM: Enhanced Language Model Interaction

Welcome to InfernoLM, an advanced language model inferencing tool designed to bridge the gap between powerful AI language models and end-users through a versatile command-line interface. By leveraging cutting-edge technologies such as OpenAI's Whisper for audio input and pyttsx3 for text-to-speech capabilities, InfernoLM offers an immersive experience for interacting with AI in real-time.

## Key Features

- **Dual Interaction Modes**: Choose between typing commands or speaking directly to InfernoLM for a truly interactive experience.
- **Voice to Text via Whisper**: Seamlessly transcribe your voice into text commands using the Whisper model, making interactions more natural and accessible.
- **Text-to-Speech Output**: Hear responses directly from your device with integrated TTS, providing a conversational feel to interactions.
- **Flexible Model Support**: Easily switch between different models to suit your needs, with support for custom model paths.
- **Optimized for Performance**: Utilizes CUDA for GPU acceleration (where available) to ensure fast and efficient processing.
- **Dynamic Model Loading/Unloading**: Manage resources effectively by loading and unloading models on demand.
- **Color-Coded CLI Feedback**: Enhance your CLI experience with Colorama-powered, color-coded feedback for easier navigation and interaction.

## Upcoming Enhancements

- **Customizable EOS Tokens**: Future versions will introduce the ability to customize end-of-sentence tokens, offering finer control over the behavior and output of generated text.

## Installation Guide

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or later
- Required Python libraries: PyTorch, Transformers, Whisper, pyttsx3, sounddevice, scipy, colorama, keyboard

### Setup Instructions

1. Clone the InfernoLM repository to your local machine:
```bash
git clone https://github.com/yourusername/InfernoLM.git
```

2. Navigate to the InfernoLM directory:
```bash
cd InfernoLM
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## How to Use InfernoLM

After installation, you can start InfernoLM with the following command:
```bash
python inferno_lm.py
```

Follow the prompts to select your model path, enable TTS, and choose your preferred interaction mode. InfernoLM supports a rich set of commands for dynamic interaction, including loading models, toggling TTS, and switching between text and voice modes.

## Contributing

We welcome contributions to InfernoLM, whether it's through feature suggestions, bug reports, or pull requests. Please feel free to contribute in any way that you can. For major changes, please open an issue first to discuss what you would like to change.

## License

InfernoLM is released under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- Thanks to OpenAI for the Transformers and Whisper models that power the core functionality of InfernoLM.
- Appreciation to the developers of the pyttsx3, sounddevice, and other libraries that make this tool possible.

---

Thank you for exploring InfernoLM. Dive in and discover the endless possibilities of interacting with AI through InfernoLM!
