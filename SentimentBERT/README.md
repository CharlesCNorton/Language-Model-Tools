
# SentimentBERT

SentimentBERT is a Python-based machine learning application designed to perform sentiment analysis on textual data using the BERT (Bidirectional Encoder Representations from Transformers) model. This project illustrates the remarkable capabilities of AI-assisted programming, demonstrating that complex, functional scripts for natural language processing (NLP) and machine learning can be assembled rapidly. Notably, the initial version of this project, including the first model training on the I...

## Project Overview

- **Rapid Development:** Built in 3 hours with the aid of AI tools.
- **High Performance:** Achieved an F1 score of ~0.90 on the IMDB reviews dataset.
- **Technology Stack:** Utilizes Python, PyTorch, Transformers, and several other libraries.

## Features

- **Dataset Management:** Automates the downloading and preparation of the IMDB reviews sentiment analysis dataset.
- **Model Training:** Supports training and evaluating sentiment analysis models using pre-trained BERT models.
- **Hyperparameter Tuning:** Allows automatic training with a variety of hyperparameter settings to optimize performance.
- **Sentiment Prediction:** Enables users to predict the sentiment of new text inputs.
- **Model Persistence:** Provides functionality to save and load trained models for further use.

## Installation

To set up SentimentBERT, ensure you have Python 3.x installed and then run the following commands to install necessary packages:
```bash
pip install pandas torch transformers datasets sklearn colorama tkinter
```

## Usage

Start the program using:
```bash
python SentimentBERT.py
```
Follow the interactive command-line menu to manage datasets, train models, and predict text sentiment. The menu includes:
- **1. Download and prepare the IMDB reviews dataset**
- **2. Train new model with custom hyperparameters**
- **3. Auto-train model with different hyperparameters**
- **4. Predict sentiment of new text**
- **5. Save current best model**
- **6. Load model**
- **7. Exit**

## Contributing

Contributions are welcome. Please fork the repository, make your changes, and submit a pull request. Ensure your contributions adhere to the project's code of conduct.

## License

This project is released under the MIT License. For more details, see the `LICENSE` file.

## Acknowledgments

Thanks to AI tools such as OpenAI's GPT-4 and Anthropic's Claude for assisting in the rapid development and optimization of this project.

## Creator

- **phanerozoic**

Enjoy using SentimentBERT for your sentiment analysis needs across various types of textual data!
