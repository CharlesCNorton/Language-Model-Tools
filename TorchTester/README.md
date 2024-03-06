
# TorchTester

TorchTester is a utility tool designed to help users verify and test their PyTorch installation, CUDA setup, and model training capabilities. It provides a simple yet comprehensive way to ensure that your PyTorch environment is correctly set up, allowing for efficient development and research in machine learning projects.

## Features

TorchTester includes the following key features:

- **Color-Coded Outputs**: Makes use of the `colorama` library to produce easy-to-read, color-coded output, helping you quickly assess the status of each test.
- **PyTorch Installation Test**: Verifies the current version of PyTorch installed in your environment and ensures it's correctly installed.
- **CUDA Availability Check**: Determines whether CUDA is available on your system, identifies the CUDA device, and displays the CUDA and cuDNN versions, aiding in troubleshooting and optimizing GPU-based workflows.
- **Model Training Test**: Executes a simple model training process using the CIFAR10 dataset, verifying that your setup is capable of training neural networks efficiently.
- **Installation Verification**: Generates a random tensor to confirm that PyTorch operations can be executed successfully.

## Installation

TorchTester does not require a separate installation. However, ensure you have the following prerequisites installed:

- Python 3.x
- PyTorch (1.x or later)
- torchvision
- colorama

## Usage

To use TorchTester, simply clone this repository or download the `TorchTester.py` script to your local machine. Then, run the script using Python:

```bash
python TorchTester.py
```

Follow the on-screen menu to select the test you wish to run. The options are:

1. Test PyTorch Installation
2. Test CUDA Availability
3. Test Model Training
4. Verify Installation
5. Exit

## Contributing

Contributions to TorchTester are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

TorchTester is open-sourced under the MIT license. See the LICENSE file for details.

## Acknowledgments

- The PyTorch Team for providing the fantastic deep learning framework.
- The torchvision team for supplying datasets and model architectures for easy access.
- The colorama project for making terminal output colorful and more readable.
