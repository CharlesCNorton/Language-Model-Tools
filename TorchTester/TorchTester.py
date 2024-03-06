import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from colorama import Fore, Back, Style

def print_colored(text, color=Fore.WHITE, background=Back.BLACK):
    """Print colored text"""
    print(f"{color}{background}{text}{Style.RESET_ALL}")

def test_pytorch_installation():
    """Test PyTorch installation"""
    print_colored(f"PyTorch Version: {torch.__version__}", Fore.GREEN)

def test_cuda_availability():
    """Test CUDA availability"""
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print_colored("CUDA Available: True", Fore.GREEN)
        print_colored(f"CUDA Device: {torch.cuda.get_device_name(0)}", Fore.YELLOW)
        print_colored(f"CUDA Version: {torch.version.cuda}", Fore.YELLOW)
        print_colored(f"cuDNN Version: {torch.backends.cudnn.version()}", Fore.YELLOW)
    else:
        device = torch.device("cpu")
        print_colored("CUDA Available: False", Fore.RED)

def test_model_training():
    """Test model training"""
    print_colored("Testing Model Training...", Fore.CYAN)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(192 * 8 * 8, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print_colored(f"Epoch {epoch+1} Loss: {running_loss/len(trainloader)}", Fore.YELLOW)

    print_colored("Model Training Complete!", Fore.GREEN)

def verify_installation():
    """Verify PyTorch installation"""
    print_colored("Verifying PyTorch Installation...", Fore.CYAN)
    x = torch.rand(5, 3)
    print_colored(f"Random Tensor:\n{x}", Fore.YELLOW)

def main():
    """Main menu"""
    global device
    test_cuda_availability()

    while True:
        print_colored("PyTorch Testing Menu", Fore.CYAN, Back.WHITE)
        print_colored("1. Test PyTorch Installation", Fore.YELLOW)
        print_colored("2. Test CUDA Availability", Fore.YELLOW)
        print_colored("3. Test Model Training", Fore.YELLOW)
        print_colored("4. Verify Installation", Fore.YELLOW)
        print_colored("5. Exit", Fore.YELLOW)

        choice = input(f"{Fore.MAGENTA}Enter your choice (1-5): {Style.RESET_ALL}")

        if choice == "1":
            test_pytorch_installation()
        elif choice == "2":
            test_cuda_availability()
        elif choice == "3":
            test_model_training()
        elif choice == "4":
            verify_installation()
        elif choice == "5":
            print_colored("Exiting...", Fore.RED)
            break
        else:
            print_colored("Invalid choice. Please try again.", Fore.RED)

if __name__ == "__main__":
    main()
