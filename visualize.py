import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST


def main():
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

    print("Enter an index between 0 and 59999: ", end='')
    idx = input()
    plt.imshow(train_set.data[int(idx)], cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
