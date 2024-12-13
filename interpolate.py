#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2023
#


import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer


def main():
    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')

    args = argParser.parse_args()

    save_file = None
    if args.s != None:
        save_file = args.s
    bottleneck_size = 0
    if args.z != None:
        bottleneck_size = args.z

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = train_transform

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    N_input = 28 * 28  # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    n = input("Define number of interpolation steps > ")
    idx1 = input('Enter Index > ')
    idx2 = input('Enter Index > ')

    n = int(n)
    idx1 = int(idx1)
    idx2 = int(idx2)

    if 0 <= idx1 <= train_set.data.size()[0] and 0 <= idx2 <= train_set.data.size()[0]:
        img1 = train_set.data[idx1]
        img2 = train_set.data[idx2]

        img1 = img1.type(torch.float32)
        img2 = img2.type(torch.float32)

        img1 = (img1 - torch.min(img1)) / torch.max(img1)
        img2 = (img2 - torch.min(img2)) / torch.max(img2)

        img1 = img1.to(device=device)
        img2 = img2.to(device=device)

        img1 = img1.view(1, img1.shape[0] * img1.shape[1]).type(torch.FloatTensor)
        img2 = img2.view(1, img2.shape[0] * img2.shape[1]).type(torch.FloatTensor)

        with torch.no_grad():
            bottleneck1 = model.encode(img1)
            bottleneck2 = model.encode(img2)

        img1 = model.decode(bottleneck1).detach()
        img1 = img1.view(28, 28).type(torch.FloatTensor)

        img2 = model.decode(bottleneck2).detach()
        img2 = img2.view(28, 28).type(torch.FloatTensor)

        f = plt.figure()
        f.add_subplot(1, n + 2, 1)
        plt.imshow(img1, cmap='gray')

        steps = []
        for i in range(bottleneck1.size(1)):
            steps.append((bottleneck2[0][i].item() - bottleneck1[0][i].item()) / (n+1))

        for i in range(n):
            for j in range(bottleneck1.size(1)):
                bottleneck1[0][j] = bottleneck1[0][j].item() + steps[j]

            output = model.decode(bottleneck1).detach()
            output = output.view(28, 28).type(torch.FloatTensor)

            f.add_subplot(1, n + 2, i + 2)
            plt.imshow(output, cmap='gray')

        f.add_subplot(1, n + 2, n + 2)
        plt.imshow(img2, cmap='gray')

        plt.show()







###################################################################

if __name__ == '__main__':
    main()