from torchvision import datasets

def main():

    ROOT = './data/cifar100'
    train_dataset = datasets.CIFAR100(root=ROOT, train=True, download=True)

    # Hold-out this data for final evaluation
    valid_dataset = datasets.CIFAR100(root=ROOT, train=False, download=True)

if __name__ == '__main__':

    main()
