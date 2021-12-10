import os
import torch
from torchvision import datasets, transforms
import config


datasets_dir_path = 'datasets'

def gray2rgb(image):
    return image.repeat(3, 1, 1)


def load(name, subset='train', validation=False):
    if name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(gray2rgb),
                                        transforms.Resize((config.input_size, config.input_size)),
                                        ])

        if subset == 'train':
            dataset = datasets.MNIST(datasets_dir_path, train=True, download=True, transform=transform)
        elif subset == 'test':
            dataset = datasets.MNIST(datasets_dir_path, train=False, download=True, transform=transform)
    
    elif name == 'cfar10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Resize((config.input_size, config.input_size))
                                        ])

        if subset == 'train':
            dataset = datasets.CIFAR10(datasets_dir_path, train=True, download=True, transform=transform)
        elif subset == 'test':
            dataset = datasets.CIFAR10(datasets_dir_path, train=False, download=True, transform=transform)

    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Resize((config.input_size, config.input_size))
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        #                      std=[0.229, 0.224, 0.225]),
                                        ])

        if subset == 'train':
            dataset = datasets.ImageFolder(root=os.path.join(datasets_dir_path, name, 'train'), transform=transform)
        elif subset == 'test':
            dataset = datasets.ImageFolder(root=os.path.join(datasets_dir_path, name, 'test'), transform=transform)

    if validation:
        train_dataset_size = int(0.8 * len(dataset))
        val_dataset_size = len(dataset) - train_dataset_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_dataset_size, val_dataset_size])
        dataloader = torch.utils.data.DataLoader(train_set, num_workers=config.num_workers, shuffle=True, batch_size=config.batch_size)
        val_dataloader = torch.utils.data.DataLoader(val_set, num_workers=config.num_workers, shuffle=True, batch_size=config.batch_size)
        print(train_dataset_size, val_dataset_size)

        return dataloader, val_dataloader, dataset.classes

    else:
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=config.num_workers, shuffle=True, batch_size=config.batch_size)
        dataset_size = int(len(dataset))
        print(dataset_size)

        return dataloader, dataset.classes
