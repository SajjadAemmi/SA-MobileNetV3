import time
import argparse

import torch
from tqdm import tqdm
from colorama import Fore

from models import mobilenet_v3_large, sa_resnet50
from utils import calc_acc
import dataset
import config


def test():
    parser = argparse.ArgumentParser(description='SA-MobileNetV3 - Test')
    parser.add_argument('--dataset', help="dataset", default='mnist', type=str)
    parser.add_argument('--gpu', help="gpu", default=False, action='store_true')
    parser.add_argument('--weights', help="pre trained weights path", default='./weights/mnist.pt', type=str)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        test_dataloader, dataset_classes = dataset.mnist('test')
    elif args.dataset == 'cfar100':
        test_dataloader, dataset_classes = dataset.cfar100('test')

    num_classes = len(dataset_classes)

    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')
    model = mobilenet_v3_large(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.eval()

    tic = time.time()

    with torch.no_grad():
        test_acc = 0
        for data, labels in tqdm(test_dataloader, desc="Testing"):
            data, labels = data.to(device), labels.to(device)
            preds = model(data)
            test_acc += calc_acc(preds, labels)

        acc = test_acc / len(test_dataloader)
        print(Fore.BLUE, f"[Test Accuracy: {acc}]", Fore.RESET)

    tac = time.time()
    print("Time Taken : ", tac - tic)


if __name__ == "__main__":
    test()
