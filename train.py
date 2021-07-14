import time
import argparse

import torch
from tqdm import tqdm
from colorama import Fore

from model import mobilenet_v3_large
from utils import calc_acc
import dataset
import config


def train():
    parser = argparse.ArgumentParser(description='SA-MobileNetV3 - Train')
    parser.add_argument('--dataset', help="dataset", default='mnist', type=str)
    parser.add_argument('--gpu', help="gpu", default=False, action='store_true')
    args = parser.parse_args()

    train_dataloader, val_dataloader, dataset_classes = dataset.load(args.dataset, 'train', config.validation)
    num_classes = len(dataset_classes)

    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')
    model = mobilenet_v3_large(num_classes=num_classes).to(device)

    tic = time.time()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.decay)

    for epoch in range(1, config.epochs + 1):
        train_loss = 0.0
        train_acc = 0.0
        model.train(True)
        for data, labels in tqdm(train_dataloader, desc="Training"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(data)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_acc += calc_acc(preds, labels)

        total_loss = train_loss / len(train_dataloader)
        total_acc = train_acc / len(train_dataloader)
        print(Fore.GREEN,
              f"Epoch: {epoch}",
              f"[Train Loss: {total_loss}]",
              f"[Train Accuracy: {total_acc}]",
              f"[lr: {optimizer.param_groups[0]['lr']}]",
              Fore.RESET)

        if config.validation:
            model.eval()
            val_acc = 0
            with torch.no_grad():
                for data, labels in tqdm(val_dataloader, desc="Validating"):
                    data, labels = data.to(device), labels.to(device)
                    preds = model(data)
                    val_acc += calc_acc(preds, labels)

                total_acc = val_acc / len(val_dataloader)
                print(Fore.BLUE,
                      f"[Validation Accuracy: {total_acc}]",
                      Fore.RESET)

        scheduler.step()

    tac = time.time()
    print("Time Taken : ", tac - tic)

    torch.save(model.state_dict(), "weights.pth")


if __name__ == "__main__":
    train()
