import time
import argparse

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from colorama import Fore

from models import mobilenet_v3_large, sa_resnet50
import config


def inference():
    parser = argparse.ArgumentParser(description='SA-MobileNetV3 - Predict')
    parser.add_argument('--input', help="input image", default='input/4.png', type=str)
    parser.add_argument('--output', help="output image", default='output/visualization.png', type=str)
    parser.add_argument('--dataset', help="dataset", default='mnist', type=str)
    parser.add_argument('--gpu', help="gpu", default=True, action='store_true')
    parser.add_argument('--weights', help="pre trained weights path", default='./weights/mnist.pt', type=str)
    parser.add_argument('--use-gradcam', help="use gradcam", default=True, action='store_true')
    args = parser.parse_args()

    if args.dataset == 'mnist':
        num_classes = 10
    elif args.dataset == 'cfar100':
        num_classes = 100

    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')
    model = mobilenet_v3_large(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.eval()

    tic = time.time()

    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((config.input_size, config.input_size))
                                    #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #                       std=[0.229, 0.224, 0.225]),
                                    ])
    
    image = cv2.imread(args.input)
    image_tensor = transform(image).unsqueeze(0).to(device)

    output = F.softmax(model(image_tensor), dim=1)
    print(Fore.GREEN, torch.argmax(output), Fore.RESET)

    if args.use_gradcam:
        target_layer = model.features[-1]
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=args.gpu)

        input_tensor = image_tensor.float()
        grayscale_cam = cam(input_tensor=input_tensor, target_category=None, aug_smooth=True)
        grayscale_cam = grayscale_cam[0]

        image = cv2.resize(image, (224, 224))
        image_normal = image / 255.0

        visualization = show_cam_on_image(image_normal, grayscale_cam)
        cv2.imwrite(args.output, visualization)

    tac = time.time()
    print("Time Taken : ", tac - tic)


if __name__ == "__main__":
    inference()
