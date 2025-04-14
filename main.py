import os
import requests
import zipfile
from pathlib import Path
import random
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from helper_functions import device, train, pred_and_plot_image
from model import TinyVGG

if __name__ == "__main__":
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    train_transform_trivial = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transform_simple = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    train_data_augmented = datasets.ImageFolder(root=train_dir,
                                                transform=train_transform_trivial)

    test_data_simple = datasets.ImageFolder(root=train_dir,
                                            transform=test_transform_simple)

    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()

    train_dataloader_augmented = DataLoader(dataset=train_data_augmented,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)

    test_dataloader_simple = DataLoader(dataset=train_data_augmented,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)

    model = TinyVGG(input_shape=3,
                    hidden_units=10,
                    output_shape=len(train_data_augmented.classes)).to(device)

    NUM_EPOCHS = 5

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001)

    model_1_results = train(model=model,
                            train_dataloader=train_dataloader_augmented,
                            test_dataloader=test_dataloader_simple,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)

    custom_image_transform = transforms.Compose([
        transforms.Resize(size=(64, 64))
    ])

    custom_image_path = 'pizza.jpg'

    pred_and_plot_image(model=model,
                        image_path=custom_image_path,
                        class_names=train_data_augmented.classes,
                        transform=custom_image_transform,
                        device=device)