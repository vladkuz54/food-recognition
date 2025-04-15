import os
import requests
import zipfile
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from helper_functions import device, train, plot_loss_curves, pred_and_plot_image
from model import TinyVGG

if __name__ == '__main__':  # Add the main guard to resolve multiprocessing issues
    data_path = Path("data/")
    image_path = data_path / 'pizza_steak_sushi'

    if image_path.is_dir():
        print('The data folder already exists.')
    else:
        image_path.mkdir(parents=True, exist_ok=True)

        with open(data_path / "pizza_steak_sushi.zip", 'wb') as f:
            request = requests.get(
                'https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip')
            f.write(request.content)

        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", 'r') as zip_ref:
            zip_ref.extractall(image_path)

    train_dir = image_path / 'train'
    test_dir = image_path / 'test'

    train_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transform)

    test_data = datasets.ImageFolder(root=train_dir,
                                     transform=test_transform)

    # BATCH_SIZE = 32
    # NUM_WORKERS = os.cpu_count()
    #
    # train_dataloader = DataLoader(dataset=train_data,
    #                               batch_size=BATCH_SIZE,
    #                               shuffle=True,
    #                               num_workers=NUM_WORKERS)
    #
    # test_dataloader = DataLoader(dataset=train_data,
    #                              batch_size=BATCH_SIZE,
    #                              shuffle=False,
    #                              num_workers=NUM_WORKERS)
    #
    # model = TinyVGG(input_shape=3,
    #                 hidden_units=10,
    #                 output_shape=len(train_data.classes)).to(device)
    #
    # NUM_EPOCHS = 15
    #
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params=model.parameters(),
    #                              lr=0.001)
    #
    # model_results = train(model=model,
    #                       train_dataloader=train_dataloader,
    #                       test_dataloader=test_dataloader,
    #                       optimizer=optimizer,
    #                       loss_fn=loss_fn,
    #                       epochs=NUM_EPOCHS)

    MODEL_PATH = Path('models')
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)

    MODEL_NAME = 'food-recognition.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # torch.save(obj=model.state_dict(),
    #            f=MODEL_SAVE_PATH)

    loaded_model = TinyVGG(input_shape=3,
                           hidden_units=10,
                           output_shape=len(train_data.classes))

    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    loaded_model.to(device)

    custom_image_transform = transforms.Compose([
        transforms.Resize(size=(64, 64))
    ])

    custom_image_path = 'a.jpg'

    pred_and_plot_image(model=loaded_model,
                        image_path=custom_image_path,
                        class_names=train_data.classes,
                        transform=custom_image_transform,
                        device=device)
