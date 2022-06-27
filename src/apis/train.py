import os
import torch
from pathlib import Path
import src.config.model as C
import src.utils.metrics as M
from torchsummary import summary
from src.layers.models import ElegantoNN
from src.training.trainer import train_model
from src.data.custom_dataset import create_loader_from_csv
from torch.nn import MSELoss, CrossEntropyLoss

# Configuration
problem_type = "clf"
csv_path = Path("dataset/processed_train.csv")
train_validation_split_ratio = 0.8
image_shape = (224, 224)
transform = None
batch_size = 32
shuffle = True
drop_last_batch = True
in_channels = 1
config = C.ELEGANTONN_A_C
filter_size = 16
lr = 1e-4
betas = (0.9, 0.999)
loss_fn = CrossEntropyLoss()
num_epochs = 2
log_index = 20
metrics = M.__CLF_METRIC_DICT__
device = "cuda:0" if torch.cuda.is_available() else "cpu"
result_directory = Path("results")


def run():
    print(f"Current working directory : {os.getcwd()}")
    # Loading all objects
    print("Initializing Model...")
    model = ElegantoNN(in_channels=in_channels, config=config, base_filter_dim=filter_size, input_shape=image_shape)
    print("Model Loaded...")
    print(model)
    summary(model, (in_channels, *image_shape), device="cpu")
    print("Model able to process required outputs...")
    print("Initializing Optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr, betas)
    print("Optimizer Loaded...")
    print(f"Chosen Metrics : {[k for k, _ in metrics.items()]}")
    print("Generating Data Loaders...")
    train_loader, val_loader = create_loader_from_csv(csv_path=csv_path, batch_size=batch_size,
                                                      inp_shape=image_shape, ratio=train_validation_split_ratio,
                                                      augment=transform, shuffle=shuffle,
                                                      drop_last=drop_last_batch, problem_type=problem_type)
    print("DataLoaders generated...")
    # Triggering training function
    train_model(train_loader=train_loader, val_loader=val_loader, model=model,
                optim=optimizer, loss_fn=loss_fn, metrics=metrics,
                num_epochs=num_epochs, result_dir=result_directory,
                device=device, log_index=log_index, problem_type=problem_type)


if __name__ == '__main__':
    run()
