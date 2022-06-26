import torch
from pathlib import Path
import src.config.models as C
import src.utils.metrics as M
from src.layers.models import ElegantoNN
from src.training.train import train_model
from src.utils.data import create_loader_from_csv

# Configuration
csv_path = Path("../data/processed_train.csv")
train_validation_split_ratio = 0.8
image_shape = (224, 224)
transform = None
batch_size = 32
shuffle = True
drop_last_batch = True
in_channels = 3
config = C.ELEGANTONN_A
filter_size = 16
lr = 1e-4
betas = (0.9, 0.999)
num_epochs = 30
log_index = 20
metrics = M.__METRIC_DICT__
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result_directory = Path("../results")


def run():
    # Loading all objects
    model = ElegantoNN(in_channels, config, filter_size, image_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr, betas)
    train_loader, val_loader = create_loader_from_csv(csv_path, batch_size, image_shape,
                                                      train_validation_split_ratio,
                                                      transform, shuffle, drop_last_batch)

    # Triggering training function
    train_model(train_loader, val_loader, model,
                optimizer, loss_fn, metrics,
                num_epochs, result_directory,
                device, log_index)


if __name__ == '__main__':
    run()
