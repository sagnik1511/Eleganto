import torch
import seaborn as sns
import matplotlib.pyplot as plt


def show_tensor_image(tensor_image: torch.Tensor):
    plt.figure(figsize=(7, 7))
    plt.imshow(tensor_image.permute(1, 2, 0))
    plt.show()


def plot_dist(train_df, title):
    plt.figure(figsize=(20, 6))
    plt.hist(train_df[f"pix_{title}"],range(256), label=title)
    plt.title(f"Image {title}")
    plt.show()


def plot_interest_dist(train_df, title):
    plt.figure(figsize=(20, 6))
    plt.scatter(train_df[f"pix_{title}"], train_df.Interest)
    plt.title(f"Image {title}")
    plt.show()


def plot_corr_matrix(train_df):
    corr_matrix = train_df[["pix_mean", "pix_median", "pix_mode", "Interest"]].corr()
    plt.figure(figsize=(7, 7))
    sns.heatmap(corr_matrix)
    plt.title("Correlation Matrix")
    plt.show()

