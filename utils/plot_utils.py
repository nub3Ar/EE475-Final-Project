import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def show_images(x, y, num):
    if num % 2 == 1:
        num += 1

    # for batched data
    if x.ndim == 4:
        x = x.squeeze()
        y = y.squeeze()

    images = x[:num]
    labels = y[:num]
    num_row = 2
    num_col = num // 2

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray_r')
        ax.set_title('Label: {}'.format(labels[i]))

    plt.tight_layout()
    plt.show()


def plot_losses(histories, titles, ):
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 4))

    for i in range(n):
        train_loss = histories[i].history['loss']
        val_loss = histories[i].history['val_loss']
        x = np.arange(len(train_loss), )
        axes[i].plot(x, train_loss, color='blue', label='Train loss')
        axes[i].plot(x, val_loss, color='red', label='Validation loss')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(titles[i])
        axes[i].legend()
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


def plot_accuracy(histories, titles, ):
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 4))

    for i in range(n):
        train_loss = histories[i].history['accuracy']
        val_loss = histories[i].history['val_accuracy']
        x = np.arange(len(train_loss), )
        axes[i].plot(x, train_loss, color='blue', label='Train accuracy')
        axes[i].plot(x, val_loss, color='red', label='Validation accuracy')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(titles[i])
        axes[i].legend()
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
