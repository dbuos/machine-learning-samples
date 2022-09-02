
import matplotlib.pyplot as plt


def plot_images(images_dl):
    batch_imgs, _labels = next(iter(images_dl))
    plt.figure(figsize=(5,5))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(batch_imgs[i].view(28,28), cmap='binary', interpolation='none')
        plt.xticks([])
        plt.yticks([])
