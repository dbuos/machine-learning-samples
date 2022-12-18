import matplotlib.pyplot as plt
import numpy as np


def show_random_images(dataset, num_imgs=10):
    assert num_imgs % 2 == 0, "num_imgs must be even"
    rand_idx = np.random.randint(0, len(dataset)-(num_imgs+1))
    fig, ax = plt.subplots(2, num_imgs//2, figsize=(20, 10))
    for i in range(2):
        for j in range(num_imgs//2):
            ax[i, j].imshow(dataset[i*(num_imgs//2)+j+rand_idx][0])
            ax[i, j].axis('off')
    plt.show()



def select_sample_images(ds, attributes_map, attr_name, attr_value, num=10):
    ds_idxs = np.arange(0, len(ds))
    np.random.shuffle(ds_idxs)
    attr_idx = attributes_map[attr_name]
    selected = []
    idx = 0
    while len(selected) < num:
        img, attr = ds[ds_idxs[idx]]
        if attr[attr_idx] == attr_value:
            selected.append(img)
        idx += 1
    return selected    


def show_images(images):
    fig, ax = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(2):
        for j in range(5):
            ax[i, j].imshow(images[i*5+j])
            ax[i, j].axis('off')
    plt.show()

