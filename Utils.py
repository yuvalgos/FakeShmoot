import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms.functional as F


def make_image_grid(images, im_per_row=4):
    grid = make_grid(images, nrow=im_per_row, padding=2, pad_value=255)
    return grid


def visualize_batch(batch, im_per_row=4):
    grid = make_image_grid(batch, im_per_row=im_per_row)
    grid = F.to_pil_image(grid)
    plt.imshow(np.asarray(grid))
    plt.axis('off')
    plt.show()
