import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms.functional as F


def visualize_batch(batch, im_per_row=4):
    grid = make_grid(batch, nrow=im_per_row, padding=4)
    grid = F.to_pil_image(grid)
    plt.imshow(np.asarray(grid))
    plt.axis('off')
    plt.show()
