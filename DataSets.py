import torchvision.transforms
from torch.utils.data import Dataset
import glob
from torchvision import transforms
from torchvision.io import read_image


class ShmootDataSet(Dataset):
    """
    base class for shmoot datasets

    The data is separated into images that were originally taken as images
    and images that were extracted from videos. We keep that separation
    because images that were taken from videos are with poor quality, and
    we might want to prioritize original images when sampling.
    Images after the index self.from_im_last_idx are taken from videos.
    """

    dir: str  # dataset dir, have to be defined by subclasses
    im_size: int  # image size, have to be defined by subclasses

    def __init__(self, augment=True, augmentation_transform=None):
        self.from_im_last_idx = len(glob.glob(self.dir + 'FromIm/*')) - 1
        self.total_len = self.from_im_last_idx + len(glob.glob(self.dir + 'FromVid/*')) + 1
        self.augment = augment
        if augmentation_transform is None:
            self.augmentation_transform = transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
                transforms.RandomResizedCrop(size=self.im_size, scale=(0.8, 1.0), ratio=(1, 1)),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
        else:
            self.augmentation_transform = augmentation_transform

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx >= self.total_len:
            raise IndexError('Index out of range')

        if idx <= self.from_im_last_idx:
            path = self.dir + 'FromIm/' + str(idx) + '.jpg'

        else:
            idx = idx - self.from_im_last_idx - 1
            path = self.dir + 'FromVid/' + str(idx) + '.jpg'

        image = read_image(path) / 255.0
        if self.augment:
            image = self.augmentation_transform(image)

        return image


class ShmootDataSet128(ShmootDataSet):
    """ Shmoot dataset with 128x128 images """
    def __init__(self, augment=True, augmentation_transform=None):
        self.dir = './Data/DataSet128/'
        self.im_size = 128
        super().__init__(augment=augment, augmentation_transform=augmentation_transform)


class ShmootDataSet256(ShmootDataSet):
    """ Shmoot dataset with 256x256 images """
    def __init__(self, augment=True, augmentation_transform=None):
        self.dir = './Data/DataSet256/'
        self.im_size = 256
        super().__init__(augment=augment, augmentation_transform=augmentation_transform)