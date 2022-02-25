import DataSets
from torch.utils.data import Sampler, DataLoader
import random


class ImagePrioritizedSampler(Sampler):
    def __init__(self, from_im_last_idx, total_num_images, prob_from_im=0.5):
        """
        :param from_im_last_idx: The last index of images that were not originally
            from videos
        :param total_num_images: dataset total size
        :param prob_from_im: the probability to choose from the image (idx<from_im_last_idx).

        Note that there are more images that were originally from videos than normal images,
        which means that even with probability of 0.5, normal images are relatively prioritized
        """
        self.from_im_last_idx = from_im_last_idx
        self.total_num_images = total_num_images
        self.prob_from_im = prob_from_im

    def __iter__(self):
        for i in range(self.total_num_images):
            from_original_im = random.uniform(0, 1) < self.prob_from_im
            if from_original_im:
                yield random.randint(0, self.from_im_last_idx)
            else:
                yield random.randint(self.from_im_last_idx, self.total_num_images-1)

    def __len__(self):
        return self.total_num_images


def get_shmoot_dataloader(data_set: DataSets.ShmootDataSet, batch_size, prob_from_im=0.5, num_workers=0):
    sampler = ImagePrioritizedSampler(data_set.from_im_last_idx, data_set.total_len, prob_from_im)
    return DataLoader(data_set, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
