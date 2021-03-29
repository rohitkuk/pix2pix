import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

import imageio
from common.utils import dataset_root ,list_files
from common.utils import image_splitter, input_augmentation, target_augmentation


"""
To Do :
- Prepare a Data Class for Each Dataset, Divide in to A and B
    -- cityspaces
    -- edges2shoes
    -- facades
    -- maps
"""



class Pix2PixDatasets(Dataset):
    def __init__(self, dataset_name, augmentations = True, reverse = False) :
        super(Pix2PixDatasets).__init__()
        self.dataset_name = dataset_name
        self.root_dir = dataset_root(dataset_name)
        self.list_files = list_files(dir_=self.root_dir)
        self.augmentations = augmentations
        self.reverse = reverse

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index) :
        image = imageio.imread(self.list_files[index])
        input_image, target_image   = image_splitter(image, self.dataset_name, self.reverse)  #imageA will always be the input , imageB will always be ground truth
        if self.augmentations:
            # print(input_image.shape)
            input_image = input_augmentation(image = input_image)['image']
            target_image = target_augmentation(image = target_image)['image']
        return input_image, target_image


if __name__ == '__main__':
    Pix2PixDatasets()