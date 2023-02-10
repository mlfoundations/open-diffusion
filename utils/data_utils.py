from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from utils.logging import Path

import math


def batchify(list_of_examples, batch_size):
    # Simple lazy batchification of an Iterable object
    num_batches = math.ceil(len(list_of_examples) / batch_size)
    i = 0
    for _ in range(num_batches):
        yield list_of_examples[i:i + batch_size]
        i += batch_size


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)

    return has_caption and has_image


def filter_no_cls_or_no_image(sample):
    has_caption = ('cls' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)

    return has_caption and has_image


class ImageGlobDataset(Dataset):
    def __init__(self, path, extension="jpg", transform=None, target_transform=None) -> None:
        super().__init__()

        self.path = path
        self.extension = extension
        self.transform = transform
        self.target_transform = target_transform

        self._process_image_dir()
    
    def _process_image_dir(self):
        self.imgs = list(Path(self.path).glob(f"**/*.{self.extension}"))

    def __getitem__(self, i):
        path = self.imgs[i]
        img = default_loader(path)

        if self.transform is not None:
            img = self.transform(img)
        
        path = str(path)
        if self.target_transform is not None:
            target = self.target_transform(path)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def set_transform(self, new_transform):
        self.transform = new_transform
