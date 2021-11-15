import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from random import choice

import jsonlines
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.clip.clip import tokenize

random.seed(a=42)


class YCupDataset(Dataset):
    def __init__(self,
                 metadata_file, images_directory,
                 transforms):
        logging.info(f'Loading metadata file from {metadata_file}.')

        self.images = []
        self.captions = []
        self.images_directory = Path(images_directory)
        available_data_list = set([int(x.stem) for x in self.images_directory.glob("*.png")])
        counter = 0
        with jsonlines.open(metadata_file) as reader:
            for obj in reader:
                if obj['image'] in available_data_list:
                    try:
                        self.captions.append(obj['queries'])
                        self.images.append(obj['image'])
                    except IndexError:
                        pass
                    counter += 1
                if counter == 1000:
                    break

        self.transforms = transforms

        logging.info('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.transforms(
            Image.open((self.images_directory / f'{int(self.images[idx]):07d}').with_suffix('.png')))
        text = tokenize([str(choice(self.captions[idx]))])[0]
        return image, text


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def get_ycup_dataset(args, preprocess_fn, is_train):
    metadata_file = args.metadata_file
    images_directory = Path(args.images_directory)

    assert metadata_file
    dataset = YCupDataset(
        metadata_file,
        images_directory,
        preprocess_fn,
    )
    num_samples = len(dataset)
    logging.info(f'num samples {num_samples}')
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    logging.info(f'Sampler type: {type(sampler)}, dist {args.distributed}, is_train {is_train}')
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(dataset_type):
    if dataset_type == 'ycup':
        return get_ycup_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.metadata_file:
        data["train"] = get_dataset_fn(args.dataset_type)(
            args, preprocess_train, is_train=True)

    return data
