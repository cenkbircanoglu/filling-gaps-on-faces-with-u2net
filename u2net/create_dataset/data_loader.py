import random

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFile

from u2net.create_dataset import task
from u2net.create_dataset.image_folder import make_dataset


class CreateDataset(data.Dataset):
    def __init__(self, img_file, mask_file, load_size, fine_size, is_train, resize_or_crop, no_augment, no_flip,
                 no_rotation, mask_type=[0, 1]):
        self.mask_type = mask_type
        self.is_train = is_train
        self.fine_size = fine_size
        self.img_paths, self.img_size = make_dataset(img_file)
        # provides random file for training and testing
        if mask_file != 'none':
            self.mask_paths, self.mask_size = make_dataset(mask_file)
        self.transform = get_transform(load_size, fine_size, is_train, resize_or_crop, no_augment, no_flip, no_rotation)

    def __getitem__(self, index):
        # load image
        img, img_path = self.load_img(index)
        # load mask
        mask = self.load_mask(img, index)
        return {'img': img, 'img_path': img_path, 'mask': mask}

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.img_paths[index % self.img_size]
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path

    def load_mask(self, img, index):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.mask_type) - 1)
        mask_type = self.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return task.center_mask(img)

        # random regular mask
        if mask_type == 1:
            return task.random_regular_mask(img)

        # random irregular mask
        if mask_type == 2:
            return task.random_irregular_mask(img)

        # external mask from "Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV18)"
        if mask_type == 3:
            if self.is_train:
                mask_index = random.randint(0, self.mask_size - 1)
            else:
                mask_index = index
            mask_pil = Image.open(self.mask_paths[mask_index]).convert('RGB')
            size = mask_pil.size[0]
            if size > mask_pil.size[1]:
                size = mask_pil.size[1]
            mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(10),
                                                 transforms.CenterCrop([size, size]),
                                                 transforms.Resize(self.fine_size),
                                                 transforms.ToTensor()
                                                 ])
            mask = (mask_transform(mask_pil) == 0).float()
            mask_pil.close()
            return mask


def dataloader(img_file, mask_file, load_size, fine_size, is_train, resize_or_crop, no_augment, no_flip,
               no_rotation, mask_type, batch_size, no_shuffle, n_threads):
    datasets = CreateDataset(img_file, mask_file, load_size, fine_size, is_train, resize_or_crop, no_augment, no_flip,
                             no_rotation, mask_type)
    dataset = data.DataLoader(datasets, batch_size=batch_size, shuffle=not no_shuffle)

    return dataset


def get_transform(load_size, fine_size, is_train, resize_or_crop, no_augment, no_flip, no_rotation):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [load_size[0], load_size[1]]
    fsize = [fine_size[0], fine_size[1]]
    if is_train:
        if resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Resize(osize))
            transform_list.append(transforms.RandomCrop(fsize))
        elif resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(fsize))
        if not no_augment:
            transform_list.append(transforms.ColorJitter(0.0, 0.0, 0.0, 0.0))
        if not no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if not no_rotation:
            transform_list.append(transforms.RandomRotation(3))
    else:
        transform_list.append(transforms.Resize(fsize))

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)
